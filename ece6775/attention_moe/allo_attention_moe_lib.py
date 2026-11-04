# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Attention + MoE in Allo
# Q, K, V -> Attention -> MoE -> Output
# uses library functions (nn.linear2d, nn.GeLU) for expert

import numpy as np
import allo
import allo.library.nn as allo_nn
from allo.library.nn import linear2d, GeLU  # Direct import for type inference
from allo.ir.types import float32, int32
from allo import dsl
import sys
import os

# path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
llm_config_dir = os.path.join(project_root, "llm_config")
if llm_config_dir not in sys.path:
    sys.path.insert(0, llm_config_dir)

MODE = "sw_emu"  # llvm, sw_emu, hw_emu, hw, csyn

from llm_config import DEFAULT_CONFIG_MODE, get_moe_config, print_config_info

CONFIG_MODE = DEFAULT_CONFIG_MODE


def softmax_1d[Ty, N, K](X: "Ty[N, K]") -> "Ty[N, K]":
    """
    Softmax over last dimension for [N, K] shape.

    Args:
        X: Input tensor of shape [N, K]
    Returns:
        Z: Output tensor of shape [N, K] with softmax applied over dimension K
    """
    Z: Ty[N, K]
    E_exp: Ty[N, K]
    M: Ty[N] = -1000000000000.0
    S: Ty[N] = 0.0

    # Find max for each row (over dimension K)
    for n, k in dsl.grid(N, K, name="row_max"):
        if X[n, k] > M[n]:
            M[n] = X[n, k]

    # Compute exp and sum
    for n, k in dsl.grid(N, K, name="exp_sum"):
        E_exp[n, k] = dsl.exp(X[n, k] - M[n])
        S[n] += E_exp[n, k]

    # Normalize
    for n, k in dsl.grid(N, K, name="update"):
        # Add small epsilon to prevent division by zero
        Z[n, k] = E_exp[n, k] / (S[n])

    return Z


# ----------------------------------------------------------------------------------
# Softmax 2D: Softmax over last dimension for [L, L] shape (for attention)
# ----------------------------------------------------------------------------------
def softmax_2d[Ty, L](X: "Ty[L, L]") -> "Ty[L, L]":
    """
    Softmax over last dimension for [L, L] shape.
    Used for attention scores.

    Args:
        X: Input tensor of shape [L, L]
    Returns:
        Z: Output tensor of shape [L, L] with softmax applied over last dimension
    """
    Z: Ty[L, L]
    E_exp: Ty[L, L]
    M: Ty[L] = -1000000000000.0
    S: Ty[L] = 0.0

    # Find max for each row
    for i, j in dsl.grid(L, L, name="row_max"):
        if X[i, j] > M[i]:
            M[i] = X[i, j]

    # Compute exp and sum
    for i, j in dsl.grid(L, L, name="exp_sum"):
        E_exp[i, j] = dsl.exp(X[i, j] - M[i])
        S[i] += E_exp[i, j]

    # Normalize
    for i, j in dsl.grid(L, L, name="update"):
        Z[i, j] = E_exp[i, j] / S[i]

    return Z


# ----------------------------------------------------------------------------------
# Scaled Dot-Product Attention: Custom implementation matching PyTorch
# ----------------------------------------------------------------------------------
def scaled_dot_product_attention[
    Ty, H, L, D
](Q: "Ty[L, D]", K: "Ty[L, D]", V: "Ty[L, D]") -> "Ty[L, D]":
    """
    Scaled Dot-Product Attention (Multi-Head Attention).

    This implementation matches the PyTorch version:
    - Input: Q, K, V of shape [L, D]
    - Split into H heads, each with dimension D // H
    - For each head: softmax(QK^T / sqrt(head_dim)) @ V
    - Merge heads back to [L, D]
    - Uses linear2d for matrix multiplication (replacing systolic for HLS compatibility)

    Args:
        Q: Query tensor of shape [L, D]
        K: Key tensor of shape [L, D]
        V: Value tensor of shape [L, D]

    Returns:
        Z: Output tensor of shape [L, D]
    """
    Z: Ty[L, D] = 0.0

    # Compute scale factor: 1 / sqrt(head_dim) = 1 / sqrt(D // H)
    # For D=8, H=2: head_dim=4, scale = 1/sqrt(4) = 0.5
    scale: Ty = 1.0 / dsl.sqrt(float(D // H))

    # Process each head
    for h in range(H, name="head_loop"):
        # Split Q, K, V for this head
        Q_h: Ty[L, D // H] = 0.0
        K_h: Ty[L, D // H] = 0.0  # Not transposed, will transpose for linear2d
        V_h: Ty[L, D // H] = 0.0

        for i, j in dsl.grid(L, D // H, name="split_qkv"):
            Q_h[i, j] = Q[i, h * (D // H) + j]
            K_h[i, j] = K[i, h * (D // H) + j]  # Store K normally (not transposed)
            V_h[i, j] = V[i, h * (D // H) + j]

        # Compute QK^T = [L, D//H] @ [D//H, L] = [L, L]
        # Using linear2d: Q_h @ K_h^T = linear2d(Q_h, K_h_transposed, 0)
        # linear2d computes X @ W^T + b, so we need K_h^T as W
        # K_h is [L, D//H], so K_h^T is [D//H, L], but linear2d expects W[N, K] = [L, D//H]
        # So we need to transpose K_h to [D//H, L], then use it as W[L, D//H] (which is K_h^T)
        # Actually: Q_h[L, D//H] @ K_h^T[D//H, L] = Q_h @ (K_h^T)
        # linear2d(X[M, K], W[N, K], b) computes X @ W^T, so:
        # - X = Q_h [L, D//H] -> M=L, K=D//H
        # - W should be [L, D//H] to get W^T = [D//H, L]
        # - So W = K_h^T, but we have K_h [L, D//H], so we need to transpose it
        K_h_T: Ty[D // H, L] = 0.0
        for i, j in dsl.grid(L, D // H, name="transpose_K"):
            K_h_T[j, i] = K_h[i, j]

        # Now use linear2d: Q_h[L, D//H] @ K_h_T^T[D//H, L] = Q_h @ K_h_T^T
        # But linear2d expects W[N, K], so we need W[L, D//H] such that W^T = [D//H, L]
        # Actually, we can use K_h directly as W[L, D//H], then W^T = [D//H, L] = K_h^T
        # Wait, let me reconsider: linear2d(X[M, K], W[N, K], b) = X @ W^T
        # For QK^T = Q_h[L, D//H] @ K_h^T[D//H, L]:
        # - X = Q_h [L, D//H] -> M=L, K=D//H
        # - We want result [L, L], so N=L
        # - W should be [L, D//H] such that W^T = [D//H, L] = K_h^T
        # - So W = K_h (which is [L, D//H]), then W^T = [D//H, L] = K_h^T ✓
        zero_bias: Ty[L] = 0.0
        Y = linear2d[Ty, Ty, Ty, L, L, D // H](Q_h, K_h, zero_bias)  # Q_h @ K_h^T

        # Scale by 1/sqrt(head_dim)
        Y_scaled: Ty[L, L] = 0.0
        for i, j in dsl.grid(L, L, name="scale"):
            Y_scaled[i, j] = Y[i, j] * scale

        # Apply softmax over last dimension
        S: Ty[L, L] = 0.0
        E_exp: Ty[L, L] = 0.0
        M: Ty[L] = -1000000000000.0
        Sum: Ty[L] = 0.0

        # Find max for each row
        for i, j in dsl.grid(L, L, name="softmax_max"):
            if Y_scaled[i, j] > M[i]:
                M[i] = Y_scaled[i, j]

        # Compute exp and sum
        for i, j in dsl.grid(L, L, name="softmax_exp"):
            E_exp[i, j] = dsl.exp(Y_scaled[i, j] - M[i])
            Sum[i] += E_exp[i, j]

        # Normalize
        for i, j in dsl.grid(L, L, name="softmax_norm"):
            S[i, j] = E_exp[i, j] / Sum[i]

        # Compute S @ V_h = [L, L] @ [L, D//H] = [L, D//H]
        # Using linear2d: S @ V_h = linear2d(S, V_h_transposed, 0)
        # linear2d(X[M, K], W[N, K], b) = X @ W^T
        # For S[L, L] @ V_h[L, D//H]:
        # - X = S [L, L] -> M=L, K=L
        # - We want result [L, D//H], so N=D//H
        # - W should be [D//H, L] such that W^T = [L, D//H] = V_h
        # - So W = V_h^T (which is [D//H, L]), then W^T = [L, D//H] = V_h ✓
        V_h_T: Ty[D // H, L] = 0.0
        for i, j in dsl.grid(L, D // H, name="transpose_V"):
            V_h_T[j, i] = V_h[i, j]

        zero_bias_v: Ty[D // H] = 0.0
        C_h = linear2d[Ty, Ty, Ty, L, D // H, L](S, V_h_T, zero_bias_v)  # S @ V_h

        # Merge back to Z
        for i, j in dsl.grid(L, D // H, name="merge_heads"):
            Z[i, h * (D // H) + j] = C_h[i, j]

    return Z


# ----------------------------------------------------------------------------------
# Top1 selection: Top-k selection for k=1 (argmax)
# ----------------------------------------------------------------------------------
def top1_select[Ty, N, E](logits: "Ty[N, E]") -> "int32[N]":
    """
    Select top-1 expert (argmax) for each token.

    Args:
        logits: Input logits of shape [N, E]
    Returns:
        indices: Top-1 expert indices of shape [N]
    """
    indices: int32[N]
    max_val: Ty[N] = -1000000000000.0

    # Initialize indices and max_val with first expert
    for n in range(N, name="init"):
        indices[n] = 0
        max_val[n] = logits[n, 0]

    # Find argmax for each token (search from index 1 onwards)
    for n, e in dsl.grid(N, E, name="argmax"):
        if e > 0:  # Skip e=0 (already initialized)
            if logits[n, e] > max_val[n]:
                max_val[n] = logits[n, e]
                indices[n] = e

    return indices


# ----------------------------------------------------------------------------------
# Expert: A simple feed-forward expert network
# ----------------------------------------------------------------------------------
def expert[
    Ty, N, D_in, D_hidden, D_out
](
    x: "Ty[N, D_in]",
    fc1_weight: "Ty[D_hidden, D_in]",
    fc1_bias: "Ty[D_hidden]",
    fc2_weight: "Ty[D_out, D_hidden]",
    fc2_bias: "Ty[D_out]",
) -> "Ty[N, D_out]":
    """
    A simple feed-forward expert network.

    This implementation uses linear2d and GeLU from Allo library
    for optimized computation.

    Args:
        x: Input tensor of shape [N, D_in]
        fc1_weight: First linear layer weights of shape [D_hidden, D_in]
        fc1_bias: First linear layer bias of shape [D_hidden]
        fc2_weight: Second linear layer weights of shape [D_out, D_hidden]
        fc2_bias: Second linear layer bias of shape [D_out]
    Returns:
        output: Output tensor of shape [N, D_out]
    """
    # Step 1: First linear layer using linear2d from library
    fc1_out = linear2d[Ty, Ty, Ty, N, D_hidden, D_in](x, fc1_weight, fc1_bias)

    # Step 2: GELU activation using GeLU from library
    gelu_out = GeLU[Ty, N, D_hidden](fc1_out)

    # Step 3: Second linear layer using linear2d from library
    fc2_out = linear2d[Ty, Ty, Ty, N, D_out, D_hidden](gelu_out, fc2_weight, fc2_bias)

    return fc2_out


# ----------------------------------------------------------------------------------
# MoE Layer: Main MoE layer that routes tokens to experts
# ----------------------------------------------------------------------------------
def moe_layer[
    Ty, N, D_in, D_out, E, K, D_hidden
](
    x: "Ty[N, D_in]",
    # Gate weights
    gate_weight: "Ty[E, D_in]",
    gate_bias: "Ty[E]",
    # Expert weights (E experts, each with 2 linear layers)
    expert_fc1_weights: "Ty[E, D_hidden, D_in]",
    expert_fc1_biases: "Ty[E, D_hidden]",
    expert_fc2_weights: "Ty[E, D_out, D_hidden]",
    expert_fc2_biases: "Ty[E, D_out]",
) -> "Ty[N, D_out]":
    """
    Mixture of Experts layer for inference.

    Args:
        x: Input tensor of shape [N, D_in] (already flattened from [B, L, D_in])
        gate_weight: Gate weight matrix of shape [E, D_in]
        gate_bias: Gate bias vector of shape [E]
        expert_fc1_weights: Expert FC1 weights of shape [E, D_hidden, D_in]
        expert_fc1_biases: Expert FC1 biases of shape [E, D_hidden]
        expert_fc2_weights: Expert FC2 weights of shape [E, D_out, D_hidden]
        expert_fc2_biases: Expert FC2 biases of shape [E, D_out]
    Returns:
        output: Output tensor of shape [N, D_out]
    """
    # Step 1: Compute gate logits using linear2d
    gate_logits = linear2d[Ty, Ty, Ty, N, E, D_in](x, gate_weight, gate_bias)  # [N, E]

    # Step 2: Select top-1 expert using top1_select function
    top1_indices_1d: int32[N] = top1_select[Ty, N, E](gate_logits)

    # Step 3: Get top-k logits and apply softmax
    top_k_logits: Ty[N, K] = 0.0
    for n, k in dsl.grid(N, K, name="topk_logits"):
        expert_idx = top1_indices_1d[n] if k == 0 else 0  # For k=1, K=1
        top_k_logits[n, k] = gate_logits[n, expert_idx]

    # Step 4: Apply softmax to top-k logits using softmax_1d function
    top_k_weights = softmax_1d[Ty, N, K](top_k_logits)  # [N, K]

    # Step 5: Create sparse weight matrix from top-k weights
    top_k_indices: int32[N, K] = 0
    gate_weights: Ty[N, E] = 0.0

    for n in range(N, name="gate_weights"):
        # Store top-k indices (for k=1)
        top_k_indices[n, 0] = top1_indices_1d[n]
        # Set gate weights from softmax output
        expert_idx = top1_indices_1d[n]
        gate_weights[n, expert_idx] = top_k_weights[n, 0]  # For k=1, K=1

    # Step 6: Process each expert: compute outputs for all tokens
    expert_outputs: Ty[E, N, D_out] = 0.0

    for e in range(E, name="expert_loop"):
        # Extract expert weights for this expert
        expert_fc1_w: Ty[D_hidden, D_in] = 0.0
        expert_fc1_b: Ty[D_hidden] = 0.0
        expert_fc2_w: Ty[D_out, D_hidden] = 0.0
        expert_fc2_b: Ty[D_out] = 0.0

        for d_hidden, d_in in dsl.grid(D_hidden, D_in, name="extract_fc1_w"):
            expert_fc1_w[d_hidden, d_in] = expert_fc1_weights[e, d_hidden, d_in]

        for d_hidden in range(D_hidden, name="extract_fc1_b"):
            expert_fc1_b[d_hidden] = expert_fc1_biases[e, d_hidden]

        for d_out, d_hidden in dsl.grid(D_out, D_hidden, name="extract_fc2_w"):
            expert_fc2_w[d_out, d_hidden] = expert_fc2_weights[e, d_out, d_hidden]

        for d_out in range(D_out, name="extract_fc2_b"):
            expert_fc2_b[d_out] = expert_fc2_biases[e, d_out]

        # Process all tokens through this expert using the expert function
        expert_out = expert[Ty, N, D_in, D_hidden, D_out](
            x, expert_fc1_w, expert_fc1_b, expert_fc2_w, expert_fc2_b
        )  # [N, D_out]

        # Store expert outputs
        for n, d_out in dsl.grid(N, D_out, name="store_expert_out"):
            expert_outputs[e, n, d_out] = expert_out[n, d_out]

    # Step 7: Combine expert outputs using gate weights
    output: Ty[N, D_out] = 0.0
    for n, e, d_out in dsl.grid(N, E, D_out, name="combine_outputs"):
        weight: Ty = gate_weights[n, e]
        output[n, d_out] += expert_outputs[e, n, d_out] * weight

    return output


# ----------------------------------------------------------------------------------
# Attention + MoE Layer: Combined layer
# Data flow: Q, K, V -> Attention -> MoE -> Output
# ----------------------------------------------------------------------------------
def attention_moe_layer[
    Ty, B, L, D, H, E, TopK, D_hidden
](
    Query: "Ty[B, L, D]",
    Key: "Ty[B, L, D]",
    Value: "Ty[B, L, D]",
    # Gate weights
    gate_weight: "Ty[E, D]",
    gate_bias: "Ty[E]",
    # Expert weights (E experts, each with 2 linear layers)
    expert_fc1_weights: "Ty[E, D_hidden, D]",
    expert_fc1_biases: "Ty[E, D_hidden]",
    expert_fc2_weights: "Ty[E, D, D_hidden]",
    expert_fc2_biases: "Ty[E, D]",
) -> "Ty[B, L, D]":
    """
    Combined Attention + MoE layer.

    Data flow: Q, K, V -> Attention -> MoE -> Output

    Args:
        Query: Query tensor of shape [B, L, D]
        Key: Key tensor of shape [B, L, D]
        Value: Value tensor of shape [B, L, D]
        gate_weight: Gate weight matrix of shape [E, D]
        gate_bias: Gate bias vector of shape [E]
        expert_fc1_weights: Expert FC1 weights of shape [E, D_hidden, D]
        expert_fc1_biases: Expert FC1 biases of shape [E, D_hidden]
        expert_fc2_weights: Expert FC2 weights of shape [E, D, D_hidden]
        expert_fc2_biases: Expert FC2 biases of shape [E, D]
    Returns:
        output: Output tensor of shape [B, L, D]
    """
    # Output tensor
    output: Ty[B, L, D] = 0.0

    # Process each batch item
    for b in range(B, name="batch_loop"):
        # Step 1: Extract Q, K, V for this batch item -> [L, D]
        Q_b: Ty[L, D] = 0.0
        K_b: Ty[L, D] = 0.0
        V_b: Ty[L, D] = 0.0

        for l, d in dsl.grid(L, D, name="extract_qkv"):
            Q_b[l, d] = Query[b, l, d]
            K_b[l, d] = Key[b, l, d]
            V_b[l, d] = Value[b, l, d]

        # Step 2: Apply custom scaled_dot_product_attention
        # scaled_dot_product_attention[Ty, H, L, D](Q, K, V) -> [L, D]
        attn_out = scaled_dot_product_attention[Ty, H, L, D](Q_b, K_b, V_b)  # [L, D]

        # Step 3: Apply MoE layer
        # moe_layer expects [N, D_in], so we use N=L for single batch
        moe_out = moe_layer[Ty, L, D, D, E, TopK, D_hidden](
            attn_out,
            gate_weight,
            gate_bias,
            expert_fc1_weights,
            expert_fc1_biases,
            expert_fc2_weights,
            expert_fc2_biases,
        )  # [L, D]

        # Step 4: Store output for this batch item
        for l, d in dsl.grid(L, D, name="store_output"):
            output[b, l, d] = moe_out[l, d]

    return output


# ==================================================================================
# Schedule optimization function
# ==================================================================================
def optimize_attention_moe_with_composition(
    batch_size, seq_len, embed_dim, num_heads, num_experts, k, hidden_dim
):
    """
    Create optimized schedules for Attention + MoE and compose them together.

    Args:
        batch_size: Batch size (B)
        seq_len: Sequence length (L)
        embed_dim: Embedding dimension (D)
        num_heads: Number of attention heads (H)
        num_experts: Number of experts (E)
        k: Top-k value (currently only k=1 is supported)
        hidden_dim: Hidden dimension for experts

    Returns:
        s_attn_moe: Optimized schedule for attention_moe_layer with all sub-schedules composed
    """
    Ty = float32
    B = batch_size
    L = seq_len
    D = embed_dim
    H = num_heads
    E = num_experts
    K = k
    D_hidden = hidden_dim

    print("=" * 60)
    print("Creating and optimizing Attention + MoE schedules...")
    print("=" * 60)

    # Step 1: Create schedule for top1_select
    print("\n[1] Creating schedule for top1_select...")
    s_top1 = allo.customize(top1_select, instantiate=[Ty, L, E])
    print("  - Created top1_select schedule for [L, E]")

    # Step 2: Create schedule for softmax_1d (for MoE gate)
    print("\n[2] Creating schedule for softmax_1d...")
    s_softmax_1d = allo.customize(softmax_1d, instantiate=[Ty, L, K])
    print("  - Created softmax_1d schedule for [L, K]")

    # Step 3: Create schedule for expert
    print("\n[3] Creating schedule for expert...")

    # Create schedules for library functions that expert uses
    print("  - Creating schedules for library functions (linear2d, GeLU)...")
    s_linear_fc1 = allo.customize(
        allo_nn.linear2d, instantiate=[Ty, Ty, Ty, L, D_hidden, D]
    )
    s_gelu = allo.customize(allo_nn.GeLU, instantiate=[Ty, L, D_hidden])
    s_linear_fc2 = allo.customize(
        allo_nn.linear2d, instantiate=[Ty, Ty, Ty, L, D, D_hidden]
    )
    print("  - Library function schedules created")

    # Create expert schedule
    print("  - Creating expert schedule...")
    s_expert = allo.customize(expert, instantiate=[Ty, L, D, D_hidden, D])

    # Compose library function schedules
    s_expert.compose(s_linear_fc1, id="expert_fc1")
    s_expert.compose(s_gelu, id="expert_gelu")
    s_expert.compose(s_linear_fc2, id="expert_fc2")
    print("  - Created expert schedule")
    print("  - Composed nn.linear2d and nn.GeLU schedules for expert")

    # Step 4: Create schedule for moe_layer
    print("\n[4] Creating schedule for moe_layer...")
    s_moe = allo.customize(moe_layer, instantiate=[Ty, L, D, D, E, K, D_hidden])

    # Compose gate linear
    s_gate_linear = allo.customize(allo_nn.linear2d, instantiate=[Ty, Ty, Ty, L, E, D])
    s_moe.compose(s_gate_linear, id="gate")
    s_moe.compose(s_top1)
    s_moe.compose(s_softmax_1d)
    s_moe.compose(s_expert)
    print("  - Created moe_layer schedule")

    # Step 5: Create schedule for custom attention
    print("\n[5] Creating schedule for custom scaled_dot_product_attention...")
    s_attn = allo.customize(scaled_dot_product_attention, instantiate=[Ty, H, L, D])
    print("  - Created scaled_dot_product_attention schedule")

    # Step 6: Create schedule for main attention_moe_layer function
    print("\n[6] Creating schedule for attention_moe_layer...")
    s_attn_moe = allo.customize(
        attention_moe_layer, instantiate=[Ty, B, L, D, H, E, K, D_hidden]
    )

    # Step 7: Compose all schedules together
    print("\n[7] Composing all schedules together...")
    s_attn_moe.compose(s_attn)
    s_attn_moe.compose(s_moe)
    print("  - Composed scaled_dot_product_attention schedule")
    print("  - Composed moe_layer schedule")

    print("\n" + "=" * 60)
    print("Schedule composition complete!")
    print("=" * 60)

    return s_attn_moe


# ==================================================================================
# Test function to compare Allo and PyTorch implementations
# ==================================================================================

if __name__ == "__main__":
    import torch
    import torch.nn as torch_nn
    from pytorch_attention_moe import AttentionMoE

    # ============================================================================
    # Configuration parameters - use shared config from llm_config.py
    # ============================================================================
    # Get MoE configuration from shared config
    moe_config = get_moe_config(CONFIG_MODE)

    batch_size = moe_config["batch_size"]
    seq_len = moe_config["seq_len"]
    embed_dim = moe_config["input_dim"]  # D, must be divisible by num_heads
    num_experts = moe_config["num_experts"]  # E
    k = moe_config["k"]  # Top-k MoE
    hidden_dim = moe_config["hidden_dim"]  # D_hidden

    # Attention-specific parameter: num_heads
    # Choose num_heads such that embed_dim is divisible by num_heads
    # Common choices: 2, 4, 8, 12, 16 (depending on embed_dim)
    if embed_dim >= 768:
        num_heads = 12  # Standard for BERT-base
    elif embed_dim >= 512:
        num_heads = 8
    elif embed_dim >= 256:
        num_heads = 8
    elif embed_dim >= 128:
        num_heads = 4
    elif embed_dim >= 64:
        num_heads = 4
    else:
        num_heads = 2

    # Ensure embed_dim is divisible by num_heads
    while embed_dim % num_heads != 0 and num_heads > 1:
        num_heads -= 1

    seed = 42

    print("=" * 60)
    print("Attention + MoE Allo Implementation Test")
    print("=" * 60)
    print(f"Configuration Mode: {CONFIG_MODE}")
    print(f"Configuration:")
    print(f"  batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")
    print(f"  num_heads={num_heads}, head_dim={embed_dim // num_heads}")
    print(f"  num_experts={num_experts}, k={k}, hidden_dim={hidden_dim}")
    print(f"  Seed: {seed}")
    print("=" * 60)

    # ----------------------------------------------------------------------------------
    # Run PyTorch implementation to get weights and outputs
    # ----------------------------------------------------------------------------------
    print("\n[1] Running PyTorch implementation...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create PyTorch AttentionMoE layer
    pytorch_model = AttentionMoE(
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_experts=num_experts,
        k=k,
        expert_hidden_dim=hidden_dim,
    )
    pytorch_model.eval()

    # Initialize with Xavier uniform
    for param in pytorch_model.parameters():
        if param.dim() > 1:
            torch_nn.init.xavier_uniform_(param)
        else:
            torch_nn.init.zeros_(param)

    # Create random inputs (Q, K, V)
    torch.manual_seed(seed)
    Q_pt = torch.randn(batch_size, seq_len, embed_dim)
    K_pt = torch.randn(batch_size, seq_len, embed_dim)
    V_pt = torch.randn(batch_size, seq_len, embed_dim)

    # Run PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(Q_pt, K_pt, V_pt, verbose=False)

    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(
        f"PyTorch output range: [{pytorch_output.min().item():.6f}, {pytorch_output.max().item():.6f}]"
    )

    # ----------------------------------------------------------------------------------
    # Extract weights and biases from PyTorch model
    # ----------------------------------------------------------------------------------
    print("\n[2] Extracting weights from PyTorch model...")

    # Gate weights
    gate_weight_pt = (
        pytorch_model.moe.gate.gate_linear.weight.data
    )  # [num_experts, embed_dim]
    gate_bias_pt = pytorch_model.moe.gate.gate_linear.bias
    if gate_bias_pt is not None:
        gate_bias_pt = gate_bias_pt.data
    else:
        gate_bias_pt = torch.zeros(num_experts)

    # Expert weights
    expert_fc1_weights_pt = torch.stack(
        [exp.fc1.weight.data for exp in pytorch_model.moe.experts]
    )
    expert_fc1_biases_pt = torch.stack(
        [exp.fc1.bias.data for exp in pytorch_model.moe.experts]
    )
    expert_fc2_weights_pt = torch.stack(
        [exp.fc2.weight.data for exp in pytorch_model.moe.experts]
    )
    expert_fc2_biases_pt = torch.stack(
        [exp.fc2.bias.data for exp in pytorch_model.moe.experts]
    )

    # ----------------------------------------------------------------------------------
    # Convert to numpy arrays
    # ----------------------------------------------------------------------------------
    print("\n[3] Converting weights to numpy arrays...")
    Q_np = np.ascontiguousarray(Q_pt.detach().numpy(), dtype=np.float32)
    K_np = np.ascontiguousarray(K_pt.detach().numpy(), dtype=np.float32)
    V_np = np.ascontiguousarray(V_pt.detach().numpy(), dtype=np.float32)

    gate_weight_np = np.ascontiguousarray(
        gate_weight_pt.detach().numpy(), dtype=np.float32
    )
    gate_bias_np = np.ascontiguousarray(gate_bias_pt.detach().numpy(), dtype=np.float32)

    expert_fc1_weights_np = np.ascontiguousarray(
        expert_fc1_weights_pt.detach().numpy(), dtype=np.float32
    )
    expert_fc1_biases_np = np.ascontiguousarray(
        expert_fc1_biases_pt.detach().numpy(), dtype=np.float32
    )
    expert_fc2_weights_np = np.ascontiguousarray(
        expert_fc2_weights_pt.detach().numpy(), dtype=np.float32
    )
    expert_fc2_biases_np = np.ascontiguousarray(
        expert_fc2_biases_pt.detach().numpy(), dtype=np.float32
    )

    print(f"Q shape: {Q_np.shape}")
    print(f"K shape: {K_np.shape}")
    print(f"V shape: {V_np.shape}")
    print(f"Gate weight shape: {gate_weight_np.shape}")
    print(f"Expert FC1 weights shape: {expert_fc1_weights_np.shape}")
    print(f"Expert FC2 weights shape: {expert_fc2_weights_np.shape}")

    # ----------------------------------------------------------------------------------
    # Run Allo implementation
    # ----------------------------------------------------------------------------------
    print("\n[4] Running Allo implementation...")
    try:
        # Create optimized schedule with composition
        allo_schedule = optimize_attention_moe_with_composition(
            batch_size, seq_len, embed_dim, num_heads, num_experts, k, hidden_dim
        )

        # Generate project name
        project_name = f"allo_attention_moe_lib_{CONFIG_MODE}.prj"
        print(f"Using project name: {project_name}")

        # Build module
        print("\n[5] Building Allo module...")
        if MODE == "llvm":
            mod = allo_schedule.build(target="llvm")
        elif MODE == "sw_emu":
            mod = allo_schedule.build(
                target="vitis_hls", mode="sw_emu", project=project_name
            )
        elif MODE == "hw_emu":
            mod = allo_schedule.build(
                target="vitis_hls", mode="hw_emu", project=project_name
            )
        elif MODE == "hw":
            mod = allo_schedule.build(
                target="vitis_hls", mode="hw", project=project_name
            )
        elif MODE == "csyn":
            mod = allo_schedule.build(
                target="vitis_hls", mode="csyn", project=project_name
            )
        else:
            raise ValueError(f"Unsupported mode: {MODE}")

        # Run Allo inference
        print("\n[6] Running Allo inference...")
        if MODE == "llvm":
            allo_output = mod(
                Q_np,
                K_np,
                V_np,
                gate_weight_np,
                gate_bias_np,
                expert_fc1_weights_np,
                expert_fc1_biases_np,
                expert_fc2_weights_np,
                expert_fc2_biases_np,
            )
        elif MODE in ["sw_emu", "hw_emu", "hw"]:
            allo_output = np.zeros((batch_size, seq_len, embed_dim), dtype=np.float32)
            mod(
                Q_np,
                K_np,
                V_np,
                gate_weight_np,
                gate_bias_np,
                expert_fc1_weights_np,
                expert_fc1_biases_np,
                expert_fc2_weights_np,
                expert_fc2_biases_np,
                allo_output,
            )
        elif MODE == "csyn":
            allo_output = np.zeros((batch_size, seq_len, embed_dim), dtype=np.float32)
            mod()
        else:
            raise ValueError(f"Unsupported mode: {MODE}")

        print(f"Allo output shape: {allo_output.shape}")
        print(f"Allo output range: [{allo_output.min():.6f}, {allo_output.max():.6f}]")

        # ----------------------------------------------------------------------------------
        # Compare the Allo and PyTorch outputs
        # ----------------------------------------------------------------------------------
        print("\n[7] Comparing outputs...")
        pytorch_output_np = pytorch_output.detach().numpy()

        # Compute differences
        diff = np.abs(allo_output - pytorch_output_np)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        rel_diff = np.mean(diff / (np.abs(pytorch_output_np) + 1e-8))

        print(f"Mean absolute difference: {mean_diff:.6e}")
        print(f"Max absolute difference: {max_diff:.6e}")
        print(f"Mean relative difference: {rel_diff:.6e}")

        # Check if outputs are close
        atol = 5e-4
        rtol = 2e-3
        is_close = np.allclose(allo_output, pytorch_output_np, atol=atol, rtol=rtol)

        if is_close:
            print(
                f"\n✓ SUCCESS: Allo output matches PyTorch output (atol={atol}, rtol={rtol})"
            )
        else:
            print(
                f"\n✗ WARNING: Allo output differs from PyTorch output (atol={atol}, rtol={rtol})"
            )
            print("First few differences:")
            print(diff.flatten()[:10])

        # ----------------------------------------------------------------------------------
        # Print sample outputs for comparison
        # ----------------------------------------------------------------------------------
        print("\n[8] Sample outputs (first token, first 5 dimensions):")
        print(f"PyTorch: {pytorch_output_np[0, 0, :5]}")
        print(f"Allo:    {allo_output[0, 0, :5]}")
        print(f"Diff:    {diff[0, 0, :5]}")

    except Exception as e:
        print(f"\n✗ ERROR: Failed to run Allo implementation: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 60)
