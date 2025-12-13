# Attention + MoE in Allo
# Q, K, V -> Attention -> MoE -> Output
# custom implementations (no library functions) for full control
# optimizations: fused GeLU, row-level dataflow, unrolled loops, array partitioning

import numpy as np
import allo
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

MODE = "csyn"  # llvm, sw_emu, hw_emu, hw, csyn

from llm_config import DEFAULT_CONFIG_MODE, get_moe_config, print_config_info

CONFIG_MODE = DEFAULT_CONFIG_MODE
def softmax_1d[Ty, N, K](X: "Ty[N, K]") -> "Ty[N, K]":
    # softmax over last dim
    Z: Ty[N, K]
    E_exp: Ty[N, K]
    M: Ty[N] = -1000000000000.0
    S: Ty[N] = 0.0
    
    # find max per row
    for n, k in dsl.grid(N, K, name="row_max"):
        if X[n, k] > M[n]:
            M[n] = X[n, k]
    
    # exp and sum
    for n, k in dsl.grid(N, K, name="exp_sum"):
        E_exp[n, k] = dsl.exp(X[n, k] - M[n])
        S[n] += E_exp[n, k]
    
    # normalize
    for n, k in dsl.grid(N, K, name="normalize"):
        Z[n, k] = E_exp[n, k] / S[n]
    
    return Z

# softmax for attention scores [L, L]
def softmax_2d[Ty, L](X: "Ty[L, L]") -> "Ty[L, L]":
    Z: Ty[L, L]
    E_exp: Ty[L, L]
    M: Ty[L] = -1000000000000.0
    S: Ty[L] = 0.0
    
    # find max per row
    for i, j in dsl.grid(L, L, name="row_max"):
        if X[i, j] > M[i]:
            M[i] = X[i, j]
    
    # exp and sum
    for i, j in dsl.grid(L, L, name="exp_sum"):
        E_exp[i, j] = dsl.exp(X[i, j] - M[i])
        S[i] += E_exp[i, j]
    
    # normalize
    for i, j in dsl.grid(L, L, name="normalize"):
        Z[i, j] = E_exp[i, j] / S[i]
    
    return Z

# scaled dot-product attention (multi-head)
def scaled_dot_product_attention[Ty, H, L, D](
    Q: "Ty[L, D]",
    K: "Ty[L, D]",
    V: "Ty[L, D]"
) -> "Ty[L, D]":
    # scaled dot-product attention (multi-head)
    
    # matches pytorch: split into H heads, compute attention per head, merge
    Z: Ty[L, D] = 0.0
    
    # scale factor: 1/sqrt(head_dim)
    scale: Ty = 1.0 / dsl.sqrt(float(D // H))
    
    # process each head
    for h in range(H, name="head_loop"):
        # split Q, K, V for this head
        Q_h: Ty[L, D // H] = 0.0
        K_h: Ty[L, D // H] = 0.0
        V_h: Ty[L, D // H] = 0.0
        
        for i, j in dsl.grid(L, D // H, name="split_qkv"):
            Q_h[i, j] = Q[i, h * (D // H) + j]
            K_h[i, j] = K[i, h * (D // H) + j]
            V_h[i, j] = V[i, h * (D // H) + j]
        
        # QK^T with accumulator to break read-after-write dependency
        Y: Ty[L, L] = 0.0
        for i in range(L, name="qkt_i"):
            for j in range(L, name="qkt_j"):
                acc: Ty = 0.0
                for k in range(D // H, name="qkt_k"):
                    acc += Q_h[i, k] * K_h[j, k]  # K_h[j, k] = K_h^T[k, j]
                Y[i, j] = acc
        
        # scale
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
        # Loop order (i, k, j) for better memory access and pipelining:
        # - Sequential access to V_h[k, j] as j changes (inner loop)
        # - Better pipelining of reduction loop k
        C_h: Ty[L, D // H] = 0.0
        for i in range(L, name="sv_i"):
            for k in range(L, name="sv_k"):
                for j in range(D // H, name="sv_j"):
                    C_h[i, j] += S[i, k] * V_h[k, j]
        
        # Merge back to Z
        for i, j in dsl.grid(L, D // H, name="merge_heads"):
            Z[i, h * (D // H) + j] = C_h[i, j]
    
    return Z

# top-1 selection (argmax)
def top1_select[Ty, N, E](logits: "Ty[N, E]") -> "int32[N]":
    # pick best expert per token
    indices: int32[N]
    max_val: Ty[N] = -1000000000000.0
    
    for n in range(N, name="init"):
        indices[n] = 0
        max_val[n] = logits[n, 0]
    
    for n, e in dsl.grid(N, E, name="argmax"):
        if e > 0:  # skip e=0
            if logits[n, e] > max_val[n]:
                max_val[n] = logits[n, e]
                indices[n] = e
    
    return indices

# FFN expert with fused GeLU
# optimizations: fuse GeLU into FC1, row-level dataflow, unroll reduction loops
def expert[Ty, N, D_in, D_hidden, D_out](
    x: "Ty[N, D_in]",
    fc1_weight: "Ty[D_hidden, D_in]",
    fc1_bias: "Ty[D_hidden]",
    fc2_weight: "Ty[D_out, D_hidden]",
    fc2_bias: "Ty[D_out]"
) -> "Ty[N, D_out]":
    """
    A simple feed-forward expert network with fused GeLU and row-level dataflow.
    
    Optimizations applied:
    1. GeLU fused into FC1: Compute GeLU immediately after each FC1 element
       to eliminate intermediate array storage and enable streaming
    2. Row-level processing: Outer loop over rows (N) enables dataflow between
       FC1+GeLU and FC2 stages - while FC2 processes row n, FC1 can process row n+1
    3. Separate reduction loops enable unroll pragmas
    
    Args:
        x: Input tensor of shape [N, D_in]
        fc1_weight: First linear layer weights of shape [D_hidden, D_in]
        fc1_bias: First linear layer bias of shape [D_hidden]
        fc2_weight: Second linear layer weights of shape [D_out, D_hidden]
        fc2_bias: Second linear layer bias of shape [D_out]
    Returns:
        output: Output tensor of shape [N, D_out]
    """
    output: Ty[N, D_out] = 0.0
    
    # Row-level processing: outer loop over rows (tokens)
    # This structure enables dataflow between FC1+GeLU and FC2
    for n in range(N, name="row_loop"):
        # =====================================================================
        # Stage 1: FC1 + Fused GeLU (produces one row of hidden activations)
        # =====================================================================
        # Intermediate buffer for this row's hidden activations (after GeLU)
        hidden_row: Ty[D_hidden] = 0.0
        
        # Compute FC1 output for each hidden dimension, fuse GeLU immediately
        for h in range(D_hidden, name="fc1_gelu_loop"):
            # FC1: dot product x[n,:] @ fc1_weight[h,:] + fc1_bias[h]
            acc: Ty = 0.0
            for k in range(D_in, name="fc1_reduce"):
                acc += x[n, k] * fc1_weight[h, k]
            fc1_val: Ty = acc + fc1_bias[h]
            
            # Fused GeLU: apply immediately to avoid storing fc1_out array
            # GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            # This matches PyTorch's GELU implementation (tanh approximation)
            # sqrt(2/π) ≈ 0.7978845608028654
            x3: Ty = fc1_val * fc1_val * fc1_val
            inner: Ty = 0.7978845608028654 * (fc1_val + 0.044715 * x3)
            tanh_val: Ty = dsl.tanh(inner)
            gelu_val: Ty = 0.5 * fc1_val * (1.0 + tanh_val)
            hidden_row[h] = gelu_val
        
        # =====================================================================
        # Stage 2: FC2 (consumes hidden row, produces output row)
        # =====================================================================
        for o in range(D_out, name="fc2_loop"):
            # FC2: dot product hidden_row @ fc2_weight[o,:] + fc2_bias[o]
            acc2: Ty = 0.0
            for k in range(D_hidden, name="fc2_reduce"):
                acc2 += hidden_row[k] * fc2_weight[o, k]
            output[n, o] = acc2 + fc2_bias[o]
    
    return output

# MoE layer - routes tokens to experts
def moe_layer[Ty, N, D_in, D_out, E, K, D_hidden](
    x: "Ty[N, D_in]",
    # Gate weights
    gate_weight: "Ty[E, D_in]",
    gate_bias: "Ty[E]",
    # Expert weights (E experts, each with 2 linear layers)
    expert_fc1_weights: "Ty[E, D_hidden, D_in]",
    expert_fc1_biases: "Ty[E, D_hidden]",
    expert_fc2_weights: "Ty[E, D_out, D_hidden]",
    expert_fc2_biases: "Ty[E, D_out]"
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
    # =========================================================================
    # Step 1: Compute gate logits (custom linear, no library)
    # gate_logits[n, e] = x[n, :] @ gate_weight[e, :] + gate_bias[e]
    # =========================================================================
    gate_logits: Ty[N, E] = 0.0
    for n, e in dsl.grid(N, E, name="gate_linear"):
        acc: Ty = 0.0
        for k in range(D_in, name="gate_reduce"):
            acc += x[n, k] * gate_weight[e, k]
        gate_logits[n, e] = acc + gate_bias[e]
    
    # =========================================================================
    # Step 2: Select top-1 expert using top1_select function
    # =========================================================================
    top1_indices_1d: int32[N] = top1_select[Ty, N, E](gate_logits)
    
    # =========================================================================
    # Step 3: Get top-k logits and apply softmax
    # =========================================================================
    top_k_logits: Ty[N, K] = 0.0
    for n, k in dsl.grid(N, K, name="topk_logits"):
        expert_idx = top1_indices_1d[n] if k == 0 else 0  # For k=1, K=1
        top_k_logits[n, k] = gate_logits[n, expert_idx]
    
    # =========================================================================
    # Step 4: Apply softmax to top-k logits using softmax_1d function
    # =========================================================================
    top_k_weights = softmax_1d[Ty, N, K](top_k_logits)  # [N, K]
    
    # =========================================================================
    # Step 5: Create sparse weight matrix from top-k weights
    # =========================================================================
    gate_weights: Ty[N, E] = 0.0
    for n in range(N, name="sparse_gate"):
        expert_idx = top1_indices_1d[n]
        gate_weights[n, expert_idx] = top_k_weights[n, 0]  # For k=1, K=1
    
    # =========================================================================
    # Step 6: Process each expert: compute outputs for all tokens
    # =========================================================================
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
        
        # Process all tokens through this expert (uses optimized expert function)
        expert_out = expert[Ty, N, D_in, D_hidden, D_out](
            x, expert_fc1_w, expert_fc1_b, expert_fc2_w, expert_fc2_b
        )  # [N, D_out]
        
        # Store expert outputs
        for n, d_out in dsl.grid(N, D_out, name="store_expert_out"):
            expert_outputs[e, n, d_out] = expert_out[n, d_out]
    
    # =========================================================================
    # Step 7: Combine expert outputs using gate weights
    # =========================================================================
    output: Ty[N, D_out] = 0.0
    for n, e, d_out in dsl.grid(N, E, D_out, name="combine_outputs"):
        weight: Ty = gate_weights[n, e]
        output[n, d_out] += expert_outputs[e, n, d_out] * weight
    
    return output

#----------------------------------------------------------------------------------
# Attention + MoE Layer: Combined layer
# Data flow: Q, K, V -> Attention -> MoE -> Output
#----------------------------------------------------------------------------------
def attention_moe_layer[Ty, B, L, D, H, E, TopK, D_hidden](
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
    expert_fc2_biases: "Ty[E, D]"
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
            expert_fc2_biases
        )  # [L, D]
        
        # Step 4: Store output for this batch item
        for l, d in dsl.grid(L, D, name="store_output"):
            output[b, l, d] = moe_out[l, d]
    
    return output


#==================================================================================
# Schedule optimization function with HLS pragmas
#==================================================================================
def optimize_attention_moe_with_composition(
    batch_size, seq_len, embed_dim, num_heads, num_experts, k, hidden_dim
):
    """
    Create optimized schedules for Attention + MoE and compose them together.
    
    Optimization strategy:
    1. Pipeline innermost loops for throughput
    2. Unroll small loops for parallelism
    3. Partition arrays for parallel access
    
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
    head_dim = D // H
    
    print("=" * 60)
    print("Creating and optimizing Attention + MoE schedules...")
    print("=" * 60)
    
    # =========================================================================
    # Step 1: Create and optimize schedule for top1_select
    # =========================================================================
    print("\n[1] Creating schedule for top1_select...")
    s_top1 = allo.customize(top1_select, instantiate=[Ty, L, E])
    # Pipeline the inner loop for finding argmax
    # Note: Use "function_name:loop_var" format for dsl.grid loops
    s_top1.pipeline("top1_select:e")
    print("  - Created top1_select schedule with pipeline optimization")
    
    # =========================================================================
    # Step 2: Create and optimize schedule for softmax_1d (for MoE gate)
    # =========================================================================
    print("\n[2] Creating schedule for softmax_1d...")
    s_softmax_1d = allo.customize(softmax_1d, instantiate=[Ty, L, K])
    # Pipeline softmax loops using get_loops() to get loop handles
    loops_softmax = s_softmax_1d.get_loops(s_softmax_1d.top_func_name)
    s_softmax_1d.pipeline(loops_softmax["row_max"]["k"])
    s_softmax_1d.pipeline(loops_softmax["exp_sum"]["k"])
    s_softmax_1d.pipeline(loops_softmax["normalize"]["k"])
    print("  - Created softmax_1d schedule with pipeline optimization")
    
    # =========================================================================
    # Step 3: Create and optimize schedule for expert
    # Optimizations:
    #   - Dataflow on row_loop (pipeline FC1+GeLU and FC2 stages)
    #   - Unroll on fc1_reduce and fc2_reduce loops
    #   - Array partitioning for parallel access
    # =========================================================================
    print("\n[3] Creating schedule for expert (custom, no library functions)...")
    s_expert = allo.customize(expert, instantiate=[Ty, L, D, D_hidden, D])
    
    # Get loop handles for expert
    expert_loops = s_expert.get_loops(s_expert.top_func_name)
    print(f"  - Available loops: {list(expert_loops.loops.keys())}")
    
    # Get nested loops inside row_loop
    row_loop = expert_loops["row_loop"]
    print(f"  - row_loop sub-loops: {list(row_loop.loops.keys())}")
    
    # -------------------------------------------------------------------------
    # Note: Allo flattens nested loops and uses loop variable names as keys
    # row_loop sub-loops: ['n', 'h', 'k', 'o'] where:
    #   n = row index (from row_loop)
    #   h = hidden dim index (from fc1_gelu_loop) 
    #   k = reduction index (shared by fc1_reduce and fc2_reduce)
    #   o = output dim index (from fc2_loop)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Optimization 1: Pipeline the output dimension loops (h and o)
    # This pipelines both the FC1+GeLU loop and FC2 loop
    # -------------------------------------------------------------------------
    s_expert.pipeline(row_loop["h"])  # Pipeline FC1+GeLU hidden dim loop
    s_expert.pipeline(row_loop["o"])  # Pipeline FC2 output dim loop
    print("  - Applied pipeline to h (fc1_gelu) and o (fc2) loops")
    
    # -------------------------------------------------------------------------
    # Optimization 2: Unroll reduction loop for parallel MACs
    # The k loop is shared between FC1 and FC2 reductions
    # -------------------------------------------------------------------------
    # Determine unroll factor based on dimensions
    unroll_factor = min(4, D, D_hidden)
    
    s_expert.unroll(row_loop["k"], factor=unroll_factor)
    print(f"  - Applied unroll to k (reduction loop) factor={unroll_factor}")
    
    # -------------------------------------------------------------------------
    # Note: Removed explicit array partitioning
    # Let HLS infer partitioning from pipelining directives for better tool runtime
    # Explicit partitioning can explode synthesis time and may not be optimal
    # -------------------------------------------------------------------------
    
    print("  - Created expert schedule with pipeline and unroll optimizations")
    print("  - Note: Array partitioning inferred by HLS from pipelining")
    
    # =========================================================================
    # Step 4: Create and optimize schedule for moe_layer
    # Optimizations:
    #   - Pipeline on gate_linear and combine_outputs
    #   - Unroll on gate_reduce for parallel access
    #   - Compose optimized expert schedule
    # =========================================================================
    print("\n[4] Creating schedule for moe_layer...")
    s_moe = allo.customize(moe_layer, instantiate=[Ty, L, D, D, E, K, D_hidden])
    
    # Get loop handles for moe_layer
    moe_loops = s_moe.get_loops(s_moe.top_func_name)
    print(f"  - Available top-level loops: {list(moe_loops.loops.keys())}")
    
    # -------------------------------------------------------------------------
    # Optimize gate_linear: pipeline and unroll
    # Note: Allo flattens loops - gate_linear sub-loops are likely ['n', 'e', 'k']
    # -------------------------------------------------------------------------
    gate_linear_loop = moe_loops["gate_linear"]
    print(f"  - gate_linear sub-loops: {list(gate_linear_loop.loops.keys())}")
    
    # Pipeline the e loop (output dimension) and unroll k (reduction)
    s_moe.pipeline(gate_linear_loop["e"])
    print("  - Applied pipeline to gate_linear:e")
    
    unroll_factor_gate = min(4, D)
    s_moe.unroll(gate_linear_loop["k"], factor=unroll_factor_gate)
    print(f"  - Applied unroll to gate_linear:k (factor={unroll_factor_gate})")
    
    # -------------------------------------------------------------------------
    # Optimize other loops in moe_layer
    # Note: For dsl.grid loops, sub-loops use variable names (n, e, k, d_out, etc.)
    # -------------------------------------------------------------------------
    combine_loop = moe_loops["combine_outputs"]
    print(f"  - combine_outputs sub-loops: {list(combine_loop.loops.keys())}")
    s_moe.pipeline(combine_loop["d_out"])
    
    topk_loop = moe_loops["topk_logits"]
    print(f"  - topk_logits sub-loops: {list(topk_loop.loops.keys())}")
    s_moe.pipeline(topk_loop["k"])
    
    sparse_loop = moe_loops["sparse_gate"]
    print(f"  - sparse_gate sub-loops: {list(sparse_loop.loops.keys())}")
    s_moe.pipeline(sparse_loop["n"])
    print("  - Applied pipeline to combine_outputs:d_out, topk_logits:k, sparse_gate:n")
    
    # -------------------------------------------------------------------------
    # Compose sub-function schedules
    # -------------------------------------------------------------------------
    s_moe.compose(s_top1)
    s_moe.compose(s_softmax_1d)
    s_moe.compose(s_expert)
    print("  - Composed top1_select, softmax_1d, and expert schedules")
    print("  - Created moe_layer schedule with pipeline, unroll optimizations")
    
    # =========================================================================
    # Step 5: Create and optimize schedule for scaled_dot_product_attention
    # =========================================================================
    print("\n[5] Creating schedule for custom scaled_dot_product_attention...")
    s_attn = allo.customize(scaled_dot_product_attention, instantiate=[Ty, H, L, D])
    
    # Get loop handles for attention
    attn_loops = s_attn.get_loops(s_attn.top_func_name)
    print(f"  - Available top-level loops: {list(attn_loops.loops.keys())}")
    
    # Helper function to print loop hierarchy recursively
    def print_loop_hierarchy(loops, indent=0):
        for key, handle in loops.loops.items():
            print(" " * indent + f"- {key}")
            # Check if handle has children (is a loop wrapper)
            if hasattr(handle, "loops"):
                print_loop_hierarchy(handle, indent + 2)

    print("  - Full loop hierarchy:")
    print_loop_hierarchy(attn_loops, indent=4)
    
    # Attention has nested loops inside head_loop
    # The structure is: head_loop -> h -> [split_qkv, qkt_matmul, scale, softmax_*, sv_matmul, merge_heads]
    head_inner = attn_loops["head_loop"]
    # print(f"  - head_loop sub-loops: {list(head_inner.loops.keys())}")

    # Get loop handles for sv_matmul and qkt_matmul BEFORE applying global pipeline
    # Global pipeline using string selectors might modify loop structure/metadata
    # sv_loops = head_inner["sv_matmul"]
    # qkt_loops = head_inner["qkt_matmul"]
    # print(f"    - sv_matmul sub-loops: {list(sv_loops.loops.keys())}")
    # print(f"    - qkt_matmul sub-loops: {list(qkt_loops.loops.keys())}")
    
    # Pipeline reduction loops (k) for matmuls - this is critical for II=1
    # The reordered loops (i, k, j) allow better pipelining of k reduction
    s_attn.pipeline("scaled_dot_product_attention:k")  # Pipeline k reduction loops (qkt_k, sv_k)
    s_attn.pipeline("scaled_dot_product_attention:j")  # Pipeline j loops for output dimension
    s_attn.pipeline("scaled_dot_product_attention:i")  # Pipeline i loops (sv_i) to overlap row processing
    print("  - Applied pipeline to i (row), k (reduction), and j (output) loops")
    print("  - Note: Loop reordering (i,k,j) enables better memory access and pipelining")
    print("  - Pipelining sv_i allows overlapping processing of different output rows")

    # -------------------------------------------------------------------------
    # Apply unroll optimization to break loop-carried dependency
    # Unroll k loop by factor of 4 to allow 4 multiplications in parallel
    # -------------------------------------------------------------------------
    # print("  - Applying unroll optimization...")
    # s_attn.unroll(sv_loops["k"], factor=4)
    # print("    - Applied unroll to sv_matmul:k (factor=4)")
    # s_attn.unroll(qkt_loops["k"], factor=4)
    # print("    - Applied unroll to qkt_matmul:k (factor=4)")
    
    # -------------------------------------------------------------------------
    # Apply pipeline optimization to reduction loops
    # -------------------------------------------------------------------------
    # print("  - Applying pipeline to reduction loops...")
    # s_attn.pipeline(sv_loops["k"])
    # print("    - Applied pipeline to sv_matmul:k (reduction loop)")
    # s_attn.pipeline(qkt_loops["k"])
    # print("    - Applied pipeline to qkt_matmul:k (reduction loop)")
    
    # -------------------------------------------------------------------------
    # Apply reorder optimization for better data locality
    # For matrix multiplication C[i,j] += A[i,k] * B[k,j], reorder to i,k,j
    # This improves memory access pattern for B (V_h in sv_matmul)
    # -------------------------------------------------------------------------
    # print("  - Applying reorder optimization...")
    # s_attn.reorder(sv_loops["k"], sv_loops["j"])
    # print("    - Applied reorder to sv_matmul: (i, j, k) -> (i, k, j)")
    # s_attn.reorder(qkt_loops["k"], qkt_loops["j"])
    # print("    - Applied reorder to qkt_matmul: (i, j, k) -> (i, k, j)")
    
    # -------------------------------------------------------------------------
    # Apply buffer_at optimization to reduce memory access
    # Creates on-chip buffer for intermediate results
    # -------------------------------------------------------------------------
    # print("  - Applying buffer_at optimization...")
    # s_attn.buffer_at(s_attn.C_h, axis=sv_loops["i"])
    # print("    - Applied buffer_at to C_h at sv_matmul:i")
    # s_attn.buffer_at(s_attn.Y, axis=qkt_loops["i"])
    # print("    - Applied buffer_at to Y at qkt_matmul:i")
    
    print("  - Created scaled_dot_product_attention schedule with pipeline optimizations")
    print("  - Matmul loops reordered to (i,k,j) for better memory access and II=1")
    
    # =========================================================================
    # Step 6: Create schedule for main attention_moe_layer function
    # =========================================================================
    print("\n[6] Creating schedule for attention_moe_layer...")
    s_attn_moe = allo.customize(
        attention_moe_layer,
        instantiate=[Ty, B, L, D, H, E, K, D_hidden]
    )
    
    # Get loop handles for attention_moe_layer
    attn_moe_loops = s_attn_moe.get_loops(s_attn_moe.top_func_name)
    print(f"  - Available top-level loops: {list(attn_moe_loops.loops.keys())}")

    print("  - Full loop hierarchy for attention_moe_layer:")
    print_loop_hierarchy(attn_moe_loops, indent=4)
    
    # Pipeline top-level loops using get_loops()
    # Note: extract_qkv and store_output are inside batch_loop
    batch_loop = attn_moe_loops["batch_loop"]
    
    # s_attn_moe.pipeline(batch_loop["extract_qkv"]["d"])
    # s_attn_moe.pipeline(batch_loop["store_output"]["d"])
    
    # =========================================================================
    # Step 7: Compose all schedules together
    # =========================================================================
    print("\n[7] Composing all schedules together...")
    s_attn_moe.compose(s_attn)
    s_attn_moe.compose(s_moe)
    print("  - Composed scaled_dot_product_attention schedule")
    print("  - Composed moe_layer schedule")
    
    print("\n" + "=" * 60)
    print("Schedule composition complete with optimizations!")
    print("=" * 60)
    print("MoE Optimizations applied:")
    print("  1. Fused GeLU into FC1 (eliminates intermediate array)")
    print("  2. Row-level structure (FC1+GeLU -> FC2)")
    print("  3. Unrolled reduction loops (parallel MACs)")
    print(f"     - Expert k loop: unroll factor {min(4, D, D_hidden)}")
    print(f"     - Gate k loop: unroll factor {min(4, D)}")
    print("  4. Pipeline on h, o, e loops (output dimensions)")
    print("  5. Array partitioning inferred by HLS (not explicit)")
    print("-" * 60)
    print("Attention Optimizations (unchanged):")
    print("  - Pipeline on innermost loops (j, k)")
    print("=" * 60)
    
    return s_attn_moe


# test/compare with pytorch
if __name__ == "__main__":
    import torch
    import torch.nn as torch_nn
    from pytorch_attention_moe import AttentionMoE
    
    # get config
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
    
    #---------------------------------------------------------------------------------- 
    # Run PyTorch implementation to get weights and outputs
    #----------------------------------------------------------------------------------
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
        expert_hidden_dim=hidden_dim
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
    print(f"PyTorch output range: [{pytorch_output.min().item():.6f}, {pytorch_output.max().item():.6f}]")
    
    #---------------------------------------------------------------------------------- 
    # Extract weights and biases from PyTorch model
    #----------------------------------------------------------------------------------
    print("\n[2] Extracting weights from PyTorch model...")
    
    # Gate weights
    gate_weight_pt = pytorch_model.moe.gate.gate_linear.weight.data  # [num_experts, embed_dim]
    gate_bias_pt = pytorch_model.moe.gate.gate_linear.bias
    if gate_bias_pt is not None:
        gate_bias_pt = gate_bias_pt.data
    else:
        gate_bias_pt = torch.zeros(num_experts)
    
    # Expert weights
    expert_fc1_weights_pt = torch.stack([exp.fc1.weight.data for exp in pytorch_model.moe.experts])
    expert_fc1_biases_pt = torch.stack([exp.fc1.bias.data for exp in pytorch_model.moe.experts])
    expert_fc2_weights_pt = torch.stack([exp.fc2.weight.data for exp in pytorch_model.moe.experts])
    expert_fc2_biases_pt = torch.stack([exp.fc2.bias.data for exp in pytorch_model.moe.experts])
    
    #---------------------------------------------------------------------------------- 
    # Convert to numpy arrays
    #----------------------------------------------------------------------------------
    print("\n[3] Converting weights to numpy arrays...")
    Q_np = np.ascontiguousarray(Q_pt.detach().numpy(), dtype=np.float32)
    K_np = np.ascontiguousarray(K_pt.detach().numpy(), dtype=np.float32)
    V_np = np.ascontiguousarray(V_pt.detach().numpy(), dtype=np.float32)
    
    gate_weight_np = np.ascontiguousarray(gate_weight_pt.detach().numpy(), dtype=np.float32)
    gate_bias_np = np.ascontiguousarray(gate_bias_pt.detach().numpy(), dtype=np.float32)
    
    expert_fc1_weights_np = np.ascontiguousarray(expert_fc1_weights_pt.detach().numpy(), dtype=np.float32)
    expert_fc1_biases_np = np.ascontiguousarray(expert_fc1_biases_pt.detach().numpy(), dtype=np.float32)
    expert_fc2_weights_np = np.ascontiguousarray(expert_fc2_weights_pt.detach().numpy(), dtype=np.float32)
    expert_fc2_biases_np = np.ascontiguousarray(expert_fc2_biases_pt.detach().numpy(), dtype=np.float32)
    
    print(f"Q shape: {Q_np.shape}")
    print(f"K shape: {K_np.shape}")
    print(f"V shape: {V_np.shape}")
    print(f"Gate weight shape: {gate_weight_np.shape}")
    print(f"Expert FC1 weights shape: {expert_fc1_weights_np.shape}")
    print(f"Expert FC2 weights shape: {expert_fc2_weights_np.shape}")
    
    #---------------------------------------------------------------------------------- 
    # Run Allo implementation
    #----------------------------------------------------------------------------------
    print("\n[4] Running Allo implementation...")
    try:
        # Create optimized schedule with composition
        allo_schedule = optimize_attention_moe_with_composition(
            batch_size, seq_len, embed_dim, num_heads, num_experts, k, hidden_dim
        )
        
        # Generate project name
        project_name = f"allo_attention_moe_alt_{CONFIG_MODE}.prj"
        print(f"Using project name: {project_name}")
        
        # Build module
        print("\n[5] Building Allo module...")
        if MODE == "llvm":
            mod = allo_schedule.build(target="llvm")
        elif MODE == "sw_emu":
            mod = allo_schedule.build(target="vitis_hls", mode="sw_emu", project=project_name)
        elif MODE == "hw_emu":
            mod = allo_schedule.build(target="vitis_hls", mode="hw_emu", project=project_name)
        elif MODE == "hw":
            mod = allo_schedule.build(target="vitis_hls", mode="hw", project=project_name)
        elif MODE == "csyn":
            mod = allo_schedule.build(target="vitis_hls", mode="csyn", project=project_name)
        else:
            raise ValueError(f"Unsupported mode: {MODE}")
        
        # Run Allo inference
        print("\n[6] Running Allo inference...")
        if MODE == "llvm":
            allo_output = mod(
                Q_np, K_np, V_np,
                gate_weight_np, gate_bias_np,
                expert_fc1_weights_np, expert_fc1_biases_np,
                expert_fc2_weights_np, expert_fc2_biases_np
            )
        elif MODE in ["sw_emu", "hw_emu", "hw"]:
            allo_output = np.zeros((batch_size, seq_len, embed_dim), dtype=np.float32)
            mod(Q_np, K_np, V_np,
                gate_weight_np, gate_bias_np,
                expert_fc1_weights_np, expert_fc1_biases_np,
                expert_fc2_weights_np, expert_fc2_biases_np,
                allo_output)
        elif MODE == "csyn":
            allo_output = np.zeros((batch_size, seq_len, embed_dim), dtype=np.float32)
            mod()
        else:
            raise ValueError(f"Unsupported mode: {MODE}")
        
        print(f"Allo output shape: {allo_output.shape}")
        print(f"Allo output range: [{allo_output.min():.6f}, {allo_output.max():.6f}]")
        
        #---------------------------------------------------------------------------------- 
        # Compare the Allo and PyTorch outputs
        #----------------------------------------------------------------------------------
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
            print(f"\n✓ SUCCESS: Allo output matches PyTorch output (atol={atol}, rtol={rtol})")
        else:
            print(f"\n✗ WARNING: Allo output differs from PyTorch output (atol={atol}, rtol={rtol})")
            print("First few differences:")
            print(diff.flatten()[:10])
        
        #---------------------------------------------------------------------------------- 
        # Print sample outputs for comparison
        #----------------------------------------------------------------------------------
        print("\n[8] Sample outputs (first token, first 5 dimensions):")
        print(f"PyTorch: {pytorch_output_np[0, 0, :5]}")
        print(f"Allo:    {allo_output[0, 0, :5]}")
        print(f"Diff:    {diff[0, 0, :5]}")
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to run Allo implementation: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
