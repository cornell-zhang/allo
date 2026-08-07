# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# MoE using Allo library functions (nn.linear2d, nn.GeLU)
# follows test_bert pattern - all functions in one file with generic types

import numpy as np
import allo
import allo.library.nn as nn
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
    for n, k in dsl.grid(N, K, name="update"):
        Z[n, k] = E_exp[n, k] / (S[n])

    return Z


# top-1 selection
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


# FFN expert using library functions
def expert[
    Ty, N, D_in, D_hidden, D_out
](
    x: "Ty[N, D_in]",
    fc1_weight: "Ty[D_hidden, D_in]",
    fc1_bias: "Ty[D_hidden]",
    fc2_weight: "Ty[D_out, D_hidden]",
    fc2_bias: "Ty[D_out]",
) -> "Ty[N, D_out]":
    # using linear2d and GeLU from library (imported at top)
    fc1_out = linear2d[Ty, Ty, Ty, N, D_hidden, D_in](x, fc1_weight, fc1_bias)
    gelu_out = GeLU[Ty, N, D_hidden](fc1_out)
    fc2_out = linear2d[Ty, Ty, Ty, N, D_out, D_hidden](gelu_out, fc2_weight, fc2_bias)
    return fc2_out


# main MoE layer
def moe_layer[
    Ty, B, L, D_in, D_out, E, K, D_hidden
](
    x: "Ty[B, L, D_in]",
    gate_weight: "Ty[E, D_in]",
    gate_bias: "Ty[E]",
    expert_fc1_weights: "Ty[E, D_hidden, D_in]",
    expert_fc1_biases: "Ty[E, D_hidden]",
    expert_fc2_weights: "Ty[E, D_out, D_hidden]",
    expert_fc2_biases: "Ty[E, D_out]",
) -> "Ty[B, L, D_out]":
    # Flatten batch and sequence dimensions: N = B * L
    N = B * L
    x_flat: Ty[N, D_in] = 0.0
    for b, l, d_in in dsl.grid(B, L, D_in, name="flatten"):
        x_flat[b * L + l, d_in] = x[b, l, d_in]

    # Step 1: Compute gate logits using linear2d (inlined topk_gate logic)
    # Note: Use linear2d (direct import) to avoid conflict with torch.nn
    gate_logits = linear2d[Ty, Ty, Ty, N, E, D_in](
        x_flat, gate_weight, gate_bias
    )  # [N, E]

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

    # Step 2: Process each expert: compute outputs for all tokens
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
            x_flat, expert_fc1_w, expert_fc1_b, expert_fc2_w, expert_fc2_b
        )  # [N, D_out]

        # Store expert outputs
        for n, d_out in dsl.grid(N, D_out, name="store_expert_out"):
            expert_outputs[e, n, d_out] = expert_out[n, d_out]

    # Step 3: Combine expert outputs using gate weights
    output_flat: Ty[N, D_out] = 0.0
    for n, e, d_out in dsl.grid(N, E, D_out, name="combine_outputs"):
        weight: Ty = gate_weights[n, e]
        output_flat[n, d_out] += expert_outputs[e, n, d_out] * weight

    # Step 4: Reshape back to original shape
    output: Ty[B, L, D_out] = 0.0
    for b, l, d_out in dsl.grid(B, L, D_out, name="reshape"):
        output[b, l, d_out] = output_flat[b * L + l, d_out]

    return output


# ==================================================================================
# Schedule optimization function
# ==================================================================================
def optimize_moe_with_composition(
    batch_size, seq_len, input_dim, output_dim, num_experts, k, hidden_dim
):
    # create schedules for sub-functions and compose them
    # TODO: add optimizations later (pipeline, partition, etc)
    Ty = float32
    N = batch_size * seq_len
    D_in = input_dim
    E = num_experts
    K = k
    D_hidden = hidden_dim
    D_out = output_dim

    print("=" * 60)
    print("Creating and optimizing MoE schedules...")
    print("=" * 60)

    # top1_select schedule
    print("\n[1] Creating schedule for top1_select...")
    s_top1 = allo.customize(top1_select, instantiate=[Ty, N, E])
    print("  - Created top1_select schedule for [N, E]")

    # softmax schedule
    print("\n[2] Creating schedule for softmax_1d...")
    s_softmax = allo.customize(softmax_1d, instantiate=[Ty, N, K])
    print("  - Created softmax_1d schedule for [N, K]")

    # expert schedule - need to create library function schedules first (type inference workaround)
    print("\n[3] Creating schedule for expert...")
    import allo.library.nn as allo_nn

    print("  - Creating schedules for library functions (linear2d, GeLU)...")
    s_linear_fc1 = allo.customize(
        allo_nn.linear2d, instantiate=[Ty, Ty, Ty, N, D_hidden, D_in]
    )
    s_gelu = allo.customize(allo_nn.GeLU, instantiate=[Ty, N, D_hidden])
    s_linear_fc2 = allo.customize(
        allo_nn.linear2d, instantiate=[Ty, Ty, Ty, N, D_out, D_hidden]
    )
    print("  - Library function schedules created")

    print("  - Creating expert schedule...")
    s_expert = allo.customize(expert, instantiate=[Ty, N, D_in, D_hidden, D_out])

    # compose library functions into expert
    s_expert.compose(s_linear_fc1, id="expert_fc1")
    s_expert.compose(s_gelu, id="expert_gelu")
    s_expert.compose(s_linear_fc2, id="expert_fc2")
    print("  - Created expert schedule")
    print("  - Composed nn.linear2d and nn.GeLU schedules for expert")

    # moe_layer schedule
    print("\n[4] Creating schedule for moe_layer...")
    s_moe = allo.customize(
        moe_layer,
        instantiate=[
            Ty,
            batch_size,
            seq_len,
            input_dim,
            output_dim,
            num_experts,
            k,
            hidden_dim,
        ],
    )

    # compose everything
    print("\n[5] Composing all schedules together...")

    # gate linear2d
    s_gate_linear = allo.customize(
        allo_nn.linear2d, instantiate=[Ty, Ty, Ty, N, E, D_in]
    )
    s_moe.compose(s_gate_linear, id="gate")
    print("  - Composed linear2d (gate logits) schedule")

    # top1 and softmax
    s_moe.compose(s_top1)
    s_moe.compose(s_softmax)
    print("  - Composed top1_select and softmax_1d schedules")

    # expert (already has library functions composed)
    s_moe.compose(s_expert)
    print("  - Composed expert schedule (includes nn.linear2d and nn.GeLU)")

    print("\n" + "=" * 60)
    print("Schedule composition complete!")
    print("=" * 60)

    return s_moe


# test/compare with pytorch
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import sys
    import os

    # Import pytorch_moe from the same directory
    from pytorch_moe import MoELayer

    # ============================================================================
    # Configuration parameters - using shared config module
    # ============================================================================
    # Get configuration from shared module
    config = get_moe_config(CONFIG_MODE)
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    num_experts = config["num_experts"]
    k = config["k"]
    hidden_dim = config["hidden_dim"]

    seed = 42

    # Print configuration info using shared function
    print_config_info(CONFIG_MODE, config)
    print(f"Seed: {seed}")

    # ----------------------------------------------------------------------------------
    # Run PyTorch implementation to get weights and outputs
    # ----------------------------------------------------------------------------------
    print("\n[1] Running PyTorch implementation...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create PyTorch MoE layer from pytorch_moe.py
    pytorch_moe = MoELayer(input_dim, output_dim, num_experts, k, hidden_dim)
    pytorch_moe.eval()

    # Initialize with Xavier uniform (matching PyTorch)
    for param in pytorch_moe.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)

    # Create random input
    torch.manual_seed(seed)
    pytorch_input = torch.randn(batch_size, seq_len, input_dim)

    # Run PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_moe(pytorch_input, verbose=False)

    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(
        f"PyTorch output range: [{pytorch_output.min().item():.6f}, {pytorch_output.max().item():.6f}]"
    )

    # ----------------------------------------------------------------------------------
    # Extract weights and biases from PyTorch model
    # ----------------------------------------------------------------------------------
    print("\n[2] Extracting weights from PyTorch model...")

    # Gate weights: [input_dim, num_experts] -> [num_experts, input_dim] for Allo
    gate_weight_pt = (
        pytorch_moe.gate.gate_linear.weight.data
    )  # [num_experts, input_dim]
    gate_bias_pt = pytorch_moe.gate.gate_linear.bias
    if gate_bias_pt is not None:
        gate_bias_pt = gate_bias_pt.data
    else:
        gate_bias_pt = torch.zeros(num_experts)

    # Expert weights
    expert_fc1_weights_pt = torch.stack(
        [expert.fc1.weight.data for expert in pytorch_moe.experts]
    )  # [num_experts, hidden_dim, input_dim]
    expert_fc1_biases_pt = torch.stack(
        [expert.fc1.bias.data for expert in pytorch_moe.experts]
    )  # [num_experts, hidden_dim]
    expert_fc2_weights_pt = torch.stack(
        [expert.fc2.weight.data for expert in pytorch_moe.experts]
    )  # [num_experts, output_dim, hidden_dim]
    expert_fc2_biases_pt = torch.stack(
        [expert.fc2.bias.data for expert in pytorch_moe.experts]
    )  # [num_experts, output_dim]

    # ----------------------------------------------------------------------------------
    # Convert the weights and biases to numpy arrays (ensure C-contiguous and correct shape)
    # ----------------------------------------------------------------------------------
    print("\n[3] Converting weights to numpy arrays...")
    x_np = np.ascontiguousarray(pytorch_input.detach().numpy(), dtype=np.float32)

    # Gate weights: PyTorch has [num_experts, input_dim], Allo expects [num_experts, input_dim] (same)
    gate_weight_np = np.ascontiguousarray(
        gate_weight_pt.detach().numpy(), dtype=np.float32
    )
    gate_bias_np = np.ascontiguousarray(gate_bias_pt.detach().numpy(), dtype=np.float32)

    # Expert weights: PyTorch has [num_experts, hidden_dim, input_dim], Allo expects [num_experts, hidden_dim, input_dim] (same)
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

    print(f"Input shape: {x_np.shape}")
    print(f"Gate weight shape: {gate_weight_np.shape}")
    print(f"Expert FC1 weights shape: {expert_fc1_weights_np.shape}")
    print(f"Expert FC2 weights shape: {expert_fc2_weights_np.shape}")

    # ----------------------------------------------------------------------------------
    # Run Allo implementation
    # ----------------------------------------------------------------------------------
    print("\n[4] Running Allo implementation...")
    try:
        # Create optimized schedule with composition
        allo_schedule = optimize_moe_with_composition(
            batch_size, seq_len, input_dim, output_dim, num_experts, k, hidden_dim
        )

        # Generate project name based on CONFIG_MODE to avoid conflicts
        # This ensures different configurations use different build folders
        project_name = f"allo_moe_lib_{CONFIG_MODE}.prj"
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
                x_np,
                gate_weight_np,
                gate_bias_np,
                expert_fc1_weights_np,
                expert_fc1_biases_np,
                expert_fc2_weights_np,
                expert_fc2_biases_np,
            )
        elif MODE == "sw_emu" or MODE == "hw_emu" or MODE == "hw":
            allo_output = np.zeros((batch_size, seq_len, output_dim), dtype=np.float32)
            mod(
                x_np,
                gate_weight_np,
                gate_bias_np,
                expert_fc1_weights_np,
                expert_fc1_biases_np,
                expert_fc2_weights_np,
                expert_fc2_biases_np,
                allo_output,
            )
        elif MODE == "csyn":
            allo_output = np.zeros((batch_size, seq_len, output_dim), dtype=np.float32)
            mod()
        else:
            raise ValueError(f"Unsupported mode: {MODE}")

        print(f"Allo output shape: {allo_output.shape}")
        print(f"Allo output range: [{allo_output.min():.6f}, {allo_output.max():.6f}]")

        # compare
        print("\n[7] Comparing outputs...")
        pytorch_output_np = pytorch_output.detach().numpy()

        diff = np.abs(allo_output - pytorch_output_np)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        rel_diff = np.mean(diff / (np.abs(pytorch_output_np) + 1e-8))

        print(f"Mean absolute difference: {mean_diff:.6e}")
        print(f"Max absolute difference: {max_diff:.6e}")
        print(f"Mean relative difference: {rel_diff:.6e}")

        # check if close (1e-4 to 1e-3 diff is normal)
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

        # sample outputs
        print("\n[8] Sample outputs (first token, first 5 dimensions):")
        print(f"PyTorch: {pytorch_output_np[0, 0, :5]}")
        print(f"Allo:    {allo_output[0, 0, :5]}")
        print(f"Diff:    {diff[0, 0, :5]}")

    except Exception as e:
        print(f"\n✗ ERROR: Failed to run Allo implementation: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 60)
