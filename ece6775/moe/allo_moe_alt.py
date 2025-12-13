# MoE with fused GeLU optimization for HLS
# fuses GeLU into FC1 to save memory and enable pipelining
# small diff vs pytorch (~1e-4) from accumulation order

import numpy as np
import allo
from allo.ir.types import float32, int32
from allo import dsl

MODE = "csyn"  # llvm, sw_emu, hw_emu, hw, csyn

import sys
import os
# path setup for config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
llm_config_dir = os.path.join(project_root, "llm_config")
if llm_config_dir not in sys.path:
    sys.path.insert(0, llm_config_dir)

from llm_config import DEFAULT_CONFIG_MODE, get_moe_config, print_config_info

CONFIG_MODE = DEFAULT_CONFIG_MODE
def softmax_1d[Ty, N, K](X: "Ty[N, K]") -> "Ty[N, K]":
    # stable softmax over last dim
    Z: Ty[N, K]
    E_exp: Ty[N, K]
    M: Ty[N] = -1e12
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

# top-1 selection (argmax)
def top1_select[Ty, N, E](logits: "Ty[N, E]") -> "int32[N]":
    # pick best expert per token
    indices: int32[N]
    max_val: Ty[N] = -1e12
    
    for n in range(N, name="init"):
        indices[n] = 0
        max_val[n] = logits[n, 0]
    
    for n, e in dsl.grid(N, E, name="argmax"):
        if e > 0 and logits[n, e] > max_val[n]:
            max_val[n] = logits[n, e]
            indices[n] = e
    
    return indices
# FFN expert with fused GeLU, row-by-row for pipelining
def expert[Ty, N, D_in, D_hidden, D_out](
    x: "Ty[N, D_in]",
    fc1_weight: "Ty[D_hidden, D_in]",
    fc1_bias: "Ty[D_hidden]",
    fc2_weight: "Ty[D_out, D_hidden]",
    fc2_bias: "Ty[D_out]"
) -> "Ty[N, D_out]":
    output: Ty[N, D_out] = 0.0
    
    for n in range(N, name="row_loop"):
        hidden_row: Ty[D_hidden] = 0.0
        
        # FC1 + GeLU together
        for h in range(D_hidden, name="fc1_gelu_loop"):
            acc: Ty = 0.0
            for k in range(D_in, name="fc1_reduce"):
                acc += x[n, k] * fc1_weight[h, k]
            fc1_val: Ty = acc + fc1_bias[h]
            
            # GeLU: tanh approximation
            x3: Ty = fc1_val * fc1_val * fc1_val
            inner: Ty = 0.7978845608028654 * (fc1_val + 0.044715 * x3)  # sqrt(2/pi) ≈ 0.7978...
            hidden_row[h] = 0.5 * fc1_val * (1.0 + dsl.tanh(inner))
        
        # FC2
        for o in range(D_out, name="fc2_loop"):
            acc2: Ty = 0.0
            for k in range(D_hidden, name="fc2_reduce"):
                acc2 += hidden_row[k] * fc2_weight[o, k]
            output[n, o] = acc2 + fc2_bias[o]
    
    return output
# Top-1 MoE: pick one expert per token, weight with softmax
def moe_layer[Ty, N, D_in, D_out, E, K, D_hidden](
    x: "Ty[N, D_in]",
    gate_weight: "Ty[E, D_in]",
    gate_bias: "Ty[E]",
    expert_fc1_weights: "Ty[E, D_hidden, D_in]",
    expert_fc1_biases: "Ty[E, D_hidden]",
    expert_fc2_weights: "Ty[E, D_out, D_hidden]",
    expert_fc2_biases: "Ty[E, D_out]"
) -> "Ty[N, D_out]":
    # compute gate scores
    gate_logits: Ty[N, E] = 0.0
    for n, e in dsl.grid(N, E, name="gate_linear"):
        acc: Ty = 0.0
        for k in range(D_in, name="gate_reduce"):
            acc += x[n, k] * gate_weight[e, k]
        gate_logits[n, e] = acc + gate_bias[e]
    
    # pick best expert
    top1_idx: int32[N] = top1_select[Ty, N, E](gate_logits)
    
    # get top-k logits (for k=1)
    top_k_logits: Ty[N, K] = 0.0
    for n, k in dsl.grid(N, K, name="topk_logits"):
        expert_idx = top1_idx[n] if k == 0 else 0
        top_k_logits[n, k] = gate_logits[n, expert_idx]
    
    top_k_weights = softmax_1d[Ty, N, K](top_k_logits)
    
    # sparse weights
    gate_w: Ty[N, E] = 0.0
    for n in range(N, name="sparse_gate"):
        gate_w[n, top1_idx[n]] = top_k_weights[n, 0]
    
    # run all experts (could optimize to skip unused ones later)
    expert_out: Ty[E, N, D_out] = 0.0
    
    for e in range(E, name="expert_loop"):
        # extract weights for this expert
        fc1_w: Ty[D_hidden, D_in] = 0.0
        fc1_b: Ty[D_hidden] = 0.0
        fc2_w: Ty[D_out, D_hidden] = 0.0
        fc2_b: Ty[D_out] = 0.0
        
        for i, j in dsl.grid(D_hidden, D_in, name="extract_fc1_w"):
            fc1_w[i, j] = expert_fc1_weights[e, i, j]
        for i in range(D_hidden, name="extract_fc1_b"):
            fc1_b[i] = expert_fc1_biases[e, i]
        for i, j in dsl.grid(D_out, D_hidden, name="extract_fc2_w"):
            fc2_w[i, j] = expert_fc2_weights[e, i, j]
        for i in range(D_out, name="extract_fc2_b"):
            fc2_b[i] = expert_fc2_biases[e, i]
        
        out = expert[Ty, N, D_in, D_hidden, D_out](x, fc1_w, fc1_b, fc2_w, fc2_b)
        
        for n, d in dsl.grid(N, D_out, name="store_expert_out"):
            expert_out[e, n, d] = out[n, d]
    
    # combine with weights
    output: Ty[N, D_out] = 0.0
    for n, e, d_out in dsl.grid(N, E, D_out, name="combine_outputs"):
        output[n, d_out] += expert_out[e, n, d_out] * gate_w[n, e]
    
    return output


# build schedule with pipelining/unrolling optimizations
def optimize_moe_schedule(num_tokens, input_dim, output_dim, num_experts, k, hidden_dim):
    Ty = float32
    N = num_tokens
    D_in = input_dim
    D_out = output_dim
    E = num_experts
    K = k
    D_hidden = hidden_dim
    
    print("Building MoE schedule...")
    
    # top1_select
    s_top1 = allo.customize(top1_select, instantiate=[Ty, N, E])
    s_top1.pipeline("top1_select:e")
    
    # softmax
    s_softmax = allo.customize(softmax_1d, instantiate=[Ty, N, K])
    sm_loops = s_softmax.get_loops(s_softmax.top_func_name)
    s_softmax.pipeline(sm_loops["row_max"]["k"])
    s_softmax.pipeline(sm_loops["exp_sum"]["k"])
    s_softmax.pipeline(sm_loops["normalize"]["k"])
    
    # expert
    s_expert = allo.customize(expert, instantiate=[Ty, N, D_in, D_hidden, D_out])
    exp_loops = s_expert.get_loops(s_expert.top_func_name)
    row_loop = exp_loops["row_loop"]
    s_expert.pipeline(row_loop["h"])
    s_expert.pipeline(row_loop["o"])
    
    # unroll reduction loop for parallel MACs
    unroll_factor = min(4, D_in, D_hidden)
    s_expert.unroll(row_loop["k"], factor=unroll_factor)
    print(f"  - Applied unroll to k (reduction loop) factor={unroll_factor}")
    
    # array partitioning
    if D_hidden <= 32:
        s_expert.partition(s_expert.hidden_row, dim=0)
        print(f"  - Applied complete partition to hidden_row (D_hidden={D_hidden})")
    else:
        s_expert.partition(s_expert.hidden_row, dim=0, factor=4)
        print(f"  - Applied cyclic partition to hidden_row (factor=4)")
    
    print("  - Created expert schedule with pipeline, unroll, and partition optimizations")
    
    # moe_layer schedule
    print("\n[4] Creating schedule for moe_layer...")
    s_moe = allo.customize(moe_layer, instantiate=[Ty, N, D_in, D_out, E, K, D_hidden])
    
    moe_loops = s_moe.get_loops(s_moe.top_func_name)
    print(f"  - Available top-level loops: {list(moe_loops.loops.keys())}")
    
    # optimize gate_linear
    gate_linear_loop = moe_loops["gate_linear"]
    print(f"  - gate_linear sub-loops: {list(gate_linear_loop.loops.keys())}")
    
    s_moe.pipeline(gate_linear_loop["e"])
    print("  - Applied pipeline to gate_linear:e")
    
    unroll_factor_gate = min(4, D_in)
    s_moe.unroll(gate_linear_loop["k"], factor=unroll_factor_gate)
    print(f"  - Applied unroll to gate_linear:k (factor={unroll_factor_gate})")
    
    # other loops
    combine_loop = moe_loops["combine_outputs"]
    s_moe.pipeline(combine_loop["d_out"])
    
    topk_loop = moe_loops["topk_logits"]
    s_moe.pipeline(topk_loop["k"])
    
    sparse_loop = moe_loops["sparse_gate"]
    s_moe.pipeline(sparse_loop["n"])
    
    print("  - Applied pipeline to combine_outputs:d_out, topk_logits:k, sparse_gate:n")
    
    # compose sub-schedules
    s_moe.compose(s_top1)
    s_moe.compose(s_softmax)
    s_moe.compose(s_expert)
    print("  - Composed top1_select, softmax_1d, and expert schedules")
    
    print("\n" + "=" * 60)
    print("Schedule done!")
    print("=" * 60)
    print("Optimizations:")
    print("  1. Fused GeLU into FC1")
    print("  2. Row-level structure")
    print("  3. Unrolled reduction loops")
    print(f"     - Expert k: factor {unroll_factor}")
    print(f"     - Gate k: factor {unroll_factor_gate}")
    print("  4. Array partitioning for hidden_row")
    print("  5. Pipeline on h, o, e loops")
    print("=" * 60)
    
    return s_moe


# test/compare with pytorch
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from pytorch_moe import MoELayer
    
    moe_config = get_moe_config(CONFIG_MODE)
    
    batch_size = moe_config["batch_size"]
    seq_len = moe_config["seq_len"]
    input_dim = moe_config["input_dim"]
    output_dim = moe_config["output_dim"]
    num_experts = moe_config["num_experts"]
    k = moe_config["k"]
    hidden_dim = moe_config["hidden_dim"]
    seed = 42
    
    print("=" * 60)
    print("MoE Allo (Vayun Optimized) vs PyTorch Comparison Test")
    print("=" * 60)
    print(f"Configuration Mode: {CONFIG_MODE}")
    print(f"Configuration:")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  input_dim={input_dim}, output_dim={output_dim}")
    print(f"  num_experts={num_experts}, k={k}, hidden_dim={hidden_dim}")
    print(f"  Seed: {seed}")
    print("=" * 60)
    
    # pytorch baseline
    print("\n[1] Running PyTorch implementation...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create PyTorch MoE layer
    pytorch_moe = MoELayer(input_dim, output_dim, num_experts, k, hidden_dim)
    pytorch_moe.eval()
    
    # Initialize with Xavier uniform
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
    print(f"PyTorch output range: [{pytorch_output.min().item():.6f}, {pytorch_output.max().item():.6f}]")
    
    # extract weights
    print("\n[2] Extracting weights from PyTorch model...")
    
    # Gate weights
    gate_weight_pt = pytorch_moe.gate.gate_linear.weight.data  # [num_experts, input_dim]
    gate_bias_pt = pytorch_moe.gate.gate_linear.bias
    if gate_bias_pt is not None:
        gate_bias_pt = gate_bias_pt.data
    else:
        gate_bias_pt = torch.zeros(num_experts)
    
    # Expert weights
    expert_fc1_weights_pt = torch.stack([exp.fc1.weight.data for exp in pytorch_moe.experts])
    expert_fc1_biases_pt = torch.stack([exp.fc1.bias.data for exp in pytorch_moe.experts])
    expert_fc2_weights_pt = torch.stack([exp.fc2.weight.data for exp in pytorch_moe.experts])
    expert_fc2_biases_pt = torch.stack([exp.fc2.bias.data for exp in pytorch_moe.experts])
    
    # convert to numpy
    print("\n[3] Converting weights to numpy arrays...")
    x_np = np.ascontiguousarray(pytorch_input.detach().numpy(), dtype=np.float32)
    
    gate_weight_np = np.ascontiguousarray(gate_weight_pt.detach().numpy(), dtype=np.float32)
    gate_bias_np = np.ascontiguousarray(gate_bias_pt.detach().numpy(), dtype=np.float32)
    
    expert_fc1_weights_np = np.ascontiguousarray(expert_fc1_weights_pt.detach().numpy(), dtype=np.float32)
    expert_fc1_biases_np = np.ascontiguousarray(expert_fc1_biases_pt.detach().numpy(), dtype=np.float32)
    expert_fc2_weights_np = np.ascontiguousarray(expert_fc2_weights_pt.detach().numpy(), dtype=np.float32)
    expert_fc2_biases_np = np.ascontiguousarray(expert_fc2_biases_pt.detach().numpy(), dtype=np.float32)
    
    print(f"Input shape: {x_np.shape}")
    print(f"Gate weight shape: {gate_weight_np.shape}")
    print(f"Expert FC1 weights shape: {expert_fc1_weights_np.shape}")
    print(f"Expert FC2 weights shape: {expert_fc2_weights_np.shape}")
    
    # flatten input (N = B * L)
    num_tokens = batch_size * seq_len
    x_flat_np = x_np.reshape(num_tokens, input_dim)
    print(f"Flattened input shape: {x_flat_np.shape}")
    
    # run allo
    print("\n[4] Running Allo implementation...")
    try:
        # Create optimized schedule
        allo_schedule = optimize_moe_schedule(
            num_tokens, input_dim, output_dim, num_experts, k, hidden_dim
        )
        
        # Generate project name
        project_name = f"allo_moe_alt_{CONFIG_MODE}.prj"
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
            allo_output_flat = mod(
                x_flat_np,
                gate_weight_np, gate_bias_np,
                expert_fc1_weights_np, expert_fc1_biases_np,
                expert_fc2_weights_np, expert_fc2_biases_np
            )
        elif MODE in ["sw_emu", "hw_emu", "hw"]:
            allo_output_flat = np.zeros((num_tokens, output_dim), dtype=np.float32)
            mod(x_flat_np,
                gate_weight_np, gate_bias_np,
                expert_fc1_weights_np, expert_fc1_biases_np,
                expert_fc2_weights_np, expert_fc2_biases_np,
                allo_output_flat)
        elif MODE == "csyn":
            allo_output_flat = np.zeros((num_tokens, output_dim), dtype=np.float32)
            mod()
        else:
            raise ValueError(f"Unsupported mode: {MODE}")
        
        # Reshape output back to [B, L, D_out]
        allo_output = allo_output_flat.reshape(batch_size, seq_len, output_dim)
        
        print(f"Allo output shape: {allo_output.shape}")
        print(f"Allo output range: [{allo_output.min():.6f}, {allo_output.max():.6f}]")
        
        # compare
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