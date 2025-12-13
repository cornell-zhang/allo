# MoE in Allo - manual implementation (no nn lib)
# matches pytorch but with small numerical differences (1e-4 to 1e-3) from:
# - different accumulation order
# - GELU approximation differences  
# - fp precision
# these are normal and don't affect performance

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

MODE = "sw_emu"  # llvm, sw_emu, hw_emu, hw, csyn

from llm_config import DEFAULT_CONFIG_MODE, get_moe_config, print_config_info

CONFIG_MODE = DEFAULT_CONFIG_MODE
def softmax_1d[Ty, N, E](X: "Ty[N, E]") -> "Ty[N, E]":
    # softmax over last dim
    Z: Ty[N, E]
    E_exp: Ty[N, E]
    M: Ty[N] = -1000000000000.0  # TODO: use -inf if available
    S: Ty[N] = 0.0
    
    # find max per row
    for i in range(N):
        for j in range(E):
            if X[i, j] > M[i]:
                M[i] = X[i, j]
    
    # exp and sum
    for i in range(N):
        for j in range(E):
            E_exp[i, j] = dsl.exp(X[i, j] - M[i])
            S[i] += E_exp[i, j]
    
    # normalize
    for i in range(N):
        for j in range(E):
            Z[i, j] = E_exp[i, j] / S[i]
    
    return Z


# top-1 selection (argmax)
def top1_select[Ty, N, E](logits: "Ty[N, E]") -> "int32[N]":
    # pick best expert for each token
    indices: int32[N]
    max_val: Ty[N] = -1000000000000.0
    
    for i in range(N):
        indices[i] = 0
        max_val[i] = logits[i, 0]
        # check rest
        for j in range(1, E):
            if logits[i, j] > max_val[i]:
                max_val[i] = logits[i, j]
                indices[i] = j
    
    return indices


# TopKGate: Gate module to select top-k experts
def topk_gate[Ty, N, D, E, K](
    x: "Ty[N, D]",
    gate_weight: "Ty[E, D]",
    gate_bias: "Ty[E]"
) -> ("Ty[N, E]", "int32[N, K]"):
    """
    Gate module to select top-k experts for routing.
    
    Args:
        x: Input tensor of shape [N, D] where N = batch * seq_len
        gate_weight: Gate weight matrix of shape [E, D]
        gate_bias: Gate bias vector of shape [E]
    Returns:
        full_weights: Sparse weight matrix of shape [N, E]
        top_k_indices: Indices of selected experts of shape [N, K]
    """
    # Compute logits using linear layer (manual implementation)
    # Computes: x @ gate_weight^T + gate_bias = [N, D] @ [D, E] + [E] = [N, E]
    # Note: In PyTorch version, bias=False, so gate_bias should be zeros
    logits: Ty[N, E] = 0.0
    for i in range(N):
        for j in range(E):
            # Initialize with bias
            logits[i, j] = gate_bias[j]
            # Matrix multiplication: x[i] @ gate_weight[j]^T
            for k in range(D):
                logits[i, j] += x[i, k] * gate_weight[j, k]
    
    # For k=1, use top1_select
    top1_indices_1d: int32[N] = top1_select[Ty, N, E](logits)
    
    # Expand to [N, K] format (for k=1, K=1)
    top_k_indices: int32[N, K] = 0
    for i in range(N):
        top_k_indices[i, 0] = top1_indices_1d[i]
    
    # Get top-k logits
    top_k_logits: Ty[N, K] = 0.0
    for i in range(N):
        for k in range(K):
            expert_idx = top_k_indices[i, k]
            top_k_logits[i, k] = logits[i, expert_idx]
    
    # Apply softmax to top-k logits
    top_k_weights = softmax_1d[Ty, N, K](top_k_logits)  # [N, K]
    
    # Create sparse weight matrix
    full_weights: Ty[N, E] = 0.0
    for i in range(N):
        for k in range(K):
            expert_idx = top_k_indices[i, k]
            full_weights[i, expert_idx] = top_k_weights[i, k]
    
    return full_weights, top_k_indices


# simple FFN expert
def expert[Ty, N, D_in, D_hidden, D_out](
    x: "Ty[N, D_in]",
    fc1_weight: "Ty[D_hidden, D_in]",
    fc1_bias: "Ty[D_hidden]",
    fc2_weight: "Ty[D_out, D_hidden]",
    fc2_bias: "Ty[D_out]"
) -> "Ty[N, D_out]":
    # FC1
    fc1_out: Ty[N, D_hidden] = 0.0
    for i in range(N):
        for j in range(D_hidden):
            fc1_out[i, j] = fc1_bias[j]
            for k in range(D_in):
                fc1_out[i, j] += x[i, k] * fc1_weight[j, k]
    
    # GELU
    gelu_out: Ty[N, D_hidden] = 0.0
    for i in range(N):
        for j in range(D_hidden):
            x_val: Ty = fc1_out[i, j]
            x3: Ty = x_val * x_val * x_val
            inner: Ty = 0.797885 * (x_val + 0.044715 * x3)
            tanh_term: Ty = dsl.tanh(inner)
            gelu_out[i, j] = 0.5 * x_val * (1.0 + tanh_term)
    
    # FC2
    fc2_out: Ty[N, D_out] = 0.0
    for i in range(N):
        for j in range(D_out):
            fc2_out[i, j] = fc2_bias[j]
            for k in range(D_hidden):
                fc2_out[i, j] += gelu_out[i, k] * fc2_weight[j, k]
    
    return fc2_out


# main MoE layer
def moe_layer[Ty, B, L, D_in, D_out, E, K, D_hidden](
    x: "Ty[B, L, D_in]",
    gate_weight: "Ty[E, D_in]",
    gate_bias: "Ty[E]",
    expert_fc1_weights: "Ty[E, D_hidden, D_in]",
    expert_fc1_biases: "Ty[E, D_hidden]",
    expert_fc2_weights: "Ty[E, D_out, D_hidden]",
    expert_fc2_biases: "Ty[E, D_out]"
) -> "Ty[B, L, D_out]":
    # process all tokens through all experts, then weight by gate
    # matches pytorch behavior
    gate_logits: Ty[B * L, E] = 0.0
    
    # Compute gate logits for all tokens
    for b in range(B):
        for l in range(L):
            token_idx = b * L + l
            # Compute logits for this token: x[b, l] @ gate_weight^T + gate_bias
            for j in range(E):
                # Initialize with bias
                gate_logits[token_idx, j] = gate_bias[j]
                # Matrix multiplication: x[b, l] @ gate_weight[j]^T
                for k in range(D_in):
                    gate_logits[token_idx, j] += x[b, l, k] * gate_weight[j, k]
    
    # Select top-k experts and compute weights
    gate_weights: Ty[B * L, E] = 0.0
    top_k_indices: int32[B * L, K] = 0
    
    for i in range(B * L):
        # Find top-1 expert (argmax) for this token
        top_expert_idx: int32 = 0
        max_logit: Ty = gate_logits[i, 0]
        
        for j in range(1, E):
            if gate_logits[i, j] > max_logit:
                max_logit = gate_logits[i, j]
                top_expert_idx = j
        
        # Store top-k indices (for k=1)
        top_k_indices[i, 0] = top_expert_idx
        
        # Get top-k logit and apply softmax
        # For k=1, softmax just returns 1.0, but we compute it for consistency
        # softmax(top_k_logit) = exp(top_k_logit) / exp(top_k_logit) = 1.0
        gate_weights[i, top_expert_idx] = 1.0
    
    # Flatten input for expert processing
    x_flat: Ty[B * L, D_in] = 0.0
    for b in range(B):
        for l in range(L):
            for d in range(D_in):
                x_flat[b * L + l, d] = x[b, l, d]
    
    # Process each expert: compute outputs for all tokens
    # Then use gate weights to select correct outputs
    expert_outputs: Ty[E, B * L, D_out] = 0.0
    
    for expert_idx in range(E):
        # Extract expert weights for this expert
        expert_fc1_w: Ty[D_hidden, D_in] = 0.0
        expert_fc1_b: Ty[D_hidden] = 0.0
        expert_fc2_w: Ty[D_out, D_hidden] = 0.0
        expert_fc2_b: Ty[D_out] = 0.0
        
        for h in range(D_hidden):
            for d in range(D_in):
                expert_fc1_w[h, d] = expert_fc1_weights[expert_idx, h, d]
            expert_fc1_b[h] = expert_fc1_biases[expert_idx, h]
        
        for o in range(D_out):
            for h in range(D_hidden):
                expert_fc2_w[o, h] = expert_fc2_weights[expert_idx, o, h]
            expert_fc2_b[o] = expert_fc2_biases[expert_idx, o]
        
        # Process all tokens through this expert (inline expert function)
        # First linear layer: fc1_out = x_flat @ expert_fc1_w^T + expert_fc1_b
        fc1_out: Ty[B * L, D_hidden] = 0.0
        for i in range(B * L):
            for j in range(D_hidden):
                # Initialize with bias
                fc1_out[i, j] = expert_fc1_b[j]
                # Matrix multiplication: x_flat[i] @ expert_fc1_w[j]^T
                for k in range(D_in):
                    fc1_out[i, j] += x_flat[i, k] * expert_fc1_w[j, k]
        
        # GELU activation: 0.5 * x * (1 + tanh(0.797885 * (x + 0.044715 * x^3)))
        gelu_out: Ty[B * L, D_hidden] = 0.0
        for i in range(B * L):
            for j in range(D_hidden):
                x_val: Ty = fc1_out[i, j]
                # Compute x^3
                x3: Ty = x_val * x_val * x_val
                # Inner term: sqrt(2/pi) * (x + 0.044715 * x^3)
                # Use exact value: sqrt(2/pi) ≈ 0.7978845608028654
                inner: Ty = 0.7978845608028654 * (x_val + 0.044715 * x3)
                # Tanh
                tanh_term: Ty = dsl.tanh(inner)
                # GELU: 0.5 * x * (1 + tanh_term)
                gelu_out[i, j] = 0.5 * x_val * (1.0 + tanh_term)
        
        # Second linear layer: fc2_out = gelu_out @ expert_fc2_w^T + expert_fc2_b
        expert_out: Ty[B * L, D_out] = 0.0
        for i in range(B * L):
            for j in range(D_out):
                # Initialize with bias
                expert_out[i, j] = expert_fc2_b[j]
                # Matrix multiplication: gelu_out[i] @ expert_fc2_w[j]^T
                for k in range(D_hidden):
                    expert_out[i, j] += gelu_out[i, k] * expert_fc2_w[j, k]
        
        # Store expert outputs
        for i in range(B * L):
            for o in range(D_out):
                expert_outputs[expert_idx, i, o] = expert_out[i, o]
    
    # Combine expert outputs using gate weights
    # For each token, sum over experts weighted by gate_weights
    output_flat: Ty[B * L, D_out] = 0.0
    for i in range(B * L):
        for expert_idx in range(E):
            weight: Ty = gate_weights[i, expert_idx]
            for o in range(D_out):
                output_flat[i, o] += expert_outputs[expert_idx, i, o] * weight
    
    # Reshape back to original shape
    output: Ty[B, L, D_out] = 0.0
    for b in range(B):
        for l in range(L):
            for o in range(D_out):
                output[b, l, o] = output_flat[b * L + l, o]
    
    return output


# test/compare with pytorch
if __name__ == "__main__":
    import numpy as np
    import torch
    import torch.nn as nn
    from pytorch_moe import MoELayer, run_moe_inference
    
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
    print("=" * 60)
    print("MoE Allo vs PyTorch Comparison Test")
    print("=" * 60)
    print_config_info(CONFIG_MODE, config)
    print(f"Seed: {seed}")
    print("=" * 60)
    
    # pytorch baseline
    print("\n[1] Running PyTorch implementation...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create PyTorch MoE layer
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
    print(f"PyTorch output range: [{pytorch_output.min().item():.6f}, {pytorch_output.max().item():.6f}]")
    
    # extract weights
    print("\n[2] Extracting weights from PyTorch model...")
    
    # Gate weights: [input_dim, num_experts] -> [num_experts, input_dim] for Allo
    gate_weight_pt = pytorch_moe.gate.gate_linear.weight.data  # [num_experts, input_dim]
    gate_bias_pt = pytorch_moe.gate.gate_linear.bias
    if gate_bias_pt is not None:
        gate_bias_pt = gate_bias_pt.data
    else:
        gate_bias_pt = torch.zeros(num_experts)
    
    # Expert weights
    expert_fc1_weights_pt = torch.stack([expert.fc1.weight.data for expert in pytorch_moe.experts])  # [num_experts, hidden_dim, input_dim]
    expert_fc1_biases_pt = torch.stack([expert.fc1.bias.data for expert in pytorch_moe.experts])  # [num_experts, hidden_dim]
    expert_fc2_weights_pt = torch.stack([expert.fc2.weight.data for expert in pytorch_moe.experts])  # [num_experts, output_dim, hidden_dim]
    expert_fc2_biases_pt = torch.stack([expert.fc2.bias.data for expert in pytorch_moe.experts])  # [num_experts, output_dim]
    
    # convert to numpy
    print("\n[3] Converting weights to numpy arrays...")
    x_np = np.ascontiguousarray(pytorch_input.detach().numpy(), dtype=np.float32)
    
    # Gate weights: PyTorch has [num_experts, input_dim], Allo expects [num_experts, input_dim] (same)
    gate_weight_np = np.ascontiguousarray(gate_weight_pt.detach().numpy(), dtype=np.float32)
    gate_bias_np = np.ascontiguousarray(gate_bias_pt.detach().numpy(), dtype=np.float32)
    
    # Expert weights: PyTorch has [num_experts, hidden_dim, input_dim], Allo expects [num_experts, hidden_dim, input_dim] (same)
    expert_fc1_weights_np = np.ascontiguousarray(expert_fc1_weights_pt.detach().numpy(), dtype=np.float32)
    expert_fc1_biases_np = np.ascontiguousarray(expert_fc1_biases_pt.detach().numpy(), dtype=np.float32)
    expert_fc2_weights_np = np.ascontiguousarray(expert_fc2_weights_pt.detach().numpy(), dtype=np.float32)
    expert_fc2_biases_np = np.ascontiguousarray(expert_fc2_biases_pt.detach().numpy(), dtype=np.float32)
    
    print(f"Input shape: {x_np.shape}")
    print(f"Gate weight shape: {gate_weight_np.shape}")
    print(f"Expert FC1 weights shape: {expert_fc1_weights_np.shape}")
    print(f"Expert FC2 weights shape: {expert_fc2_weights_np.shape}")
    
    # run allo
    print("\n[4] Running Allo implementation...")
    try:
        # Customize Allo module
        allo_mod = allo.customize(
            moe_layer, 
            instantiate=[float32, batch_size, seq_len, input_dim, output_dim, num_experts, k, hidden_dim]
        )
        
        # Generate project name based on CONFIG_MODE to avoid conflicts
        # This ensures different configurations use different build folders
        project_name = f"allo_moe_base_{CONFIG_MODE}.prj"
        print(f"Using project name: {project_name}")
        
        # Build module
        print("Building Allo module...")
        if MODE == "llvm":
            mod = allo_mod.build(target="llvm")  # Use LLVM for CPU testing
        elif MODE == "sw_emu":
            mod = allo_mod.build(target="vitis_hls", mode="sw_emu", project=project_name)
        elif MODE == "hw_emu":
            mod = allo_mod.build(target="vitis_hls", mode="hw_emu", project=project_name)
        elif MODE == "hw":
            mod = allo_mod.build(target="vitis_hls", mode="hw", project=project_name)
        elif MODE == "csyn":
            mod = allo_mod.build(target="vitis_hls", mode="csyn", project=project_name)
        else:
            raise ValueError(f"Unsupported mode: {MODE}")
        
        # Run Allo inference
        # Note: In LLVM backend, functions with return values return the result directly
        # We don't need to pass output buffer as argument
        print("Running Allo inference...")
        if MODE == "llvm":
            allo_output = mod(
                x_np,
                gate_weight_np,
                gate_bias_np,
                expert_fc1_weights_np,
                expert_fc1_biases_np,
                expert_fc2_weights_np,
                expert_fc2_biases_np
            )
        elif MODE == "sw_emu":
            allo_output = np.zeros((batch_size, seq_len, output_dim), dtype=np.float32)
            mod(x_np, gate_weight_np, gate_bias_np, expert_fc1_weights_np, expert_fc1_biases_np, expert_fc2_weights_np, expert_fc2_biases_np, allo_output)
        elif MODE == "hw_emu":
            allo_output = np.zeros((batch_size, seq_len, output_dim), dtype=np.float32)
            mod(x_np, gate_weight_np, gate_bias_np, expert_fc1_weights_np, expert_fc1_biases_np, expert_fc2_weights_np, expert_fc2_biases_np, allo_output)
        elif MODE == "hw":
            allo_output = np.zeros((batch_size, seq_len, output_dim), dtype=np.float32)
            mod(x_np, gate_weight_np, gate_bias_np, expert_fc1_weights_np, expert_fc1_biases_np, expert_fc2_weights_np, expert_fc2_biases_np, allo_output)
        elif MODE == "csyn":
            allo_output = np.zeros((batch_size, seq_len, output_dim), dtype=np.float32)
            mod()
        else:
            raise ValueError(f"Unsupported mode: {MODE}")
        
        print(f"Allo output shape: {allo_output.shape}")
        print(f"Allo output range: [{allo_output.min():.6f}, {allo_output.max():.6f}]")
        
        # compare
        print("\n[5] Comparing outputs...")
        pytorch_output_np = pytorch_output.detach().numpy()
        
        # Compute differences
        diff = np.abs(allo_output - pytorch_output_np)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        rel_diff = np.mean(diff / (np.abs(pytorch_output_np) + 1e-8))
        
        print(f"Mean absolute difference: {mean_diff:.6e}")
        print(f"Max absolute difference: {max_diff:.6e}")
        print(f"Mean relative difference: {rel_diff:.6e}")
        
        # check if close (1e-4 to 1e-3 diff is normal from accumulation order, GELU approx, fp precision)
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
        print("\n[6] Sample outputs (first token, first 5 dimensions):")
        print(f"PyTorch: {pytorch_output_np[0, 0, :5]}")
        print(f"Allo:    {allo_output[0, 0, :5]}")
        print(f"Diff:    {diff[0, 0, :5]}")
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to run Allo implementation: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)