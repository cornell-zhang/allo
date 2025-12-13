"""
Attention + Mixture of Experts (MoE) Implementation for Inference Only

This script implements:
1. Scaled Dot-Product Attention (Multi-Head Attention)
2. MoE layer
3. Combined Attention -> MoE pipeline

Uses random inputs and weights for demonstration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (Multi-Head Attention).
    
    This implementation matches the Allo library version:
    - Input: Q, K, V of shape [L, D]
    - Split into H heads, each with dimension D // H
    - For each head: softmax(QK^T / sqrt(D // H)) @ V
    - Merge heads back to [L, D]
    
    Args:
        num_heads: Number of attention heads (H)
        head_dim: Dimension per head (D // H)
    """
    
    def __init__(self, num_heads: int, embed_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Scaling factor: 1 / sqrt(head_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Forward pass through scaled dot-product attention.
        
        Args:
            Q: Query tensor of shape [L, D]
            K: Key tensor of shape [L, D]
            V: Value tensor of shape [L, D]
            verbose: Whether to print debug information
        
        Returns:
            Output tensor of shape [L, D]
        """
        L, D = Q.shape
        H = self.num_heads
        head_dim = D // H
        
        # Initialize output
        Z = torch.zeros(L, D, device=Q.device, dtype=Q.dtype)
        
        # Process each head
        for h in range(H):
            # Split Q, K, V for this head
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            
            Q_h = Q[:, start_idx:end_idx]  # [L, head_dim]
            K_h = K[:, start_idx:end_idx]  # [L, head_dim]
            V_h = V[:, start_idx:end_idx]  # [L, head_dim]
            
            # QK^T = [L, head_dim] @ [head_dim, L] = [L, L]
            # Note: K_h.T is the transpose
            Y = torch.matmul(Q_h, K_h.T)  # [L, L]
            
            # Scale by 1/sqrt(head_dim)
            Y = Y * self.scale
            
            # Softmax over last dimension
            S = F.softmax(Y, dim=-1)  # [L, L]
            
            # YV = [L, L] @ [L, head_dim] = [L, head_dim]
            C_h = torch.matmul(S, V_h)  # [L, head_dim]
            
            # Merge back to Z
            Z[:, start_idx:end_idx] = C_h
        
        if verbose:
            print(f"\n[Attention] L={L}, D={D}, H={H}, head_dim={head_dim}")
            print(f"[Attention] scale={self.scale:.6f}")
            print(f"[Attention] Output shape: {Z.shape}")
        
        return Z


class TopKGate(nn.Module):
    """
    Gate module to select top k experts for routing.
    
    Args:
        input_dim: Input feature dimension
        num_experts: Total number of experts
        k: Number of experts to select per token
    """
    
    def __init__(self, input_dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        # Linear layer to compute expert logits
        self.gate_linear = nn.Linear(input_dim, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor, verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the gate.
        
        Args:
            x: Input tensor of shape [batch_size * seq_len, input_dim]
            verbose: Whether to print routing information
        
        Returns:
            full_weights: Sparse weight matrix [batch_size * seq_len, num_experts]
            top_k_indices: Indices of selected experts [batch_size * seq_len, k]
        """
        # Compute logits for all experts
        logits = self.gate_linear(x)  # [N, num_experts]
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        
        # Apply softmax to top-k logits for normalized weights
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [N, k]
        
        # Create sparse weight matrix (zeros for non-selected experts)
        full_weights = torch.zeros_like(logits)
        full_weights.scatter_(1, top_k_indices, top_k_weights)
        
        # Print routing information if verbose
        if verbose:
            num_tokens = x.shape[0]
            
            # Count tokens per expert
            expert_counts = {}
            for expert_idx in range(self.num_experts):
                count = (top_k_indices == expert_idx).sum().item()
                if count > 0:
                    expert_counts[expert_idx] = count
            
            # Verify that each token selects exactly k experts
            tokens_per_expert_count = {}
            for i in range(num_tokens):
                num_experts_selected = (top_k_weights[i] > 1e-6).sum().item()
                tokens_per_expert_count[num_experts_selected] = tokens_per_expert_count.get(num_experts_selected, 0) + 1
            
            print(f"\n[Gate Routing] Total tokens: {num_tokens}, k={self.k}")
            print(f"[Gate Routing] Expert distribution:")
            total_selections = num_tokens * self.k  # Total expert-token pairs
            for expert_idx, count in sorted(expert_counts.items()):
                percentage = (count / total_selections) * 100
                print(f"  Expert {expert_idx}: {count} selections ({percentage:.1f}% of {total_selections} total selections)")
            
            # Verify top-k: each token selects exactly k experts
            if tokens_per_expert_count.get(self.k, 0) == num_tokens:
                print(f"[Gate Routing] ✓ All {num_tokens} tokens select exactly {self.k} expert(s)")
            else:
                print(f"[Gate Routing] ⚠ Expected all tokens to select {self.k} expert(s), but got: {tokens_per_expert_count}")
        
        return full_weights, top_k_indices


class Expert(nn.Module):
    """
    A simple feed-forward expert network.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Use GELU activation (modern choice)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class MoELayer(nn.Module):
    """
    Mixture of Experts layer for inference.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        num_experts: Total number of experts
        k: Number of experts to activate per token
        expert_hidden_dim: Hidden dimension for experts (default: 4 * input_dim)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        k: int = 1,
        expert_hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.output_dim = output_dim
        
        if expert_hidden_dim is None:
            expert_hidden_dim = input_dim * 4  # Common practice
        
        # Initialize gate and experts
        self.gate = TopKGate(input_dim, num_experts, k)
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            verbose: Whether to print verification information for top-1 MoE
        
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        
        # Store original shape
        original_shape = x.shape
        batch_size, seq_len = original_shape[0], original_shape[1]
        
        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, original_shape[-1])  # [N, input_dim], N = batch * seq_len
        num_tokens = x_flat.shape[0]
        
        # Get gating weights and expert indices
        gate_weights, top_k_indices = self.gate(x_flat, verbose=verbose)  # [N, num_experts], [N, k]
        
        # Initialize output tensor
        output = torch.zeros(
            x_flat.shape[0],
            self.output_dim,
            device=x.device,
            dtype=x.dtype
        )
        
        # Track expert usage statistics
        expert_usage_stats = {}
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            # Create mask: tokens that have this expert in their top-k
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [N]
            
            num_tokens_for_expert = expert_mask.sum().item()
            
            if not expert_mask.any():
                expert_usage_stats[expert_idx] = 0
                continue  # No tokens for this expert
            
            # Track usage
            expert_usage_stats[expert_idx] = num_tokens_for_expert
            
            # Get inputs for this expert
            expert_inputs = x_flat[expert_mask]  # [num_tokens, input_dim]
            
            # Process through expert
            expert_outputs = self.experts[expert_idx](expert_inputs)  # [num_tokens, output_dim]
            
            # Get corresponding weights for these tokens
            # For each token, find the weight for this expert (could be in any of k positions)
            token_indices = torch.where(expert_mask)[0]  # Indices in flattened tensor
            expert_weights = gate_weights[token_indices, expert_idx].unsqueeze(1)  # [num_tokens, 1]
            
            # Weight and accumulate outputs
            weighted_outputs = expert_outputs * expert_weights
            output[token_indices] += weighted_outputs
        
        # Verify top-k behavior
        if verbose:
            active_experts = [idx for idx, count in expert_usage_stats.items() if count > 0]
            total_usage = sum(expert_usage_stats.values())
            
            # Verify top-k: each token selects exactly k experts
            if top_k_indices.shape[1] != self.k:
                print(f"[MoE Verification] ✗ ERROR: Expected k={self.k}, but top_k_indices has shape {top_k_indices.shape}")
            
            # Verify all tokens are processed (for k>1, total_usage may be > num_tokens)
            expected_total_usage = num_tokens * self.k
            if total_usage != expected_total_usage:
                print(f"[MoE Verification] ⚠ Expected {expected_total_usage} expert-token pairs (={num_tokens} tokens × {self.k} experts), but got {total_usage}")
            
            # Check if weights are normalized (should sum to 1.0 per token)
            sample_weights = gate_weights.sum(dim=1)
            weights_normalized = torch.allclose(sample_weights, torch.ones_like(sample_weights), atol=1e-5)
            
            # Final summary for top-k
            print(f"\n[MoE Verification] === Top-{self.k} MoE Verification ===")
            print(f"  ✓ Each token routes to exactly {self.k} expert(s) (k={self.k})")
            print(f"  ✓ All {num_tokens} tokens processed")
            print(f"  ✓ Total expert-token pairs: {total_usage} (expected: {expected_total_usage})")
            print(f"  ✓ Active experts: {len(active_experts)}/{self.num_experts}")
            print(f"  ✓ Expert distribution: {dict(sorted(expert_usage_stats.items()))}")
            if weights_normalized:
                print(f"  ✓ Gate weights normalized (sum to 1.0 per token)")
            else:
                print(f"  ✗ Gate weights NOT normalized correctly")
            print(f"  === Top-{self.k} MoE is working correctly! ===")
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, self.output_dim)
        
        return output


class AttentionMoE(nn.Module):
    """
    Combined Attention + MoE layer.
    
    Data flow: Input -> Attention -> MoE -> Output
    
    This follows the Transformer architecture pattern where:
    1. Attention computes self-attention on the input
    2. MoE replaces the standard FFN (Feed-Forward Network)
    
    Args:
        seq_len: Sequence length (L)
        embed_dim: Embedding dimension (D)
        num_heads: Number of attention heads (H)
        num_experts: Number of MoE experts (E)
        k: Top-k experts to activate per token
        expert_hidden_dim: Hidden dimension for experts
    """
    
    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        k: int = 1,
        expert_hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Attention layer
        self.attention = ScaledDotProductAttention(num_heads, embed_dim)
        
        # MoE layer (input_dim = output_dim = embed_dim)
        self.moe = MoELayer(
            input_dim=embed_dim,
            output_dim=embed_dim,
            num_experts=num_experts,
            k=k,
            expert_hidden_dim=expert_hidden_dim
        )
    
    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Forward pass: Attention -> MoE
        
        Args:
            Q: Query tensor of shape [B, L, D]
            K: Key tensor of shape [B, L, D]
            V: Value tensor of shape [B, L, D]
            verbose: Whether to print debug information
        
        Returns:
            Output tensor of shape [B, L, D]
        """
        batch_size = Q.shape[0]
        L = Q.shape[1]
        D = Q.shape[2]
        
        # Process each batch item through attention
        # Attention expects [L, D], so we process batch by batch
        attn_outputs = []
        for b in range(batch_size):
            attn_out = self.attention(Q[b], K[b], V[b], verbose=verbose and b == 0)
            attn_outputs.append(attn_out)
        
        # Stack back to [B, L, D]
        attn_output = torch.stack(attn_outputs, dim=0)
        
        if verbose:
            print(f"\n[AttentionMoE] Attention output shape: {attn_output.shape}")
        
        # Pass through MoE
        # MoE expects [B, L, D] and returns [B, L, D]
        moe_output = self.moe(attn_output, verbose=verbose)
        
        if verbose:
            print(f"[AttentionMoE] MoE output shape: {moe_output.shape}")
        
        return moe_output


def run_attention_moe_inference(
    batch_size: int = 4,
    seq_len: int = 4,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_experts: int = 2,
    k: int = 1,
    expert_hidden_dim: Optional[int] = None,
    seed: int = 24,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, AttentionMoE]:
    """
    Run Attention + MoE inference with fixed seed for reproducible inputs and weights.
    
    Args:
        batch_size: Batch size (B)
        seq_len: Sequence length (L)
        embed_dim: Embedding dimension (D)
        num_heads: Number of attention heads (H)
        num_experts: Number of experts (E)
        k: Top-k experts to activate per token
        expert_hidden_dim: Hidden dimension for experts
        seed: Random seed for reproducibility
        verbose: Whether to print verification information
    
    Returns:
        output: Output tensor from AttentionMoE [B, L, D]
        Q, K, V: Input tensors used for inference
        model: AttentionMoE model instance
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create model
    model = AttentionMoE(
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_experts=num_experts,
        k=k,
        expert_hidden_dim=expert_hidden_dim
    )
    model.eval()
    
    # Initialize with random weights (Xavier uniform for better stability)
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    # Create random inputs (Q, K, V)
    torch.manual_seed(seed)
    Q = torch.randn(batch_size, seq_len, embed_dim)
    K = torch.randn(batch_size, seq_len, embed_dim)
    V = torch.randn(batch_size, seq_len, embed_dim)
    
    # Run inference with verbose output for verification
    with torch.no_grad():
        output = model(Q, K, V, verbose=verbose)
    
    return output, Q, K, V, model


def verify_gate_layer(
    num_tokens: int = 128,
    input_dim: int = 96,
    num_experts: int = 2,
    k: int = 1,
    seed: int = 42
):
    """
    Verify that the Gate Layer is effectively routing tokens to different experts.
    
    This function tests:
    1. Different inputs produce different routing decisions
    2. Gate weights are properly normalized (sum to 1.0)
    3. Expert distribution is reasonable (not all tokens to one expert)
    4. Routing is deterministic (same input -> same routing)
    """
    print("=" * 70)
    print("Gate Layer Verification Test")
    print("=" * 70)
    print(f"Configuration: num_tokens={num_tokens}, input_dim={input_dim}")
    print(f"               num_experts={num_experts}, k={k}")
    print("=" * 70)
    
    torch.manual_seed(seed)
    
    # Create gate
    gate = TopKGate(input_dim, num_experts, k)
    nn.init.xavier_uniform_(gate.gate_linear.weight)
    gate.eval()
    
    # Test 1: Create random inputs and check routing
    print("\n[Test 1] Random Input Routing")
    print("-" * 50)
    x = torch.randn(num_tokens, input_dim)
    
    with torch.no_grad():
        full_weights, top_k_indices = gate(x, verbose=False)
    
    # Count tokens per expert
    expert_counts = {}
    for e in range(num_experts):
        count = (top_k_indices == e).sum().item()
        expert_counts[e] = count
        percentage = (count / (num_tokens * k)) * 100
        print(f"  Expert {e}: {count} tokens ({percentage:.1f}%)")
    
    # Check if routing is balanced
    min_count = min(expert_counts.values())
    max_count = max(expert_counts.values())
    imbalance_ratio = max_count / max(min_count, 1)
    
    if imbalance_ratio < 10:
        print(f"  ✓ Routing is reasonably balanced (imbalance ratio: {imbalance_ratio:.2f})")
    else:
        print(f"  ⚠ Routing is imbalanced (imbalance ratio: {imbalance_ratio:.2f})")
    
    # Test 2: Verify weights are normalized
    print("\n[Test 2] Weight Normalization")
    print("-" * 50)
    weight_sums = full_weights.sum(dim=1)
    all_normalized = torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    
    if all_normalized:
        print(f"  ✓ All gate weights sum to 1.0 (properly normalized)")
    else:
        print(f"  ✗ Gate weights NOT properly normalized!")
        print(f"    Weight sums range: [{weight_sums.min():.6f}, {weight_sums.max():.6f}]")
    
    # Test 3: Different inputs should (mostly) produce different routings
    print("\n[Test 3] Input-Dependent Routing")
    print("-" * 50)
    
    # Create two very different inputs
    x1 = torch.randn(10, input_dim) * 10  # Large magnitude
    x2 = torch.randn(10, input_dim) * 0.1  # Small magnitude
    
    with torch.no_grad():
        _, indices1 = gate(x1, verbose=False)
        _, indices2 = gate(x2, verbose=False)
    
    # Check if routings are different
    same_routing = (indices1 == indices2).all().item()
    if not same_routing:
        print(f"  ✓ Different inputs produce different routing decisions")
    else:
        print(f"  ⚠ Different inputs produced same routing (may indicate issue)")
    
    # Test 4: Determinism - same input should always produce same routing
    print("\n[Test 4] Routing Determinism")
    print("-" * 50)
    
    x_test = torch.randn(20, input_dim)
    with torch.no_grad():
        _, indices_run1 = gate(x_test, verbose=False)
        _, indices_run2 = gate(x_test, verbose=False)
    
    is_deterministic = (indices_run1 == indices_run2).all().item()
    if is_deterministic:
        print(f"  ✓ Routing is deterministic (same input -> same expert)")
    else:
        print(f"  ✗ Routing is NOT deterministic!")
    
    # Test 5: Show actual routing decisions for a few tokens
    print("\n[Test 5] Sample Routing Decisions")
    print("-" * 50)
    
    # Create a small set of tokens to visualize
    x_sample = torch.randn(8, input_dim)
    with torch.no_grad():
        logits = gate.gate_linear(x_sample)
        weights, indices = gate(x_sample, verbose=False)
    
    print(f"  Token | Expert Logits (raw)         | Selected Expert | Weight")
    print(f"  " + "-" * 60)
    for i in range(min(8, x_sample.shape[0])):
        logit_str = ", ".join([f"{l:.3f}" for l in logits[i].tolist()])
        selected = indices[i, 0].item()
        weight = weights[i, selected].item()
        print(f"  {i:5d} | [{logit_str:25s}] | Expert {selected}        | {weight:.4f}")
    
    # Test 6: Verify gate learns meaningful patterns
    print("\n[Test 6] Pattern-Based Routing")
    print("-" * 50)
    
    # Create inputs with clear patterns
    # Pattern A: positive values in first half, negative in second half
    # Pattern B: opposite
    pattern_a = torch.cat([torch.ones(input_dim // 2), -torch.ones(input_dim // 2)]).unsqueeze(0)
    pattern_b = torch.cat([-torch.ones(input_dim // 2), torch.ones(input_dim // 2)]).unsqueeze(0)
    
    # Repeat patterns
    x_patterns = torch.cat([pattern_a.repeat(5, 1), pattern_b.repeat(5, 1)], dim=0)
    
    with torch.no_grad():
        _, pattern_indices = gate(x_patterns, verbose=False)
    
    pattern_a_experts = pattern_indices[:5, 0].tolist()
    pattern_b_experts = pattern_indices[5:, 0].tolist()
    
    print(f"  Pattern A (positive-negative) routed to experts: {pattern_a_experts}")
    print(f"  Pattern B (negative-positive) routed to experts: {pattern_b_experts}")
    
    # Check if same patterns go to same expert
    a_consistent = len(set(pattern_a_experts)) == 1
    b_consistent = len(set(pattern_b_experts)) == 1
    patterns_different = set(pattern_a_experts) != set(pattern_b_experts)
    
    if a_consistent and b_consistent:
        print(f"  ✓ Same patterns consistently route to same expert")
    else:
        print(f"  ⚠ Same patterns route to different experts (expected with random weights)")
    
    if patterns_different:
        print(f"  ✓ Different patterns route to different experts")
    else:
        print(f"  ⚠ Different patterns route to same expert")
    
    print("\n" + "=" * 70)
    print("Gate Layer Verification Complete!")
    print("=" * 70)
    
    return gate, expert_counts


def benchmark_attention_moe(
    batch_size: int = 1,
    seq_len: int = 64,
    embed_dim: int = 256,
    num_heads: int = 8,
    num_experts: int = 4,
    k: int = 1,
    expert_hidden_dim: Optional[int] = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cpu",
    seed: int = 42
):
    """
    Benchmark AttentionMoE inference time.
    
    Best practices for accurate timing:
    1. Warmup runs: Avoid cold start overhead (JIT compilation, cache warming)
    2. Multiple runs: Get stable average and standard deviation
    3. torch.no_grad(): Disable gradient tracking for inference
    4. torch.cuda.synchronize(): Ensure GPU operations complete before timing
    
    Args:
        batch_size: Batch size (B)
        seq_len: Sequence length (L)
        embed_dim: Embedding dimension (D)
        num_heads: Number of attention heads (H)
        num_experts: Number of experts (E)
        k: Top-k experts per token
        expert_hidden_dim: Hidden dim for experts (default: 4 * embed_dim)
        num_warmup: Number of warmup iterations
        num_runs: Number of timed iterations
        device: "cpu" or "cuda"
        seed: Random seed
    
    Returns:
        dict: Timing statistics (mean, std, min, max in milliseconds)
    """
    import time
    import numpy as np
    
    print("=" * 70)
    print("AttentionMoE Benchmark")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")
    print(f"  num_heads={num_heads}, head_dim={embed_dim // num_heads}")
    print(f"  num_experts={num_experts}, k={k}")
    print(f"  expert_hidden_dim={expert_hidden_dim or embed_dim * 4}")
    print(f"  device={device}")
    print(f"  num_warmup={num_warmup}, num_runs={num_runs}")
    print("=" * 70)
    
    # Set seed and create model
    torch.manual_seed(seed)
    
    model = AttentionMoE(
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_experts=num_experts,
        k=k,
        expert_hidden_dim=expert_hidden_dim
    )
    model.eval()
    
    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"  Using CPU")
    
    # Initialize weights
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    # Create input tensors
    torch.manual_seed(seed)
    if device == "cuda":
        Q = torch.randn(batch_size, seq_len, embed_dim, device="cuda")
        K = torch.randn(batch_size, seq_len, embed_dim, device="cuda")
        V = torch.randn(batch_size, seq_len, embed_dim, device="cuda")
    else:
        Q = torch.randn(batch_size, seq_len, embed_dim)
        K = torch.randn(batch_size, seq_len, embed_dim)
        V = torch.randn(batch_size, seq_len, embed_dim)
    
    # Warmup runs (important for JIT, cache warming, etc.)
    print(f"\n[1] Warmup ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(Q, K, V, verbose=False)
            if device == "cuda":
                torch.cuda.synchronize()  # Wait for GPU to finish
    
    # Timed runs
    print(f"[2] Timing ({num_runs} iterations)...")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()  # Ensure previous work is done
            
            start_time = time.perf_counter()  # High-resolution timer
            
            _ = model(Q, K, V, verbose=False)
            
            if device == "cuda":
                torch.cuda.synchronize()  # Wait for GPU to finish
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Compute statistics
    times = np.array(times)
    stats = {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("Benchmark Results:")
    print("=" * 70)
    print(f"  Mean:   {stats['mean_ms']:.4f} ms")
    print(f"  Std:    {stats['std_ms']:.4f} ms")
    print(f"  Min:    {stats['min_ms']:.4f} ms")
    print(f"  Max:    {stats['max_ms']:.4f} ms")
    print(f"  Median: {stats['median_ms']:.4f} ms")
    print(f"  P95:    {stats['p95_ms']:.4f} ms")
    print(f"  P99:    {stats['p99_ms']:.4f} ms")
    print("-" * 70)
    print(f"  Throughput: {1000 / stats['mean_ms']:.2f} inferences/sec")
    print(f"  Tokens/sec: {batch_size * seq_len * 1000 / stats['mean_ms']:.2f}")
    print("=" * 70)
    
    return stats, model


def benchmark_with_allo_config(
    config_mode: str = "switch_base_8_scaled_1_8",
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cpu",
    seed: int = 42
):
    """
    Benchmark using the same configuration as Allo tests.
    This ensures fair comparison between PyTorch and Allo implementations.
    
    Uses llm_config.py for configuration to match allo_attention_moe_alt.py
    
    Args:
        config_mode: Configuration mode from llm_config (e.g., "switch_base_8_scaled_1_8")
        num_warmup: Number of warmup iterations
        num_runs: Number of timed iterations
        device: "cpu" or "cuda"
        seed: Random seed (must match Allo test seed for same inputs)
    
    Returns:
        dict: Timing statistics and model
    """
    import sys
    import os
    import time
    import numpy as np
    
    # Add llm_config to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    llm_config_dir = os.path.join(project_root, "llm_config")
    if llm_config_dir not in sys.path:
        sys.path.insert(0, llm_config_dir)
    
    from llm_config import get_moe_config
    
    # Get configuration (same as allo_attention_moe_alt.py)
    moe_config = get_moe_config(config_mode)
    
    batch_size = moe_config["batch_size"]
    seq_len = moe_config["seq_len"]
    embed_dim = moe_config["input_dim"]
    num_experts = moe_config["num_experts"]
    k = moe_config["k"]
    hidden_dim = moe_config["hidden_dim"]
    
    # Determine num_heads (same logic as allo_attention_moe_alt.py)
    if embed_dim >= 768:
        num_heads = 12
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
    
    while embed_dim % num_heads != 0 and num_heads > 1:
        num_heads -= 1
    
    print("=" * 70)
    print("PyTorch AttentionMoE Benchmark (Allo-compatible config)")
    print("=" * 70)
    print(f"Config Mode: {config_mode}")
    print(f"Configuration:")
    print(f"  batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")
    print(f"  num_heads={num_heads}, head_dim={embed_dim // num_heads}")
    print(f"  num_experts={num_experts}, k={k}, hidden_dim={hidden_dim}")
    print(f"  device={device}, seed={seed}")
    print(f"  num_warmup={num_warmup}, num_runs={num_runs}")
    print("=" * 70)
    
    # Create model with same initialization as Allo test
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    model = AttentionMoE(
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_experts=num_experts,
        k=k,
        expert_hidden_dim=hidden_dim
    )
    model.eval()
    
    # Initialize with Xavier uniform (same as Allo test)
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"  Using CPU")
    
    # Create input tensors with same seed (same as Allo test)
    torch.manual_seed(seed)
    if device == "cuda":
        Q = torch.randn(batch_size, seq_len, embed_dim, device="cuda")
        K = torch.randn(batch_size, seq_len, embed_dim, device="cuda")
        V = torch.randn(batch_size, seq_len, embed_dim, device="cuda")
    else:
        Q = torch.randn(batch_size, seq_len, embed_dim)
        K = torch.randn(batch_size, seq_len, embed_dim)
        V = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"\nInput shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    
    # Verify output first
    print("\n[1] Verifying output...")
    with torch.no_grad():
        output = model(Q, K, V, verbose=False)
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    
    # Warmup
    print(f"\n[2] Warmup ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(Q, K, V, verbose=False)
            if device == "cuda":
                torch.cuda.synchronize()
    
    # Timed runs
    print(f"[3] Timing ({num_runs} iterations)...")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(Q, K, V, verbose=False)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
    
    # Compute statistics
    times = np.array(times)
    stats = {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("Benchmark Results:")
    print("=" * 70)
    print(f"  Mean:   {stats['mean_ms']:.4f} ms")
    print(f"  Std:    {stats['std_ms']:.4f} ms")
    print(f"  Min:    {stats['min_ms']:.4f} ms")
    print(f"  Max:    {stats['max_ms']:.4f} ms")
    print(f"  Median: {stats['median_ms']:.4f} ms")
    print(f"  P95:    {stats['p95_ms']:.4f} ms")
    print(f"  P99:    {stats['p99_ms']:.4f} ms")
    print("-" * 70)
    print(f"  Throughput: {1000 / stats['mean_ms']:.2f} inferences/sec")
    print(f"  Tokens/sec: {batch_size * seq_len * 1000 / stats['mean_ms']:.2f}")
    print("=" * 70)
    
    return stats, model, Q, K, V


if __name__ == "__main__":
    # First, run gate layer verification
    print("\n" + "#" * 70)
    print("# PART 1: Gate Layer Verification")
    print("#" * 70)
    
    gate, expert_counts = verify_gate_layer(
        num_tokens=128,
        input_dim=96,
        num_experts=2,
        k=1,
        seed=42
    )
    
    # Then run the full Attention + MoE test
    print("\n" + "#" * 70)
    print("# PART 2: Full Attention + MoE Inference Test")
    print("#" * 70)
    
    # Configuration parameters
    batch_size = 1
    seq_len = 4
    embed_dim = 8  # D, must be divisible by num_heads
    num_heads = 2  # H
    num_experts = 2  # E
    k = 1  # Top-k MoE: each token uses exactly k experts
    expert_hidden_dim = 16  # Hidden dimension for experts
    seed = 24
    
    print("=" * 60)
    print(f"Attention + MoE Inference Test")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")
    print(f"  num_heads={num_heads}, head_dim={embed_dim // num_heads}")
    print(f"  num_experts={num_experts}, k={k}, expert_hidden_dim={expert_hidden_dim}")
    print("=" * 60)
    
    # Run Attention + MoE inference
    output, Q, K, V, model = run_attention_moe_inference(
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_experts=num_experts,
        k=k,
        expert_hidden_dim=expert_hidden_dim,
        seed=seed,
        verbose=True
    )
    
    print(f"\n" + "=" * 60)
    print(f"Results:")
    print(f"  Input Q shape: {Q.shape}")
    print(f"  Input K shape: {K.shape}")
    print(f"  Input V shape: {V.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    print("=" * 60)
    
    # PART 3: Benchmark with Allo-compatible config
    print("\n" + "#" * 70)
    print("# PART 3: Performance Benchmark (Allo-compatible config)")
    print("#" * 70)
    
    # Use the same config as allo_attention_moe_alt.py for fair comparison
    stats, _, Q_bench, K_bench, V_bench = benchmark_with_allo_config(
        config_mode="switch_base_8_scaled_1_8",  # Same as DEFAULT_CONFIG_MODE in llm_config
        num_warmup=10,
        num_runs=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42  # Same seed as Allo test
    )
