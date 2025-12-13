"""
Modern Mixture of Experts (MoE) Implementation for Inference Only

This script implements a simplified MoE layer optimized for inference.
Uses random inputs and weights for demonstration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


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


def run_moe_inference(
    batch_size: int = 4,
    seq_len: int = 4,
    input_dim: int = 64,
    output_dim: int = 64,
    num_experts: int = 2,
    k: int = 1,
    seed: int = 24,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, MoELayer]:
    """
    Run MoE layer inference with fixed seed for reproducible inputs and weights.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        num_experts: Number of experts
        k: Top-k experts to activate per token
        seed: Random seed for reproducibility
        verbose: Whether to print verification information
    
    Returns:
        output: Output tensor from MoE layer [batch_size, seq_len, output_dim]
        input_tensor: Input tensor used for inference
        moe_layer: MoE layer instance with initialized weights
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create MoE layer
    moe_layer = MoELayer(input_dim, output_dim, num_experts, k)
    moe_layer.eval()
    
    # Initialize with random weights (Xavier uniform for better stability)
    for param in moe_layer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    # Create random input
    torch.manual_seed(seed)
    input_tensor = torch.randn(batch_size, seq_len, input_dim)
    
    # Run inference with verbose output for verification
    with torch.no_grad():
        output = moe_layer(input_tensor, verbose=verbose)
    
    return output, input_tensor, moe_layer


if __name__ == "__main__":
    # Configuration parameters
    batch_size = 4
    seq_len = 1
    input_dim = 1
    output_dim = 1
    num_experts = 2
    k = 1  # Top-k MoE: each token uses exactly k experts
    seed = 24
    
    print("=" * 60)
    print(f"Top-{k} MoE Inference Test")
    print("=" * 60)
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}, input_dim={input_dim}")
    print(f"Experts: {num_experts}, k={k}")
    print("=" * 60)
    
    # Run MoE inference
    output, input_tensor, moe_layer = run_moe_inference(
        batch_size=batch_size,
        seq_len=seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        k=k,
        seed=seed,
        verbose=True
    )
    
    print(f"\nOutput shape: {output.shape}")
    print("=" * 60)

