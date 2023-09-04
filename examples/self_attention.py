# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch.nn as nn
import allo
import math


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.n_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)

    def permute_for_scores(self, x):
        # x: (batch_size, seq_len, hidden_size)
        new_shape = x.shape[:-1] + (self.n_heads, -1)
        x = x.view(new_shape)
        # output: (bs, head, seq, hs // head)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product(self, q, k, v):
        # (bs, head, seq, hs // head)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.attention_head_size
        )
        # (bs, head, seq, seq)
        attn_probs = F.softmax(attn_score, dim=-1)
        attn_probs = F.dropout(attn_probs, 0.1)
        # (bs, head, seq, hs // head)
        attn = torch.matmul(attn_probs, v)
        return attn

    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, hidden_size)
        # qkv layers
        q = self.permute_for_scores(self.q_proj(hidden_states))
        k = self.permute_for_scores(self.k_proj(hidden_states))
        v = self.permute_for_scores(self.v_proj(hidden_states))
        # core attention
        output = self.scaled_dot_product(q, k, v)
        # output: (bs, seq, head, hs // head)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        return output


hidden_size = 768
n_heads = 12
batch_size = 2
seq_len = 512

model = SelfAttention(hidden_size, n_heads).eval()
example_inputs = [torch.rand(batch_size, seq_len, hidden_size)]
llvm_mod = allo.frontend.from_pytorch(
    model, example_inputs=example_inputs, verbose=True
)
