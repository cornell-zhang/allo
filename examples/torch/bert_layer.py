# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# https://github.com/chhzh123/ptc-tutorial/blob/master/tutorial.ipynb

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import allo
import math

hidden_size = 768
n_heads = 12
batch_size = 2
intermediate_size = 3072
seq_len = 512


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_heads):
        super().__init__()
        self.attention = Attention(hidden_size, intermediate_size, n_heads)
        self.ffn = FFN(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        ffn_output = self.ffn(attention_output)
        return ffn_output


class Attention(nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_heads):
        super().__init__()
        self.self_attn = SelfAttention(hidden_size, n_heads)
        self.proj = Projection(hidden_size, hidden_size)

    def forward(self, hidden_states):
        self_output = self.self_attn(hidden_states)
        attention_output = self.proj(self_output, hidden_states)
        return attention_output


class FFN(nn.Module):
    """Feed forward network (FFN) with GELU activation"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.projection = Projection(intermediate_size, hidden_size)

    def forward(self, data):
        out = self.linear1(data)
        out = self.activation(out)
        out = self.projection(out, data)
        return out


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, p=0.1):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p)
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
        attn_probs = self.dropout(attn_probs)
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


class Projection(nn.Module):
    def __init__(self, intermediate_size, hidden_size, p=0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(p)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


model = TransformerLayer(hidden_size, intermediate_size, n_heads).eval()
example_inputs = [torch.rand(batch_size, seq_len, hidden_size)]
llvm_mod = allo.frontend.from_pytorch(
    model, example_inputs=example_inputs, verbose=True
)

golden = model(*example_inputs)
np_inputs = [x.detach().numpy() for x in example_inputs]
res = llvm_mod(*np_inputs)
np.testing.assert_allclose(res, golden.detach().numpy(), atol=1e-3)
