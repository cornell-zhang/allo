# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.backend import hls
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GPT2(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layers):
        super(GPT2, self).__init__()
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, n_embd * 4) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.fc = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.fc(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, ffn_hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(n_embd, num_heads)
        self.norm1 = nn.LayerNorm(n_embd)
        self.ffn = FFN(n_embd, ffn_hidden_dim, n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attn_output = self.attention(x)
        out1 = x + attn_output
        out1 = self.norm1(out1)

        ffn_output = self.ffn(out1)
        out2 = out1 + ffn_output
        out2 = self.norm2(out2)
        return out2


class FFN(nn.Module):
    def __init__(self, n_embd, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads

        self.linear_q = nn.Linear(n_embd, n_embd)
        self.linear_k = nn.Linear(n_embd, n_embd)
        self.linear_v = nn.Linear(n_embd, n_embd)

        self.linear_out = nn.Linear(n_embd, n_embd)

    def mask(self, x):
        ones = torch.ones(x.size(1), x.size(1))
        causal_mask = (1 - torch.tril(ones)) * -1e10
        return causal_mask

    def split_heads(self, x):
        # x: (batch_size, seq_len, hidden_size)
        new_shape = x.shape[:-1] + (self.num_heads, -1)
        x = x.view(new_shape)
        # output: (bs, head, seq, hs // head)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product(self, q, k, v, x):
        # (bs, head, seq, hs // head)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        ) + self.mask(x)
        # (bs, head, seq, seq)
        attn_probs = F.softmax(attn_score, dim=-1)
        # (bs, head, seq, hs // head)
        attn = torch.matmul(attn_probs, v)
        return attn

    def forward(self, x):
        # qkv layers
        q = self.split_heads(self.linear_q(x))
        k = self.split_heads(self.linear_k(x))
        v = self.split_heads(self.linear_v(x))
        # core attention
        output = self.scaled_dot_product(q, k, v, x)
        # output: (bs, seq, head, hs // head)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        output = self.linear_out(output)
        return output


# Large size
# vocab_size = 50257
# n_embd = 768
# n_head = 12
# n_layers = 12
# n_seq = 1024
vocab_size = 2
n_embd = 4
n_head = 2
n_layers = 1
n_seq = 4
batch_size = 2

module = GPT2(vocab_size, n_embd, n_head, n_layers).eval()
example_inputs = [torch.rand(batch_size, n_seq, n_embd)]
golden = module(*example_inputs)
np_inputs = [x.detach().numpy() for x in example_inputs]

llvm_mod = allo.frontend.from_pytorch(
    module,
    example_inputs=example_inputs,
    verbose=True,
)
res = llvm_mod(*np_inputs)
np.testing.assert_allclose(res, golden.detach().numpy(), atol=1e-3)
print("Test passed!")

# generate HLS module
mod = allo.frontend.from_pytorch(module, example_inputs=example_inputs, target="vhls")
print(mod.hls_code)

if hls.is_available("vitis_hls"):
    allo_C = output = np.zeros((batch_size, n_seq, vocab_size), dtype=np.float32)
    mod = allo.frontend.from_pytorch(
        module, example_inputs=example_inputs, target="vitis_hls", mode="csim"
    )
    mod(*np_inputs, allo_C)
    np.testing.assert_allclose(allo_C, golden.detach().numpy(), atol=1e-3)
    print("Test passed!")
