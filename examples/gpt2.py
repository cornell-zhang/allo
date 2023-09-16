import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Define the Feed-Forward Network module
class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# Define the Multi-Head Attention module
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)
        self.linear_v = nn.Linear(input_dim, input_dim)

        self.linear_out = nn.Linear(input_dim, input_dim)

    def split_heads(self, x):
        # x: (batch_size, seq_len, hidden_size)
        new_shape = (x.size(0), -1, self.num_heads, self.head_dim)
        x = x.view(new_shape)
        # output: (bs, head, seq, hs // head)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product(self, q, k, v):
        # (bs, head, seq, hs // head)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (bs, head, seq, seq)
        attn_probs = F.softmax(attn_score, dim=-1)
        # (bs, head, seq, hs // head)
        attn = torch.matmul(attn_probs, v)
        return attn

    def forward(self, x, mask=None):
        # qkv layers
        q = self.split_heads(self.linear_q(x))
        k = self.split_heads(self.linear_k(x))
        v = self.split_heads(self.linear_v(x))
        # core attention
        output = self.scaled_dot_product(q, k, v)
        # output: (bs, seq, head, hs // head)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        output = self.linear_out(output)
        return output


# Define the Transformer Block module
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ffn_hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.ffn = FFN(input_dim, ffn_hidden_dim, input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        attn_output = self.attention(x)
        out1 = x + attn_output
        out1 = self.norm1(out1)

        ffn_output = self.ffn(out1)
        out2 = out1 + ffn_output
        out2 = self.norm2(out2)
        return out2


# Define the GPT2 model
class GPT2(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(GPT2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, d_model * 4) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.fc(x)
        return x


# Usage example
vocab_size = 10000  # Replace with your actual vocabulary size
d_model = 512  # Replace with your desired model dimension
nhead = 8  # Replace with the number of attention heads
num_layers = 12  # Replace with the number of transformer layers

model = GPT2(vocab_size, d_model, nhead, num_layers)
input_ids = torch.tensor([1, 2, 3, 4, 5])  # Replace with your input sequence
output = model(input_ids)
print(output)
