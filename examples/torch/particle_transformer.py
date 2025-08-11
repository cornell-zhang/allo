import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import allo


class ConstituentNet(nn.Module):
    def __init__(self, in_dim, embbed_dim, num_heads, num_classes, num_transformers):
        super(ConstituentNet, self).__init__()
        self.embedding = nn.Linear(in_dim, embbed_dim)
        self.norm = nn.BatchNorm1d(embbed_dim)
        self.linear = nn.Linear(embbed_dim, num_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embbed_dim))
        self.transformers = nn.ModuleList(
            [Transformer(embbed_dim, num_heads) for _ in range(num_transformers)]
        )
        # For cls_token extraction
        self.slicer = SliceClsToken()

    def forward(self, x):
        m_batch, _, _ = x.size()
        out = self.embedding(x)
        cls_tokens = self.cls_token.repeat(m_batch, 1, 1)
        out = torch.cat((cls_tokens, out), dim=1)

        for transformer in self.transformers:
            out = transformer(out)

        out = self.slicer(out)
        out = self.norm(out)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)


class Transformer(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(Transformer, self).__init__()
        self.self_attention = SelfAttention(in_dim, num_heads)
        self.norm = nn.BatchNorm1d(in_dim)
        self.activ = nn.ReLU()
        self.ffn = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, x):
        x = self.self_attention(x)
        out = self.norm(x.transpose(1, 2)).transpose(1, 2)
        out = self.activ(out)
        out = self.ffn(out)
        out = x + out
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.latent_dim = in_dim
        self.heads = num_heads
        self.head_dim = self.latent_dim // num_heads

        self.norm = nn.BatchNorm1d(in_dim)
        self.q = nn.Linear(in_dim, in_dim, bias=False)
        self.k = nn.Linear(in_dim, in_dim, bias=False)
        self.v = nn.Linear(in_dim, in_dim, bias=False)
        self.out = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        B, L, _ = x.size()
        out = self.norm(x.transpose(1, 2)).transpose(1, 2)
        q = self.q(out).view(B, L, self.heads, -1)
        k = self.k(out).view(B, L, self.heads, -1)
        v = self.v(out).view(B, L, self.heads, -1)

        Q_ = q.permute(0, 2, 1, 3)
        K_ = k.permute(0, 2, 3, 1)
        energy = torch.matmul(Q_, K_)
        attn = F.softmax(energy / (self.head_dim**0.5), dim=-1)

        V_ = v.permute(0, 2, 1, 3)
        ctx = torch.matmul(attn, V_)
        out = ctx.permute(0, 2, 1, 3).reshape(B, L, -1)
        out = self.out(out)
        return out + x


class SliceClsToken(nn.Module):
    def forward(self, inp):
        # inp: (B, L, C), take CLS at position 0 -> (B, C)
        return inp[:, 0, :]


if __name__ == "__main__":
    batch_size = 2
    num_particles = 8
    in_dim = 3
    embbed_dim = 16
    num_heads = 2
    num_classes = 5
    num_transformers = 1

    model = ConstituentNet(
        in_dim, embbed_dim, num_heads, num_classes, num_transformers
    ).eval()
    example_inputs = [torch.randn(batch_size, num_particles, in_dim)]
    golden = model(*example_inputs).detach().numpy()
    np_inp = example_inputs[0].detach().numpy()

    mod = allo.frontend.from_pytorch(
        model,
        example_inputs=example_inputs,
        leaf_modules=(SliceClsToken,),
        verbose=False,
    )

    out = mod(np_inp)
    np.testing.assert_allclose(out, golden, atol=1e-5, rtol=1e-5)
    print("Passed!")
