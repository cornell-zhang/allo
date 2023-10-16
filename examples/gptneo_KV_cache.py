# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTneo(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layers, max_seq_len=25):
        super(GPTneo, self).__init__()
        self.layers = n_layers
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(n_embd, n_head, n_embd * 4, max_seq_len)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x, kcache=None, vcache=None, n=0):
        if kcache is None:
            kcache = [None] * len(self.transformer)
            vcache = [None] * len(self.transformer)
        k_presents = [] if kcache is not None else None
        v_presents = [] if vcache is not None else None
        for i in range(self.layers):
            block = self.transformer[i]
            x, updated_kcache, updated_vcache = block(x, kcache[i], vcache[i], n)
            k_presents.append(updated_kcache)
            v_presents.append(updated_vcache)

        x = self.ln_f(x)
        x = self.lm_head(x)
        return x, k_presents, v_presents


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, ffn_hidden_dim, max_seq_len=25):
        super(TransformerBlock, self).__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, num_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, ffn_hidden_dim, n_embd)

    def forward(self, x, k_cache=None, v_cache=None, n=0):
        ln_1 = self.ln_1(x)
        attn_output, k_cache_updated, v_cache_updated = self.attn(
            ln_1, k_cache, v_cache, n
        )
        out1 = x + attn_output

        ln_2 = self.ln_2(out1)
        ffn_output = self.mlp(ln_2)
        out2 = out1 + ffn_output
        return out2, k_cache_updated, v_cache_updated


class MLP(nn.Module):
    def __init__(self, n_embd, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x


class Attention_core(nn.Module):
    def __init__(self):
        super(Attention_core, self).__init__()

    def forward(self, q, k_cache, v_cache, n_tokens):
        cmp_k = k_cache[:, :, : n_tokens + 1, :]
        cmp_v = v_cache[:, :, : n_tokens + 1, :]
        attn = torch.matmul(q, cmp_k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = torch.matmul(attn, cmp_v)
        return attn


class KV_cache(nn.Module):
    def __init__(self):
        super(KV_cache, self).__init__()

    def forward(self, k, k_cache, v, v_cache, n_tokens):
        k_cache[:, :, n_tokens, :] = k[:, :, 0, :]
        v_cache[:, :, n_tokens, :] = v[:, :, 0, :]
        return k_cache, v_cache


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, max_seq_len=25):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        self.buffer = max_seq_len
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn = Attention_core()
        self.kvcache = KV_cache()

    def mask(self, x):
        ones = torch.ones(x.size(1), x.size(1))
        causal_mask = (1 - torch.tril(ones)) * -1e38
        return causal_mask

    def split_heads(self, x):
        # x: (batch_size, seq_len, hidden_size)
        new_shape = x.shape[:-1] + (self.num_heads, -1)
        x = x.view(new_shape)
        # output: (bs, head, seq, hs // head)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product(self, q, k, v, x):
        attn_score = torch.matmul(q, k.transpose(-2, -1))
        attn_score += self.mask(x)
        # (bs, head, seq, seq)
        attn_probs = F.softmax(attn_score, dim=-1)
        # (bs, head, seq, hs // head)
        attn = torch.matmul(attn_probs, v)
        return attn

    def forward(self, x, k_cache=None, v_cache=None, n=0):
        # qkv layers
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # split heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        if k_cache is not None:
            k_cache, v_cache = self.kvcache(k, k_cache, v, v_cache, n)
            output = self.attn(q, k_cache, v_cache, n)
        else:
            output = self.scaled_dot_product(q, k, v, x)
            shape = k.shape[:2] + (self.buffer, k.shape[-1])
            zeros = torch.zeros(shape)
            k_cache = torch.cat((k, zeros), dim=-2)
            v_cache = torch.cat((v, zeros), dim=-2)

        # output: (bs, seq, head, hs // head)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        output = self.out_proj(output)
        return output, k_cache, v_cache


class Embedding(nn.Module):
    def __init__(self, vocab_size, n_position, n_embd):
        super(Embedding, self).__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_position, n_embd)

    def forward(self, input_ids, kvcache=None):
        if kvcache is None:
            wpe_out = self.wpe(torch.arange(len(input_ids)))
            input_tensor = input_ids
        else:
            wpe_out = self.wpe(torch.tensor(len(input_ids) - 1))
            input_tensor = input_ids[-1:]
        input_tensor = torch.tensor(input_tensor, dtype=torch.long)
        x = self.wte(input_tensor) + wpe_out
        return x.unsqueeze(0)


def generate(inputs, model, embeddings, n_tokens_to_generate):
    with torch.no_grad():
        kcache, vcache = None, None
        for num in tqdm(
            range(n_tokens_to_generate), "generating"
        ):  # auto-regressive decode loop
            input_length = len(inputs) - 1
            x = embeddings(inputs, kcache)
            if kcache is None:
                llvm_mod = allo.frontend.from_pytorch(
                    model,
                    example_inputs=[x],
                    verbose=False,
                )
                x = x.detach().numpy()
                output = llvm_mod(x)
            else:
                if num == 1:
                    llvm_mod = allo.frontend.from_pytorch(
                        model,
                        example_inputs=[x, kcache, vcache, input_length],
                        customed_leaf_module=(Attention_core, KV_cache),
                        verbose=False,
                    )
                x = x.detach().numpy()
                for i in range(len(kcache)):
                    kcache[i] = kcache[i].detach().numpy()
                    vcache[i] = vcache[i].detach().numpy()
                output = llvm_mod(x, *kcache, *vcache, input_length)

            for i in range(len(output)):
                output[i] = torch.tensor(output[i])
            logits, kcache, vcache = output[0], output[1:13], output[13:25]
            next_id = torch.argmax(logits[0, -1, :]).item()
            inputs.append(next_id)  # append prediction to input
        return inputs  # only return generated ids


def load_weights(model, emb_model):
    GPT_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").eval()
    dic_from = GPT_model.state_dict()
    dic_to = model.state_dict()
    dic_emb = emb_model.state_dict()

    for k, weight in dic_from.items():
        new_name = k.replace("h.", "").replace("attention.", "")
        if "wte" in k or "wpe" in k:
            dic_emb[k[12:]] = weight
        if "ln_f" in k:
            dic_to[k[12:]] = weight
        if new_name in dic_to:
            dic_to[new_name] = weight

    model.load_state_dict(dic_to)
    emb_model.load_state_dict(dic_emb)


vocab_size = 50257
n_embd = 768
n_head = 12
n_layers = 12
n_position = 2048

max_seq_len = 13

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neo-125M", is_split_into_words=True
)
input_text = "This cute cat is"
in_tokens = tokenizer.encode(input_text)


module = GPTneo(vocab_size, n_embd, n_head, n_layers, max_seq_len).eval()
embeddings = Embedding(vocab_size, n_position, n_embd).eval()
load_weights(module, embeddings)
out = generate(in_tokens, module, embeddings, max_seq_len)
generated_text = tokenizer.decode(out)
full_string = "".join(generated_text)
print(out)
print(full_string)
