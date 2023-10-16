# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# This file is the library of Allo frontend functions, which is responsible for converting PyTorch leaf modules to Allo representation.


def CoreAttention(node):
    """
    This function in Allo representation is equivalent to the following PyTorch snippet,
    describing the Attention computation core in GPT Attention:
    ```
    def forward(self, q, k_cache, v_cache, n_tokens):
        cmp_k = k_cache[:, :, : n_tokens + 1, :]
        cmp_v = v_cache[:, :, : n_tokens + 1, :]
        attn = torch.matmul(q, cmp_k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = torch.matmul(attn, cmp_v)
        return attn
    ```
    """
    attn = f"{node.name}_atten"
    sumRow = f"{node.name}_sumRow"
    shape = tuple(node.meta["tensor_meta"][0])
    res = f"tmp: float32{list(shape)} = 0.0\n"
    res += f"  {attn}: float32{list(shape)} = 0.0\n"
    res += f"  {sumRow}: float32{list(shape[:3])} = 0.0\n"
    res += f"  for i, j, p in dsl.grid{shape[:3]}:\n"
    res += f"    for m in range(0, {node.args[-1]} + 1):\n"
    res += f"      for l in range({shape[-1]}):\n"
    res += f"        tmp[i, j, p, m] += {node.args[0]}[i, j, p, l] * {node.args[1]}[i, j, m, l]\n"
    res += "      tmp[i, j, p, m] = dsl.exp(tmp[i, j, p, m])\n"
    res += f"      {sumRow}[i, j, p] += tmp[i, j, p, m]\n"
    res += f"  for i, j, p in dsl.grid{shape[:3]}:\n"
    res += f"    for m in range(0, {node.args[-1]} + 1):\n"
    res += f"      tmp[i, j, p, m] = tmp[i, j, p, m] / {sumRow}[i, j, p]\n"

    res += f"  for i, j, p, l in dsl.grid{shape}:\n"
    res += f"    for m in range(0, {node.args[-1]} + 1):\n"
    res += f"      {attn}[i, j, p, l] += tmp[i, j, p, m] * {node.args[2]}[i, j, m, l]\n"
    res += f"  {node.name} = dsl.copy({attn})"
    return res


def KVCache(node):
    """
    This function in Allo representation is equivalent to the following PyTorch snippet,
    describing the KV_cache update in GPT Attention:
    ```
    def forward(self, k, k_cache, v, v_cache, n_tokens):
        k_cache[:, :, n_tokens, :] = k[:, :, 0, :]
        v_cache[:, :, n_tokens, :] = v[:, :, 0, :]
        return k_cache, v_cache
    ```
    """
    shape = tuple(node.meta["tensor_meta"][0][0])
    loop_1 = shape[:2]
    loop_2 = f"({node.args[-1]}, {node.args[-1]} + 1)"
    res = f"for i, j in dsl.grid{loop_1}:\n"
    res += f"    for p in range{loop_2}:\n"
    res += f"      for m in range({shape[-1]}):\n"
    res += f"        {node.args[1]}[i, j, p, m] = {node.args[0]}[i, j, 0, m]\n"
    res += f"        {node.args[3]}[i, j, p, m] = {node.args[2]}[i, j, 0, m]\n"
    res += f"  {node.name}_0 = dsl.copy({node.args[1]})\n"
    res += f"  {node.name}_1 = dsl.copy({node.args[3]})"
    return res
