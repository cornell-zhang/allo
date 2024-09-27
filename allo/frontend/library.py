# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable = unsubscriptable-object, unsupported-assignment-operation
# This file is the library of Allo frontend functions, which is responsible for converting PyTorch leaf modules to Allo representation.
from .. import dsl
from ..ir.types import float32, int32


def KVCache_lib(s_0, s_1, s_2, s_3):
    def KVCache(
        inp: float32[s_0, s_1, 1, s_3],
        cache: float32[s_0, s_1, s_2, s_3],
        n_tokens: int32,
    ) -> float32[s_0, s_1, s_2, s_3]:
        for i, j in dsl.grid(s_0, s_1):
            for m in range(s_3):
                cache[i, j, n_tokens, m] = inp[i, j, 0, m]
        return cache

    return KVCache


def CoreAttention_lib(s_0, s_1, s_2, s_3):
    def CoreAttention(
        q: float32[s_0, s_1, 1, s_3],
        k_cache: float32[s_0, s_1, s_2, s_3],
        v_cache: float32[s_0, s_1, s_2, s_3],
        n_tokens: int32,
    ) -> float32[s_0, s_1, 1, s_3]:
        Attn: float32[s_0, s_1, 1, s_3] = 0.0
        Attn_tmp: float32[s_0, s_1, 1, s_3] = 0.0
        sumRow: float32[s_0, s_1, 1] = 0.0
        for i, j, p in dsl.grid(s_0, s_1, 1):
            for m in range(0, n_tokens + 1):
                for l in range(s_3):
                    Attn_tmp[i, j, p, m] += q[i, j, p, l] * k_cache[i, j, m, l]
                Attn_tmp[i, j, p, m] = dsl.exp(Attn_tmp[i, j, p, m])
                sumRow[i, j, p] += Attn_tmp[i, j, p, m]
        for i, j, p in dsl.grid(s_0, s_1, 1):
            for m in range(0, n_tokens + 1):
                Attn_tmp[i, j, p, m] = Attn_tmp[i, j, p, m] / sumRow[i, j, p]
        for i, j, p, l in dsl.grid(s_0, s_1, 1, s_3):
            for m in range(0, n_tokens + 1):
                Attn[i, j, p, l] += Attn_tmp[i, j, p, m] * v_cache[i, j, m, l]
        return Attn

    return CoreAttention
