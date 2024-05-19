# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import int8, float32


def test_linear():
    from allo.library.systolic import systolic

    # L, D = 512, 768
    # M0, M1 = 16, 16
    L, D = 8, 8
    M0, M1 = 2, 2
    W_A = np.random.randint(-4, 4, size=(D, 4 * D)).astype(np.int8)
    allo_C = np.zeros((L, 4 * D), dtype=np.int8)

    s = allo.customize(systolic, instantiate=[int8, int8, int8, L, D, 4 * D, M0, M1])
    # CPU testing
    mod = s.build()
    X = np.random.randint(-4, 4, size=(L, D)).astype(np.int8)
    mod(X, W_A, allo_C)
    np_C = X @ W_A
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    print("Passed!")
    print(s.build(target="vhls"))


def test_softmax():
    from allo.library.nn import softmax

    s = allo.customize(softmax, instantiate=[float32, 8])
    mod = s.build()
    inp = np.random.randn(8, 8).astype(np.float32)
    allo_out = mod(inp)
    np_out = np.exp(inp) / np.exp(inp).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")
    print(s.build(target="vhls"))


def test_layernorm():
    from allo.library.nn import layernorm

    L, D = 8, 8
    s = allo.customize(layernorm, instantiate=[float32, L, D])
    mod = s.build()
    inp = np.random.randn(L, D).astype(np.float32)
    gamma = np.random.randn(D).astype(np.float32)
    beta = np.random.randn(D).astype(np.float32)
    allo_out = mod(inp, gamma, beta)
    mean = inp.mean(axis=1)
    mean2 = (inp ** 2).mean(axis=1)
    var = mean2 - mean ** 2
    np_out = gamma * (inp - mean[:, None]) / np.sqrt(var[:, None] + 1e-5) + beta
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")
    print(s.build(target="vhls"))


def test_gelu():
    from allo.library.nn import gelu

    L, D = 8, 8
    s = allo.customize(gelu, instantiate=[float32, L, D])
    mod = s.build()
    inp = np.random.randn(L, D).astype(np.float32)
    allo_out = mod(inp)
    np_out = 0.5 * inp * (1 + np.tanh(0.797885 * (inp + 0.044715 * inp ** 3)))
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")
    print(s.build(target="vhls"))


def test_self_attention():
    def Self_attention[
        Ty, H, L, D
    ](Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D],) -> Ty[L, D]:
        # softmax(QK^T/sqrt(D // H))
        Context: Ty[L, D]

        for h in range(H):
            Q_h: float32[L, D // H]
            K_h: float32[L, D // H]
            V_h: float32[L, D // H]

            for i, j in allo.grid(D, D // H, name="mha_split"):
                Q_h[i, j] = Q[i, h * 64 + j]
                K_h[i, j] = K[i, h * 64 + j]
                V_h[i, j] = V[i, h * 64 + j]
            Attn = Attention_layer(Q_h, K_h)
            Attn = Softmax_layer(Attn)
            C_h = Context_layer(Attn, V_h)

            for i, j in allo.grid(L, D // H, name="mha_merge"):
                Context[i, h * 64 + j] = C_h[i, j]

        return Context


if __name__ == "__main__":
    test_gelu()
