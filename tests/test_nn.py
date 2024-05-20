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


def test_linear_float():
    from allo.library.systolic import systolic

    # L, D = 512, 768
    # M0, M1 = 16, 16
    L, D = 8, 8
    M0, M1 = 2, 2
    W_A = np.random.randn(D, 4 * D).astype(np.float32)
    allo_C = np.zeros((L, 4 * D), dtype=np.float32)

    s = allo.customize(
        systolic, instantiate=[float32, float32, float32, L, D, 4 * D, M0, M1]
    )
    # CPU testing
    mod = s.build()
    X = np.random.randn(L, D).astype(np.float32)
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
    mean2 = (inp**2).mean(axis=1)
    var = mean2 - mean**2
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
    np_out = 0.5 * inp * (1 + np.tanh(0.797885 * (inp + 0.044715 * inp**3)))
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")
    print(s.build(target="vhls"))


def test_sdp():
    from allo.library.nn import scaled_dot_product_attention

    H, L, D = 2, 8, 8
    s = allo.customize(scaled_dot_product_attention, instantiate=[float32, H, L, D])
    mod = s.build()
    Q = np.random.randn(L, D).astype(np.float32)
    K = np.random.randn(L, D).astype(np.float32)
    V = np.random.randn(L, D).astype(np.float32)
    allo_out = mod(Q, K, V)

    def sdp(Q, K, V):
        context = np.zeros(Q.shape)
        h_d = D // H
        for i in range(H):
            # split Q, K, V
            Q_h = Q[:, i * h_d : (i + 1) * h_d]
            K_h = K[:, i * h_d : (i + 1) * h_d]
            V_h = V[:, i * h_d : (i + 1) * h_d]
            # compute attention
            attention = np.matmul(Q_h, K_h.T)
            Y = np.exp(attention) / np.exp(attention).sum(axis=1, keepdims=True)
            context_i = np.matmul(Y, V_h)
            context[:, i * h_d : (i + 1) * h_d] = context_i
        return context

    np_out = sdp(Q, K, V)
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")


if __name__ == "__main__":
    test_sdp()
