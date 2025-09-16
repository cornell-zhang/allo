# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
import allo.library.nn as nn
from allo.ir.types import int8, float32
import pytest


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


def test_builtin_linear():
    M, N, K = 16, 32, 16
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(N, K).astype(np.float32)
    b = np.random.randn(
        N,
    ).astype(np.float32)
    s = allo.customize(nn.linear2d, instantiate=[float32, float32, float32, M, N, K])
    mod = s.build(target="llvm")
    Z = mod(X, W, b)
    np.testing.assert_allclose(Z, np.dot(X, W.T) + b, atol=1e-5)
    print("Passed!")


def test_cascaded_linear():
    BS, M0, M1, M2 = 16, 32, 16, 10
    X = np.random.randn(BS, M0).astype(np.float32)
    W0 = np.random.randn(M1, M0).astype(np.float32)
    W1 = np.random.randn(M2, M1).astype(np.float32)
    b0 = np.random.randn(
        M1,
    ).astype(np.float32)
    b1 = np.random.randn(
        M2,
    ).astype(np.float32)

    def cascaded_linear(
        X: float32[BS, M0],
        # weights are transposed
        W0: float32[M1, M0],
        W1: float32[M2, M1],
        b0: float32[M1],
        b1: float32[M2],
    ) -> float32[BS, M2]:
        Z0 = nn.linear2d[float32, float32, float32, BS, M1, M0](X, W0, b0)
        Z1 = nn.linear2d[float32, float32, float32, BS, M2, M1](Z0, W1, b1)
        return Z1

    s = allo.customize(cascaded_linear)
    s.compose(nn.linear2d, instantiate=[float32, float32, float32, BS, M1, M0])
    s.compose(nn.linear2d, id="1", instantiate=[float32, float32, float32, BS, M2, M1])
    print(s.module)
    mod = s.build(target="llvm")
    Z = mod(X, W0, W1, b0, b1)
    np.testing.assert_allclose(Z, np.dot(np.dot(X, W0.T) + b0, W1.T) + b1, atol=1e-4)
    print("Passed!")


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


def np_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def test_softmax():
    from allo.library.nn import softmax

    s = allo.customize(softmax, instantiate=[float32, 8])
    mod = s.build()
    inp = np.random.randn(8, 8).astype(np.float32)
    inp = 1000 * inp
    allo_out = mod(inp)
    np_out = np_softmax(inp)
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")
    print(s.build(target="vhls"))


def np_layernorm(inp, gamma, beta):
    mean = inp.mean(axis=1)
    mean2 = (inp**2).mean(axis=1)
    var = mean2 - mean**2
    np_out = gamma * (inp - mean[:, None]) / np.sqrt(var[:, None] + 1e-5) + beta
    return np_out


def test_layernorm():
    from allo.library.nn import layer_norm

    L, D = 8, 8
    s = allo.customize(layer_norm, instantiate=[float32, L, D])
    mod = s.build()
    inp = np.random.randn(L, D).astype(np.float32)
    gamma = np.random.randn(D).astype(np.float32)
    beta = np.random.randn(D).astype(np.float32)
    allo_out = mod(inp, gamma, beta)
    np_out = np_layernorm(inp, gamma, beta)
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")
    print(s.build(target="vhls"))


def test_gelu():
    from allo.library.nn import GeLU

    L, D = 8, 8
    s = allo.customize(GeLU, instantiate=[float32, L, D])
    mod = s.build()
    inp = np.random.randn(L, D).astype(np.float32)
    allo_out = mod(inp)
    np_out = 0.5 * inp * (1 + np.tanh(0.797885 * (inp + 0.044715 * inp**3)))
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")
    print(s.build(target="vhls"))


def sdp(Q, K, V, H, D):
    context = np.zeros(Q.shape)
    h_d = D // H
    for i in range(H):
        # split Q, K, V
        Q_h = Q[:, i * h_d : (i + 1) * h_d]
        K_h = K[:, i * h_d : (i + 1) * h_d]
        V_h = V[:, i * h_d : (i + 1) * h_d]
        # compute attention
        attention = np.matmul(Q_h, K_h.T)
        Y = np_softmax(attention)
        context_i = np.matmul(Y, V_h)
        context[:, i * h_d : (i + 1) * h_d] = context_i
    return context


def test_sdp():
    from allo.library.nn import scaled_dot_product_attention

    M0, M1 = 2, 2
    H, L, D = 2, 8, 8
    s = allo.customize(
        scaled_dot_product_attention, instantiate=[float32, H, L, D, M0, M1]
    )
    mod = s.build()
    Q = np.random.randn(L, D).astype(np.float32)
    K = np.random.randn(L, D).astype(np.float32)
    V = np.random.randn(L, D).astype(np.float32)
    allo_out = mod(Q, K, V)
    np_out = sdp(Q, K, V, H, D)
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")


def test_bert():
    from allo.library.systolic import systolic
    from allo.library.nn import (
        scaled_dot_product_attention,
        layer_norm,
        GeLU,
        residual_add,
    )

    H, L, D, Dffn = 2, 8, 8, 16
    M0, M1 = 2, 2

    def BertLayer[
        Ty, H, L, D, Dffn, M0, M1
    ](
        X: "Ty[L, D]",
        Wq: "Ty[D, D]",
        Wk: "Ty[D, D]",
        Wv: "Ty[D, D]",
        Wp: "Ty[D, D]",
        W1: "Ty[D, Dffn]",
        W2: "Ty[Dffn, D]",
        gamma1: "Ty[D]",
        beta1: "Ty[D]",
        gamma2: "Ty[D]",
        beta2: "Ty[D]",
    ) -> "Ty[L, D]":
        # 1. Bert Attention
        # 1.0 project Q, K, V
        Q: Ty[L, D] = 0
        K: Ty[L, D] = 0
        V: Ty[L, D] = 0
        systolic[Ty, Ty, Ty, L, D, D, M0, M1, "Q"](X, Wq, Q)
        systolic[Ty, Ty, Ty, L, D, D, M0, M1, "K"](X, Wk, K)
        systolic[Ty, Ty, Ty, L, D, D, M0, M1, "V"](X, Wv, V)
        # 1.1 self attention
        attn = scaled_dot_product_attention[Ty, H, L, D, M0, M1](Q, K, V)
        # 1.2 output dense
        O_proj: Ty[L, D] = 0
        systolic[Ty, Ty, Ty, L, D, D, M0, M1, "P"](attn, Wp, O_proj)
        # 1.3 Residual layer
        res_attn = residual_add[Ty, L, D, "res_attn"](O_proj, X)
        # 1.4 layer norm
        ln = layer_norm[Ty, L, D, "ln1"](res_attn, gamma1, beta1)
        # 2. Feed Forward Network
        # 2.1 ffn dense 1
        ffn1: Ty[L, Dffn] = 0
        systolic[Ty, Ty, Ty, L, D, Dffn, M0, M1, "ffn1"](ln, W1, ffn1)
        # 2.2 gelu layer
        gelu_outp = GeLU[Ty, L, Dffn](ffn1)
        # 2.3 ffn dense 2
        ffn2: Ty[L, D] = 0
        systolic[Ty, Ty, Ty, L, Dffn, D, M0, M1, "ffn2"](gelu_outp, W2, ffn2)
        # 2.4 Residual layer
        res_ffn = residual_add[Ty, L, D, "res_ffn"](ffn2, ln)
        # 2.5 layer norm
        ffn_ln_outp = layer_norm[Ty, L, D, "ln2"](res_ffn, gamma2, beta2)
        return ffn_ln_outp

    s = allo.customize(
        BertLayer,
        instantiate=[float32, H, L, D, Dffn, M0, M1],
    )
    mod = s.build()
    X = np.random.randn(L, D).astype(np.float32)
    # weights are supposed to be transposed
    Wq = np.random.randn(D, D).astype(np.float32)
    Wk = np.random.randn(D, D).astype(np.float32)
    Wv = np.random.randn(D, D).astype(np.float32)
    Wp = np.random.randn(D, D).astype(np.float32)
    W1 = np.random.randn(D, Dffn).astype(np.float32)
    W2 = np.random.randn(Dffn, D).astype(np.float32)
    gamma1 = np.random.randn(D).astype(np.float32)
    beta1 = np.random.randn(D).astype(np.float32)
    gamma2 = np.random.randn(D).astype(np.float32)
    beta2 = np.random.randn(D).astype(np.float32)
    allo_out = mod(X, Wq, Wk, Wv, Wp, W1, W2, gamma1, beta1, gamma2, beta2)

    def bert_layer(X, Wq, Wk, Wv, Wp, W1, W2, gamma1, beta1, gamma2, beta2):
        # 1. Bert Attention
        # 1.0 project Q, K, V
        Q = np.matmul(X, Wq)
        K = np.matmul(X, Wk)
        V = np.matmul(X, Wv)
        # 1.1 self attention
        attn = sdp(Q, K, V, H, D)
        # 1.2 output dense
        O_proj = np.matmul(attn, Wp)
        # 1.3 Residual layer
        res_attn = O_proj + X
        # 1.4 layer norm
        ln = np_layernorm(res_attn, gamma1, beta1)
        # 2. Feed Forward Network
        # 2.1 ffn dense 1
        ffn1 = np.matmul(ln, W1)
        # 2.2 gelu layer
        gelu_outp = 0.5 * ffn1 * (1 + np.tanh(0.797885 * (ffn1 + 0.044715 * ffn1**3)))
        # 2.3 ffn dense 2
        ffn2 = np.matmul(gelu_outp, W2)
        # 2.4 Residual layer
        res_ffn = ffn2 + ln
        # 2.5 layer norm
        ffn_ln_outp = np_layernorm(res_ffn, gamma2, beta2)
        return ffn_ln_outp

    np_out = bert_layer(X, Wq, Wk, Wv, Wp, W1, W2, gamma1, beta1, gamma2, beta2)
    np.testing.assert_allclose(allo_out, np_out, atol=1e-2, rtol=1e-2)
    print("Passed!")
    print(s.build(target="vhls"))


def np_rope(X, cos, sin, num_heads=8):
    X1 = X[:, :, :32]
    X2 = X[:, :, 32:]

    X_rotated = np.zeros_like(X)  # [1024, 8, 64]

    for i in range(num_heads):
        X_1_i = X1[:, i, :]
        X_2_i = X2[:, i, :]
        X_rotated_i = np.concatenate(
            (X_1_i * cos - X_2_i * sin, X_1_i * sin + X_2_i * cos), axis=-1
        )

        X_rotated[:, i, :] = X_rotated_i  # [1024, 8, 64]
    return X_rotated


def test_RoPE():
    from allo.library.nn import RoPE

    L, D = 1024, 512
    H = 8
    s = allo.customize(RoPE, instantiate=[float32, H, L, D])
    mod = s.build()
    Q = np.random.randn(L, D).astype(np.float32)
    cos = np.random.randn(L, 32).astype(np.float32)
    sin = np.random.randn(L, 32).astype(np.float32)
    allo_out = mod(Q, cos, sin)
    Q_np = Q.reshape(1024, 8, 64)
    np_out = np_rope(Q_np, cos, sin)
    np_out = np_out.reshape(1024, 512)
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")


def np_modulate_fused(x, shift, scale):
    output = x * (1 + scale) + shift
    return output


def test_modulate_fused():
    from allo.library.nn import modulate_fused
    from allo.library.nn import schedule_modulate_fused

    L, D = 1024, 512
    X = np.random.randn(L, D).astype(np.float32)
    X_norm = X
    s = allo.customize(modulate_fused, instantiate=[float32, L, D])
    schedule_modulate_fused(s)
    print(s.module)
    mod = s.build(target="llvm")
    scale = np.random.randn(D).astype(np.float32)
    shift = np.random.randn(D).astype(np.float32)
    allo_out = mod(X, scale, shift)
    np_out = np_modulate_fused(X_norm, shift=shift, scale=scale)
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")


def np_conv2d(inp, filter, stride=1, padding=0):
    N, C, H, W = inp.shape
    F, _, HH, WW = filter.shape
    H_out = (H + 2 * padding - HH) // stride + 1
    W_out = (W + 2 * padding - WW) // stride + 1
    out = np.zeros((N, F, H_out, W_out))
    inp_padded = np.pad(inp, ((0,), (0,), (padding,), (padding,)), mode="constant")
    for n, f, h, w in np.ndindex(N, F, H_out, W_out):
        out[n, f, h, w] = np.sum(
            inp_padded[
                n,
                :,
                h * stride : h * stride + HH,
                w * stride : w * stride + WW,
            ]
            * filter[f]
        )
    return out


def test_conv2d():
    from allo.library.nn import conv2d

    N, C, H, W = 1, 3, 16, 16
    K, F, S, P = 2, 2, 2, 1

    inp = np.random.randn(N, C, H, W).astype(np.float32)
    kernel = np.random.randn(F, C, K, K).astype(np.float32)
    bias = np.random.randn(F).astype(np.float32)

    Oh = (H + 2 * P - K) // S + 1
    Ow = (W + 2 * P - K) // S + 1

    s = allo.customize(
        conv2d, instantiate=[float32, N, C, F, H, W, K, K, Oh, Ow, S, S, P, P]
    )
    mod = s.build()
    allo_out = mod(inp, kernel, bias)
    np_out = np_conv2d(inp, kernel, S, P) + bias.reshape(1, F, 1, 1)
    np.testing.assert_allclose(allo_out, np_out, atol=1e-3)
    print("Passed!")
    print(s.build(target="vhls"))


def test_maxpool2d():
    from allo.library.nn import maxpool2d

    N, C, H, W = 1, 3, 16, 16
    K, S, P = 2, 2, 1

    inp = np.random.randn(N, C, H, W).astype(np.float32)

    Oh = (H + 2 * P - K) // S + 1
    Ow = (W + 2 * P - K) // S + 1

    s = allo.customize(maxpool2d, instantiate=[float32, N, C, H, W, K, Oh, Ow, S, P])
    mod = s.build()
    allo_out = mod(inp)

    inp_padded = np.pad(
        inp, ((0,), (0,), (P,), (P,)), mode="constant", constant_values=-np.inf
    )
    np_out = np.zeros((N, C, Oh, Ow), dtype=np.float32)
    for n, c, h, w in np.ndindex(N, C, Oh, Ow):
        h_start = h * S
        w_start = w * S
        window = inp_padded[n, c, h_start : h_start + K, w_start : w_start + K]
        np_out[n, c, h, w] = np.max(window)

    np.testing.assert_allclose(allo_out, np_out)
    print("Passed!")
    print(s.build(target="vhls"))


def test_avgpool2d():
    from allo.library.nn import avgpool2d

    N, C, H, W = 1, 3, 16, 16
    K, S, P = 2, 2, 1

    inp = np.random.randn(N, C, H, W).astype(np.float32)

    Oh = (H + 2 * P - K) // S + 1
    Ow = (W + 2 * P - K) // S + 1

    s = allo.customize(avgpool2d, instantiate=[float32, N, C, H, W, K, Oh, Ow, S, P])
    mod = s.build()
    allo_out = mod(inp)

    inp_padded = np.pad(inp, ((0,), (0,), (P,), (P,)), mode="constant")
    np_out = np.zeros((N, C, Oh, Ow), dtype=np.float32)
    for n, c, h, w in np.ndindex(N, C, Oh, Ow):
        h_start = h * S
        w_start = w * S
        window = inp_padded[n, c, h_start : h_start + K, w_start : w_start + K]
        np_out[n, c, h, w] = np.mean(window)

    np.testing.assert_allclose(allo_out, np_out)
    print("Passed!")
    print(s.build(target="vhls"))


def test_batchnorm2d():
    from allo.library.nn import batchnorm2d

    N, C, H, W = 1, 3, 16, 16
    inp = np.random.randn(N, C, H, W).astype(np.float32)
    gamma = np.random.randn(C).astype(np.float32)
    beta = np.random.randn(C).astype(np.float32)

    # Simulating running mean and variance, which are usually computed during training and different input means and var
    running_mean = np.random.randn(C).astype(np.float32)
    running_var = np.abs(np.random.randn(C)).astype(np.float32)

    s = allo.customize(batchnorm2d, instantiate=[float32, N, C, H, W])
    mod = s.build()
    allo_out = mod(inp, gamma, beta, 1e-5, running_mean, running_var)

    np_out = (inp - running_mean.reshape(1, C, 1, 1)) / np.sqrt(
        running_var.reshape(1, C, 1, 1) + 1e-5
    ) * gamma.reshape(1, C, 1, 1) + beta.reshape(1, C, 1, 1)

    np.testing.assert_allclose(allo_out, np_out, rtol=2e-04)
    print("Passed!")
    print(s.build(target="vhls"))


def np_log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    y = x - x_max
    return y - np.log(np.sum(np.exp(y), axis=axis, keepdims=True))


def test_log_softmax():
    from allo.library.nn import log_softmax

    # log_softmax[Ty, B, C]
    s = allo.customize(log_softmax, instantiate=[float32, 8, 8])
    mod = s.build()
    inp = np.random.randn(8, 8).astype(np.float32)
    inp = 1000 * inp
    allo_out = mod(inp)
    np_out = np_log_softmax(inp).astype(np.float32)
    np.testing.assert_allclose(allo_out, np_out, atol=1e-5, rtol=1e-5)
    print("Passed!")
    print(s.build(target="vhls"))


def test_batchnorm1d_2d():
    from allo.library.nn import batchnorm1d_2d

    B, C = 4, 8
    inp = np.random.randn(B, C).astype(np.float32)
    gamma = np.random.randn(C).astype(np.float32)
    beta = np.random.randn(C).astype(np.float32)

    # Simulating running mean and variance
    running_mean = np.random.randn(C).astype(np.float32)
    running_var = np.abs(np.random.randn(C)).astype(np.float32)

    s = allo.customize(batchnorm1d_2d, instantiate=[float32, B, C])
    mod = s.build()
    allo_out = mod(inp, gamma, beta, 1e-5, running_mean, running_var)

    np_out = (inp - running_mean.reshape(1, C)) / np.sqrt(
        running_var.reshape(1, C) + 1e-5
    ) * gamma.reshape(1, C) + beta.reshape(1, C)

    np.testing.assert_allclose(allo_out, np_out, rtol=1e-5, atol=1e-5)
    print("Passed!")
    print(s.build(target="vhls"))


def test_batchnorm1d_3d():
    from allo.library.nn import batchnorm1d_3d

    B, C, L = 2, 6, 5
    inp = np.random.randn(B, C, L).astype(np.float32)
    gamma = np.random.randn(C).astype(np.float32)
    beta = np.random.randn(C).astype(np.float32)

    running_mean = np.random.randn(C).astype(np.float32)
    running_var = np.abs(np.random.randn(C)).astype(np.float32)

    s = allo.customize(batchnorm1d_3d, instantiate=[float32, B, C, L])
    mod = s.build()
    allo_out = mod(inp, gamma, beta, 1e-5, running_mean, running_var)

    np_out = (inp - running_mean.reshape(1, C, 1)) / np.sqrt(
        running_var.reshape(1, C, 1) + 1e-5
    ) * gamma.reshape(1, C, 1) + beta.reshape(1, C, 1)

    np.testing.assert_allclose(allo_out, np_out, rtol=1e-5, atol=1e-5)
    print("Passed!")
    print(s.build(target="vhls"))


def test_repeat_batch3d():
    from allo.library.nn import repeat_batch3d

    B, L, C, N = 3, 4, 5, 2  # Output (N*B, L, C)
    inp = np.random.randn(B, L, C).astype(np.float32)

    s = allo.customize(repeat_batch3d, instantiate=[float32, B, L, C, N])
    mod = s.build()

    allo_out = mod(inp)
    # Copy inp N times along the batch dimension
    np_out = np.tile(inp, (N, 1, 1)).astype(np.float32)

    np.testing.assert_allclose(allo_out, np_out, rtol=1e-5, atol=1e-5)
    print("Passed!")
    print(s.build(target="vhls"))


def test_concat():
    from allo.library.nn import concat

    B, N1, N2, C = 2, 3, 4, 5
    x1 = np.random.randn(B, N1, C).astype(np.float32)
    x2 = np.random.randn(B, N2, C).astype(np.float32)

    s = allo.customize(concat, instantiate=[float32, B, N1, N2, C])
    mod = s.build()

    allo_out = mod(x1, x2)
    # Concatenate along dim 1 (N1 + N2, C)
    np_out = np.concatenate([x1, x2], axis=1).astype(np.float32)

    np.testing.assert_allclose(allo_out, np_out, rtol=1e-5, atol=1e-5)
    print("Passed!")
    print(s.build(target="vhls"))


def test_relu2d():
    from allo.library.nn import relu2d

    H, W = 8, 10
    x = (np.random.randn(H, W)).astype(np.float32)

    s = allo.customize(relu2d, instantiate=[float32, H, W])
    mod = s.build()

    allo_out = mod(x)
    np_out = np.maximum(x, 0.0).astype(np.float32)

    np.testing.assert_allclose(allo_out, np_out, rtol=1e-5, atol=1e-5)
    print("Passed!")
    print(s.build(target="vhls"))


def test_relu3d():
    from allo.library.nn import relu3d

    N, L, C = 3, 5, 7
    x = (np.random.randn(N, L, C)).astype(np.float32)

    s = allo.customize(relu3d, instantiate=[float32, N, L, C])
    mod = s.build()

    allo_out = mod(x)
    np_out = np.maximum(x, 0.0).astype(np.float32)

    np.testing.assert_allclose(allo_out, np_out, rtol=1e-5, atol=1e-5)
    print("Passed!")
    print(s.build(target="vhls"))


def test_relu4d():
    from allo.library.nn import relu4d

    N, C, H, W = 2, 4, 6, 8
    x = (np.random.randn(N, C, H, W)).astype(np.float32)

    s = allo.customize(relu4d, instantiate=[float32, N, C, H, W])
    mod = s.build()

    allo_out = mod(x)
    np_out = np.maximum(x, 0.0).astype(np.float32)

    np.testing.assert_allclose(allo_out, np_out, rtol=1e-5, atol=1e-5)
    print("Passed!")
    print(s.build(target="vhls"))


if __name__ == "__main__":
    pytest.main([__file__])
