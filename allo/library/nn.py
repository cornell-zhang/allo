# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unsupported-assignment-operation, chained-comparison

from .. import dsl
from .systolic import systolic

# BEGIN NATIVE INT8 QUANTIZATION: ADDED integer scalar types
# pylint: disable=consider-using-min-builtin,consider-using-max-builtin
from ..ir.types import float32, int16, int32, int64

# END NATIVE INT8 QUANTIZATION: ADDED integer scalar types


def linear2d[
    TyX, TyW, TyO, M, N, K
](X: "TyX[M, K]", W: "TyW[N, K]", b: "TyO[N]") -> "TyO[M, N]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    Z: TyO[M, N]
    buf: TyO[N]
    for i in range(M):
        for j_init in range(N):
            buf[j_init] = 0
        for k in range(K):
            # reorder reduction loop outside, and pipeline
            x: TyX = X[i, k]
            for j in range(N):
                buf[j] += x * W[j, k]
        for j_back in range(N):
            Z[i, j_back] = buf[j_back] + b[j_back]
    return Z


def schedule_linear2d(s):
    s.pipeline("linear2d:j")
    s.pipeline("linear2d:j_init")
    s.pipeline("linear2d:j_back")
    return s


def linear3d[
    TyX, TyW, TyO, B, L, D, M
](X: "TyX[B, L, D]", W: "TyW[M, D]", bias: "TyO[M]") -> "TyO[B, L, M]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    Z: TyO[B, L, M]
    buf: TyO[M]
    for b in range(B):
        for i in range(L):
            for j_init in range(M):
                buf[j_init] = 0
            for k in range(D):
                # reorder reduction loop outside, and pipeline
                x: TyX = X[b, i, k]
                for j in range(M):
                    buf[j] += x * W[j, k]
            for j_back in range(M):
                Z[b, i, j_back] = buf[j_back] + bias[j_back]
    return Z


def schedule_linear3d(s):
    s.pipeline("linear3d:j")
    s.pipeline("linear3d:j_init")
    s.pipeline("linear3d:j_back")
    return s


def relu2d[Ty, H, W](X: "Ty[H, W]") -> "Ty[H, W]":
    Z: Ty[H, W]
    for h, w in dsl.grid(H, W):
        Z[h, w] = max(0.0, X[h, w])
    return Z


def schedule_relu2d(s):
    s.pipeline("relu2d:w")
    return s


def relu4d[Ty, N, C, H, W](X: "Ty[N, C, H, W]") -> "Ty[N, C, H, W]":
    Z: Ty[N, C, H, W]
    for n, c, h, w in dsl.grid(N, C, H, W):
        Z[n, c, h, w] = max(0.0, X[n, c, h, w])
    return Z


def schedule_relu4d(s):
    s.pipeline("relu4d:w")
    return s


def relu3d[Ty, N, L, C](X: "Ty[N, L, C]") -> "Ty[N, L, C]":
    Z: Ty[N, L, C]
    for n, l, c in dsl.grid(N, L, C):
        Z[n, l, c] = max(0.0, X[n, l, c])
    return Z


def schedule_relu3d(s):
    s.pipeline("relu3d:c")
    return s


def softmax[Ty, L](X: "Ty[L, L]") -> "Ty[L, L]":
    Z: Ty[L, L]
    E: Ty[L, L]
    M: Ty[L] = -1000000000000.0
    S: Ty[L] = 0.0

    for i, j in dsl.grid(L, L, name="row_max"):
        if X[i, j] > M[i]:
            M[i] = X[i, j]

    # compute exp and sum
    for i, j in dsl.grid(L, L, name="exp_sum"):
        E[i, j] = dsl.exp(X[i, j] - M[i])
        S[i] += E[i, j]

    for i, j in dsl.grid(L, L, name="update"):
        Z[i, j] = E[i, j] / S[i]

    return Z


def schedule_softmax(s):
    lj = s.get_loops(s.top_func_name)["exp_sum"]["j"]
    s.pipeline(lj)
    lj = s.get_loops(s.top_func_name)["update"]["j"]
    s.pipeline(lj)
    return s


def log_softmax[Ty, B, C](X: "Ty[B, C]") -> "Ty[B, C]":
    Z: Ty[B, C]
    E: Ty[B, C]
    M: Ty[B] = -1000000000000.0
    S: Ty[B] = 0.0

    # Row-wise max
    for i, j in dsl.grid(B, C, name="row_max"):
        if X[i, j] > M[i]:
            M[i] = X[i, j]

    # Compute exp and sum
    for i, j in dsl.grid(B, C, name="exp_sum"):
        E[i, j] = dsl.exp(X[i, j] - M[i])
        S[i] += E[i, j]

    # Log softmax update
    for i, j in dsl.grid(B, C, name="update"):
        Z[i, j] = X[i, j] - M[i] - dsl.log(S[i])

    return Z


def schedule_log_softmax(s):
    lj = s.get_loops(s.top_func_name)["exp_sum"]["j"]
    s.pipeline(lj)
    lj = s.get_loops(s.top_func_name)["update"]["j"]
    s.pipeline(lj)
    return s


def layer_norm[Ty, L, D](X: "Ty[L, D]", gamma: "Ty[D]", beta: "Ty[D]") -> "Ty[L, D]":
    Z: Ty[L, D]
    mean: Ty[L] = 0.0
    mean2: Ty[L] = 0.0
    var: Ty[L]

    for i, j in dsl.grid(L, D, name="sum"):
        mean[i] += X[i, j]
        mean2[i] += X[i, j] * X[i, j]

    for i in dsl.grid(L, name="mean_var"):
        mean[i] = mean[i] / float(D)
        mean2[i] = mean2[i] / float(D)
        var[i] = mean2[i] - mean[i] * mean[i]

    for i, j in dsl.grid(L, D, name="norm"):
        Z[i, j] = gamma[j] * (X[i, j] - mean[i]) / dsl.sqrt(var[i] + 0.00001) + beta[j]

    return Z


def schedule_layernorm(s):
    lj = s.get_loops(s.top_func_name)["sum"]["j"]
    s.pipeline(lj)
    li = s.get_loops(s.top_func_name)["mean_var"]["i"]
    s.pipeline(li)
    lj = s.get_loops(s.top_func_name)["norm"]["j"]
    s.pipeline(lj)
    return s


def GeLU[Ty, L, D](X: "Ty[L, D]") -> "Ty[L, D]":
    Z: Ty[L, D]
    for i, j in dsl.grid(L, D, name="gelu"):
        Z[i, j] = (
            0.5
            * X[i, j]
            * (
                1.0
                + dsl.tanh(0.797885 * (X[i, j] + 0.044715 * dsl.power(X[i, j], 3.0)))
            )
        )
    return Z


def schedule_gelu(s):
    lj = s.get_loops(s.top_func_name)["gelu"]["j"]
    s.pipeline(lj)
    return s


def residual_add[Ty, L, D](X1: "Ty[L, D]", X2: "Ty[L, D]") -> "Ty[L, D]":
    Z: Ty[L, D]
    for i, j in dsl.grid(L, D):
        Z[i, j] = X1[i, j] + X2[i, j]
    return Z


def scaled_dot_product_attention[
    Ty, H, L, D, M0, M1
](Q: "Ty[L, D]", K: "Ty[L, D]", V: "Ty[L, D]") -> "Ty[L, D]":
    # softmax(QK^T/sqrt(D // H))
    Z: Ty[L, D]

    for h in range(H):
        Q_h: Ty[L, D // H]
        K_h: Ty[D // H, L]
        V_h: Ty[L, D // H]

        # split Q, K, V
        for i, j in dsl.grid(L, D // H, name="mha_split"):
            Q_h[i, j] = Q[i, h * (D // H) + j]
            # transposed
            K_h[j, i] = K[i, h * (D // H) + j]
            V_h[i, j] = V[i, h * (D // H) + j]

        # QK^T = (L, D//H) x (D//H, L) = (L, L)
        C_h: Ty[L, D // H] = 0
        Y: Ty[L, L] = 0
        systolic[Ty, Ty, Ty, L, D // H, L, M0, M1, "QKT"](Q_h, K_h, Y)
        # Need to return a new value
        S = softmax[Ty, L](Y)
        # YV = (L, L) x (L, D//H) = (L, D//H)
        systolic[Ty, Ty, Ty, L, L, D // H, M0, M1, "YV"](S, V_h, C_h)

        for i, j in dsl.grid(L, D // H, name="mha_merge"):
            Z[i, h * (D // H) + j] = C_h[i, j]

    return Z


def RoPE[
    Ty, H, L, D
](X: "Ty[L, D]", cos: "Ty[L, D // H // 2]", sin: "Ty[L, D // H // 2]") -> "Ty[L, D]":
    # Rotary Position Embedding
    # Reference: https://arxiv.org/abs/2104.09864
    X_rotary: Ty[L, D]
    for h in range(H):
        X_1_h: Ty[L, D // H // 2]
        X_2_h: Ty[L, D // H // 2]
        for i, j in dsl.grid(L, D // H // 2, name="rope_split_1"):
            X_1_h[i, j] = X[i, h * (D // H) + j]
        for i, j in dsl.grid(L, D // H // 2, name="rope_split_2"):
            X_2_h[i, j] = X[i, h * (D // H) + D // H // 2 + j]
        X_1_rotary: Ty[L, D // H // 2] = 0
        X_2_rotary: Ty[L, D // H // 2] = 0
        for i, j in dsl.grid(L, D // H // 2, name="rotary_1"):
            X_1_rotary[i, j] = cos[i, j] * X_1_h[i, j] - sin[i, j] * X_2_h[i, j]
        for i, j in dsl.grid(L, D // H // 2, name="rotary_2"):
            X_2_rotary[i, j] = sin[i, j] * X_1_h[i, j] + cos[i, j] * X_2_h[i, j]
        for i, j in dsl.grid(L, D // H // 2, name="rotary_merge_1"):
            X_rotary[i, h * (D // H) + j] = X_1_rotary[i, j]
        for i, j in dsl.grid(L, D // H // 2, name="rotary_merge_2"):
            X_rotary[i, h * (D // H) + D // H // 2 + j] = X_2_rotary[i, j]
    return X_rotary


def modulate_fused[
    Ty, L, D
](X: "Ty[L,D]", scale: "Ty[D]", shift: "Ty[D]") -> "Ty[L, D]":
    Z: Ty[L, D]
    for i, j in dsl.grid(L, D, name="m_fused"):
        Z[i, j] = X[i, j] * (1 + scale[j]) + shift[j]
    return Z


def schedule_modulate_fused(s):
    lj = s.get_loops(s.top_func_name)["m_fused"]["j"]
    s.pipeline(lj)


def conv2d[
    Ty, B, Cin, Cout, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1
](
    inp: "Ty[B, Cin, H, W]", kernel: "Ty[Cout, Cin, Kh, Kw]", bias: "Ty[Cout]"
) -> "Ty[B, Cout, Oh, Ow]":
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    Z: Ty[B, Cout, Oh, Ow]

    # Current implementation is does not support dilation other than 1
    for batch, cout, oh, ow in dsl.grid(B, Cout, Oh, Ow):
        temp: Ty = bias[cout]

        for cin, kh, kw in dsl.grid(Cin, Kh, Kw):
            h_pos: Ty = oh * Sh + kh - Pd0
            w_pos: Ty = ow * Sw + kw - Pd1
            if h_pos >= 0 and h_pos < H and w_pos >= 0 and w_pos < W:
                temp += inp[batch, cin, h_pos, w_pos] * kernel[cout, cin, kh, kw]

        Z[batch, cout, oh, ow] = temp
    return Z


def schedule_conv2d(s):
    s.pipeline("conv2d:cout")
    s.pipeline("conv2d:ow")
    return s


def maxpool2d[
    Ty, B, C, H, W, K, Oh, Ow, S, Pd
](inp: "Ty[B, C, H, W]",) -> "Ty[B, C, Oh, Ow]":
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    Z: Ty[B, C, Oh, Ow]
    for batch, c, oh, ow in dsl.grid(B, C, Oh, Ow):
        max_val: Ty = -1000000000000.0
        for kh, kw in dsl.grid(K, K):
            h_pos: Ty = oh * S + kh - Pd
            w_pos: Ty = ow * S + kw - Pd
            if h_pos >= 0 and h_pos < H and w_pos >= 0 and w_pos < W:
                new_max: Ty = max(max_val, inp[batch, c, h_pos, w_pos])
                max_val = new_max
        Z[batch, c, oh, ow] = max_val
    return Z


def schedule_maxpool2d(s):
    s.pipeline("maxpool2d:c")
    s.pipeline("maxpool2d:ow")
    return s


def avgpool2d[
    Ty, B, C, H, W, K, Oh, Ow, S, Pd
](inp: "Ty[B, C, H, W]",) -> "Ty[B, C, Oh, Ow]":
    # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    Z: Ty[B, C, Oh, Ow]
    for batch, c, oh, ow in dsl.grid(B, C, Oh, Ow):
        temp: Ty = 0.0
        for kh, kw in dsl.grid(K, K):
            h_pos: Ty = oh * S + kh - Pd
            w_pos: Ty = ow * S + kw - Pd
            if h_pos >= 0 and h_pos < H and w_pos >= 0 and w_pos < W:
                temp += inp[batch, c, h_pos, w_pos]
        Z[batch, c, oh, ow] = temp / (K * K)
    return Z


def schedule_avgpool2d(s):
    s.pipeline("avgpool2d:c")
    s.pipeline("avgpool2d:ow")
    return s


def batchnorm2d[
    Ty, B, C, H, W
](
    X: "Ty[B, C, H, W]",
    gamma: "Ty[C]",
    beta: "Ty[C]",
    eps: "Ty",
    mean: "Ty[C]",
    var: "Ty[C]",
) -> "Ty[B, C, H, W]":
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    Z: Ty[B, C, H, W]
    for b, c, h, w in dsl.grid(B, C, H, W):
        Z[b, c, h, w] = (
            gamma[c] * (X[b, c, h, w] - mean[c]) / dsl.sqrt(var[c] + eps) + beta[c]
        )

    return Z


def schedule_batchnorm2d(s):
    s.pipeline("batchnorm2d:w")
    return s


def batchnorm1d_2d[
    Ty, B, C
](
    X: "Ty[B, C]", gamma: "Ty[C]", beta: "Ty[C]", eps: "Ty", mean: "Ty[C]", var: "Ty[C]"
) -> "Ty[B, C]":
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    Z: Ty[B, C]
    for b, c in dsl.grid(B, C):
        Z[b, c] = gamma[c] * (X[b, c] - mean[c]) / dsl.sqrt(var[c] + eps) + beta[c]
    return Z


def schedule_batchnorm1d_2d(s):
    s.pipeline("batchnorm1d_2d:c")
    return s


def batchnorm1d_3d[
    Ty, B, C, L
](
    X: "Ty[B, C, L]",
    gamma: "Ty[C]",
    beta: "Ty[C]",
    eps: "Ty",
    mean: "Ty[C]",
    var: "Ty[C]",
) -> "Ty[B, C, L]":
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    Z: Ty[B, C, L]
    for b, c, l in dsl.grid(B, C, L):
        Z[b, c, l] = (
            gamma[c] * (X[b, c, l] - mean[c]) / dsl.sqrt(var[c] + eps) + beta[c]
        )
    return Z


def schedule_batchnorm1d_3d(s):
    s.pipeline("batchnorm1d_3d:l")
    return s


def repeat_batch3d[Ty, B, L, C, N](X: "Ty[B, L, C]") -> "Ty[N*B, L, C]":
    """
    Repeat X along batch dimension N times for cls_token.
    """
    Y: Ty[N * B, L, C]
    for r, b, l, c in dsl.grid(N, B, L, C):
        Y[r * B + b, l, c] = X[b, l, c]
    return Y


def schedule_repeat_batch3d(s):
    s.pipeline("repeat_batch3d:c")
    return s


def concat[
    Ty, B, N1, N2, C
](X1: "Ty[B, N1, C]", X2: "Ty[B, N2, C]") -> "Ty[B, N1+N2, C]":
    Y: Ty[B, N1 + N2, C]
    for b, n, c in dsl.grid(B, N1, C):
        Y[b, n, c] = X1[b, n, c]
    for b2, n2, c2 in dsl.grid(B, N2, C):
        Y[b2, n2 + N1, c2] = X2[b2, n2, c2]
    return Y


def schedule_concat(s):
    s.pipeline("concat:c")
    return s


# BEGIN NATIVE INT8 QUANTIZATION: ADDED rank-2/rank-3 integer kernels

# These kernels are separate from the existing floating-point kernels above.
# The widened TyAcc parameter is instantiated as int64 by TorchBuilder.


def roundeven(value: float32) -> int32:
    """Round to nearest integer, resolving exact ties toward the even value."""

    truncated: int32 = int(value)
    fraction: float32 = value - float(truncated)
    if fraction < 0.0:
        fraction = -fraction
    result: int32 = truncated
    if fraction > 0.5 or (fraction == 0.5 and (truncated & 1) != 0):
        if value < 0.0:
            result -= 1
        else:
            result += 1
    return result


def quantize2d[
    TyIn, TyOut, H, W
](
    X: "TyIn[H, W]",
    scale: float32,
    zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[H, W]":
    Z: TyOut[H, W]
    for h, w in dsl.grid(H, W):
        value: int32 = roundeven(X[h, w] / scale) + zero_point
        value = max(qmin, min(qmax, value))
        Z[h, w] = value
    return Z


def quantize3d[
    TyIn, TyOut, B, L, D
](
    X: "TyIn[B, L, D]",
    scale: float32,
    zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[B, L, D]":
    Z: TyOut[B, L, D]
    for b, l, d in dsl.grid(B, L, D):
        value: int32 = roundeven(X[b, l, d] / scale) + zero_point
        value = max(qmin, min(qmax, value))
        Z[b, l, d] = value
    return Z


def dequantize2d[
    TyIn, TyOut, H, W
](X: "TyIn[H, W]", scale: float32, zero_point: int32,) -> "TyOut[H, W]":
    Z: TyOut[H, W]
    for h, w in dsl.grid(H, W):
        centered: int32 = X[h, w] - zero_point
        Z[h, w] = centered * scale
    return Z


def dequantize3d[
    TyIn, TyOut, B, L, D
](X: "TyIn[B, L, D]", scale: float32, zero_point: int32,) -> "TyOut[B, L, D]":
    Z: TyOut[B, L, D]
    for b, l, d in dsl.grid(B, L, D):
        centered: int32 = X[b, l, d] - zero_point
        Z[b, l, d] = centered * scale
    return Z


def requantize2d[
    TyIn, TyAcc, TyOut, H, W
](
    X: "TyIn[H, W]",
    multiplier: "TyAcc",
    shift: int32,
    input_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[H, W]":
    Z: TyOut[H, W]
    for h, w in dsl.grid(H, W):
        centered: TyAcc = X[h, w] - input_zero_point
        scaled: TyAcc = centered * multiplier
        rounded: TyAcc = scaled
        if shift > 0:
            magnitude: TyAcc = scaled
            if magnitude < 0:
                magnitude = -magnitude
            quotient: TyAcc = magnitude >> shift
            remainder: TyAcc = magnitude - (quotient << shift)
            one: TyAcc = 1
            halfway: TyAcc = one << (shift - 1)
            increment: TyAcc = 0
            if remainder > halfway:
                increment = 1
            if remainder == halfway and (quotient & 1) == 1:
                increment = 1
            rounded = quotient + increment
            if scaled < 0:
                rounded = -rounded
        if shift < 0:
            rounded = scaled << (0 - shift)
        quantized: TyAcc = rounded + output_zero_point
        if quantized < qmin:
            quantized = qmin
        if quantized > qmax:
            quantized = qmax
        Z[h, w] = quantized
    return Z


def requantize3d[
    TyIn, TyAcc, TyOut, B, L, D
](
    X: "TyIn[B, L, D]",
    multiplier: "TyAcc",
    shift: int32,
    input_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[B, L, D]":
    Z: TyOut[B, L, D]
    for b, l, d in dsl.grid(B, L, D):
        centered: TyAcc = X[b, l, d] - input_zero_point
        scaled: TyAcc = centered * multiplier
        rounded: TyAcc = scaled
        if shift > 0:
            magnitude: TyAcc = scaled
            if magnitude < 0:
                magnitude = -magnitude
            quotient: TyAcc = magnitude >> shift
            remainder: TyAcc = magnitude - (quotient << shift)
            one: TyAcc = 1
            halfway: TyAcc = one << (shift - 1)
            increment: TyAcc = 0
            if remainder > halfway:
                increment = 1
            if remainder == halfway and (quotient & 1) == 1:
                increment = 1
            rounded = quotient + increment
            if scaled < 0:
                rounded = -rounded
        if shift < 0:
            rounded = scaled << (0 - shift)
        quantized: TyAcc = rounded + output_zero_point
        if quantized < qmin:
            quantized = qmin
        if quantized > qmax:
            quantized = qmax
        Z[b, l, d] = quantized
    return Z

# BEGIN NATIVE INT8 QUANTIZATION: ADDED per-output-channel requantization


def requantize_per_channel2d[
    TyIn, TyAcc, TyOut, H, W
](
    X: "TyIn[H, W]",
    multipliers: "TyAcc[W]",
    shifts: "int32[W]",
    input_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[H, W]":
    Z: TyOut[H, W]

    for h, w in dsl.grid(H, W):
        channel_shift: int32 = shifts[w]
        centered: TyAcc = X[h, w] - input_zero_point
        scaled: TyAcc = centered * multipliers[w]
        rounded: TyAcc = scaled

        if channel_shift > 0:
            magnitude: TyAcc = scaled
            if magnitude < 0:
                magnitude = -magnitude

            quotient: TyAcc = magnitude >> channel_shift
            remainder: TyAcc = (
                magnitude - (quotient << channel_shift)
            )
            one: TyAcc = 1
            halfway: TyAcc = one << (channel_shift - 1)
            increment: TyAcc = 0

            if remainder > halfway:
                increment = 1
            if (
                remainder == halfway
                and (quotient & 1) == 1
            ):
                increment = 1

            rounded = quotient + increment
            if scaled < 0:
                rounded = -rounded

        if channel_shift < 0:
            rounded = scaled << (0 - channel_shift)

        quantized: TyAcc = rounded + output_zero_point

        if quantized < qmin:
            quantized = qmin
        if quantized > qmax:
            quantized = qmax

        Z[h, w] = quantized

    return Z


def requantize_per_channel3d[
    TyIn, TyAcc, TyOut, B, L, D
](
    X: "TyIn[B, L, D]",
    multipliers: "TyAcc[D]",
    shifts: "int32[D]",
    input_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[B, L, D]":
    Z: TyOut[B, L, D]

    for b, l, d in dsl.grid(B, L, D):
        channel_shift: int32 = shifts[d]
        centered: TyAcc = X[b, l, d] - input_zero_point
        scaled: TyAcc = centered * multipliers[d]
        rounded: TyAcc = scaled

        if channel_shift > 0:
            magnitude: TyAcc = scaled
            if magnitude < 0:
                magnitude = -magnitude

            quotient: TyAcc = magnitude >> channel_shift
            remainder: TyAcc = (
                magnitude - (quotient << channel_shift)
            )
            one: TyAcc = 1
            halfway: TyAcc = one << (channel_shift - 1)
            increment: TyAcc = 0

            if remainder > halfway:
                increment = 1
            if (
                remainder == halfway
                and (quotient & 1) == 1
            ):
                increment = 1

            rounded = quotient + increment
            if scaled < 0:
                rounded = -rounded

        if channel_shift < 0:
            rounded = scaled << (0 - channel_shift)

        quantized: TyAcc = rounded + output_zero_point

        if quantized < qmin:
            quantized = qmin
        if quantized > qmax:
            quantized = qmax

        Z[b, l, d] = quantized

    return Z


# END NATIVE INT8 QUANTIZATION: ADDED per-output-channel requantization

def qadd2d[
    TyL, TyR, TyAcc, TyOut, H, W
](
    lhs: "TyL[H, W]",
    rhs: "TyR[H, W]",
    lhs_multiplier: "TyAcc",
    rhs_multiplier: "TyAcc",
    shift: int32,
    lhs_zero_point: int32,
    rhs_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[H, W]":
    Z: TyOut[H, W]
    for h, w in dsl.grid(H, W):
        lhs_scaled: TyAcc = (lhs[h, w] - lhs_zero_point) * lhs_multiplier
        rhs_scaled: TyAcc = (rhs[h, w] - rhs_zero_point) * rhs_multiplier
        total: TyAcc = lhs_scaled + rhs_scaled
        rounded: TyAcc = total
        if shift > 0:
            magnitude: TyAcc = total
            if magnitude < 0:
                magnitude = -magnitude
            quotient: TyAcc = magnitude >> shift
            remainder: TyAcc = magnitude - (quotient << shift)
            one: TyAcc = 1
            halfway: TyAcc = one << (shift - 1)
            increment: TyAcc = 0
            if remainder > halfway:
                increment = 1
            if remainder == halfway and (quotient & 1) == 1:
                increment = 1
            rounded = quotient + increment
            if total < 0:
                rounded = -rounded
        if shift < 0:
            rounded = total << (0 - shift)
        quantized: TyAcc = rounded + output_zero_point
        if quantized < qmin:
            quantized = qmin
        if quantized > qmax:
            quantized = qmax
        Z[h, w] = quantized
    return Z


def qadd3d[
    TyL, TyR, TyAcc, TyOut, B, L, D
](
    lhs: "TyL[B, L, D]",
    rhs: "TyR[B, L, D]",
    lhs_multiplier: "TyAcc",
    rhs_multiplier: "TyAcc",
    shift: int32,
    lhs_zero_point: int32,
    rhs_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[B, L, D]":
    Z: TyOut[B, L, D]
    for b, l, d in dsl.grid(B, L, D):
        lhs_scaled: TyAcc = (lhs[b, l, d] - lhs_zero_point) * lhs_multiplier
        rhs_scaled: TyAcc = (rhs[b, l, d] - rhs_zero_point) * rhs_multiplier
        total: TyAcc = lhs_scaled + rhs_scaled
        rounded: TyAcc = total
        if shift > 0:
            magnitude: TyAcc = total
            if magnitude < 0:
                magnitude = -magnitude
            quotient: TyAcc = magnitude >> shift
            remainder: TyAcc = magnitude - (quotient << shift)
            one: TyAcc = 1
            halfway: TyAcc = one << (shift - 1)
            increment: TyAcc = 0
            if remainder > halfway:
                increment = 1
            if remainder == halfway and (quotient & 1) == 1:
                increment = 1
            rounded = quotient + increment
            if total < 0:
                rounded = -rounded
        if shift < 0:
            rounded = total << (0 - shift)
        quantized: TyAcc = rounded + output_zero_point
        if quantized < qmin:
            quantized = qmin
        if quantized > qmax:
            quantized = qmax
        Z[b, l, d] = quantized
    return Z


# BEGIN NATIVE INT8 QUANTIZATION: ADDED transformer kernels


def rms_norm3d[
    Ty, B, L, D
](X: "Ty[B, L, D]", weight: "Ty[D]", eps: float32) -> "Ty[B, L, D]":
    """RMSNorm over the final dimension."""

    Z: Ty[B, L, D]
    sum_sq: Ty[B, L] = 0.0
    for b, l, d in dsl.grid(B, L, D, name="rms_sum_sq"):
        sum_sq[b, l] += X[b, l, d] * X[b, l, d]
    for b, l, d in dsl.grid(B, L, D, name="rms_normalize"):
        mean_sq: Ty = sum_sq[b, l] / float(D)
        Z[b, l, d] = X[b, l, d] * weight[d] / dsl.sqrt(mean_sq + eps)
    return Z


def silu3d[Ty, B, L, D](X: "Ty[B, L, D]") -> "Ty[B, L, D]":
    """SiLU activation used by Llama-family gated MLPs."""

    Z: Ty[B, L, D]
    for b, l, d in dsl.grid(B, L, D, name="silu"):
        Z[b, l, d] = X[b, l, d] / (1.0 + dsl.exp(0.0 - X[b, l, d]))
    return Z


def rope3d[
    Ty, H, L, D
](X: "Ty[H, L, D]", cos: "Ty[L, D]", sin: "Ty[L, D]") -> "Ty[H, L, D]":
    """Llama-style rotary embedding with the head axis folded into batch."""

    Z: Ty[H, L, D]
    for h, l, d in dsl.grid(H, L, D // 2, name="rope_first_half"):
        Z[h, l, d] = X[h, l, d] * cos[l, d] - X[h, l, d + D // 2] * sin[l, d]
    for h, l, d in dsl.grid(H, L, D // 2, name="rope_second_half"):
        Z[h, l, d + D // 2] = (
            X[h, l, d + D // 2] * cos[l, d + D // 2]
            + X[h, l, d] * sin[l, d + D // 2]
        )
    return Z


def positioned_rope3d[
    Ty, H, L, S, D
](
    X: "Ty[H, L, D]",
    cos: "Ty[S, D]",
    sin: "Ty[S, D]",
    position: int32,
) -> "Ty[H, L, D]":
    """RoPE using a runtime start position and a fixed maximum table."""

    Z: Ty[H, L, D]
    for h, l, d in dsl.grid(H, L, D // 2, name="positioned_rope_first"):
        Z[h, l, d] = (
            X[h, l, d] * cos[position + l, d]
            - X[h, l, d + D // 2] * sin[position + l, d]
        )
    for h, l, d in dsl.grid(H, L, D // 2, name="positioned_rope_second"):
        Z[h, l, d + D // 2] = (
            X[h, l, d + D // 2] * cos[position + l, d + D // 2]
            + X[h, l, d] * sin[position + l, d + D // 2]
        )
    return Z


def repeat_interleave3d[
    Ty, H, L, D, R
](X: "Ty[H, L, D]") -> "Ty[H * R, L, D]":
    """Repeat each KV head consecutively for grouped-query attention."""

    Z: Ty[H * R, L, D]
    for h, r, l, d in dsl.grid(H, R, L, D, name="repeat_interleave"):
        Z[h * R + r, l, d] = X[h, l, d]
    return Z


def causal_softmax3d[
    Ty, H, L, S
](X: "Ty[H, L, S]", causal_offset: int32) -> "Ty[H, L, S]":
    """Last-dimension softmax with a causal prefix offset."""

    Z: Ty[H, L, S]
    E: Ty[H, L, S] = 0.0
    row_max: Ty[H, L] = -1000000000000.0
    row_sum: Ty[H, L] = 0.0
    for h, l, s in dsl.grid(H, L, S, name="causal_row_max"):
        if s < causal_offset + l + 1 and X[h, l, s] > row_max[h, l]:
            row_max[h, l] = X[h, l, s]
    for h, l, s in dsl.grid(H, L, S, name="causal_exp_sum"):
        if s < causal_offset + l + 1:
            E[h, l, s] = dsl.exp(X[h, l, s] - row_max[h, l])
            row_sum[h, l] += E[h, l, s]
    for h, l, s in dsl.grid(H, L, S, name="causal_normalize"):
        if s < causal_offset + l + 1:
            Z[h, l, s] = E[h, l, s] / row_sum[h, l]
        else:
            Z[h, l, s] = 0.0
    return Z


def embedding2d[
    TyW, B, L, V, D
](input_ids: "int32[B, L]", weight: "TyW[V, D]") -> "TyW[B, L, D]":
    """Token embedding lookup for fixed-rank token inputs."""

    Z: TyW[B, L, D]
    for b, l, d in dsl.grid(B, L, D, name="embedding_lookup"):
        Z[b, l, d] = weight[input_ids[b, l], d]
    return Z


def kv_cache_update3d[
    Ty, H, L, S, D
](values: "Ty[H, L, D]", cache: "Ty[H, S, D]", position: int32) -> "Ty[H, S, D]":
    """Copy a fixed token block into a KV cache and return the updated cache."""

    Z: Ty[H, S, D]
    for h, s, d in dsl.grid(H, S, D, name="kv_cache_copy"):
        Z[h, s, d] = cache[h, s, d]
    for h, l, d in dsl.grid(H, L, D, name="kv_cache_update"):
        if position + l < S:
            Z[h, position + l, d] = values[h, l, d]
    return Z


def qmul3d[
    TyL, TyR, TyAcc, TyOut, B, L, D
](
    lhs: "TyL[B, L, D]",
    rhs: "TyR[B, L, D]",
    multiplier: "TyAcc",
    shift: int32,
    lhs_zero_point: int32,
    rhs_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[B, L, D]":
    """Elementwise integer multiply followed by widened requantization."""

    Z: TyOut[B, L, D]
    for b, l, d in dsl.grid(B, L, D, name="qmul"):
        product: TyAcc = (lhs[b, l, d] - lhs_zero_point) * (
            rhs[b, l, d] - rhs_zero_point
        )
        scaled: TyAcc = product * multiplier
        rounded: TyAcc = scaled
        if shift > 0:
            magnitude: TyAcc = scaled
            if magnitude < 0:
                magnitude = -magnitude
            quotient: TyAcc = magnitude >> shift
            remainder: TyAcc = magnitude - (quotient << shift)
            one: TyAcc = 1
            halfway: TyAcc = one << (shift - 1)
            increment: TyAcc = 0
            if remainder > halfway:
                increment = 1
            if remainder == halfway and (quotient & 1) == 1:
                increment = 1
            rounded = quotient + increment
            if scaled < 0:
                rounded = -rounded
        if shift < 0:
            rounded = scaled << (0 - shift)
        quantized: TyAcc = rounded + output_zero_point
        if quantized < qmin:
            quantized = qmin
        if quantized > qmax:
            quantized = qmax
        Z[b, l, d] = quantized
    return Z


def qmatmul3d[
    TyL, TyR, TyAcc, TyWide, TyOut, B, M, K, N
](
    lhs: "TyL[B, M, K]",
    rhs: "TyR[B, K, N]",
    multiplier: "TyWide",
    shift: int32,
    lhs_zero_point: int32,
    rhs_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[B, M, N]":
    """Batched integer matmul with int32 accumulation and int64 requantization."""

    Z: TyOut[B, M, N]
    for b, m, n in dsl.grid(B, M, N, name="qmatmul_output"):
        acc: TyAcc = 0
        for k in range(K):
            acc += (lhs[b, m, k] - lhs_zero_point) * (
                rhs[b, k, n] - rhs_zero_point
            )
        scaled: TyWide = acc * multiplier
        rounded: TyWide = scaled
        if shift > 0:
            magnitude: TyWide = scaled
            if magnitude < 0:
                magnitude = -magnitude
            quotient: TyWide = magnitude >> shift
            remainder: TyWide = magnitude - (quotient << shift)
            one: TyWide = 1
            halfway: TyWide = one << (shift - 1)
            increment: TyWide = 0
            if remainder > halfway:
                increment = 1
            if remainder == halfway and (quotient & 1) == 1:
                increment = 1
            rounded = quotient + increment
            if scaled < 0:
                rounded = -rounded
        if shift < 0:
            rounded = scaled << (0 - shift)
        quantized: TyWide = rounded + output_zero_point
        if quantized < qmin:
            quantized = qmin
        if quantized > qmax:
            quantized = qmax
        Z[b, m, n] = quantized
    return Z


def qsilu3d[
    TyIn, TyOut, B, L, D
](X: "TyIn[B, L, D]", table: "TyOut[256]", qmin: int32) -> "TyOut[B, L, D]":
    """Integer SiLU using a compile-time calibrated 256-entry LUT."""

    Z: TyOut[B, L, D]
    for b, l, d in dsl.grid(B, L, D, name="qsilu"):
        index: int32 = X[b, l, d] - qmin
        Z[b, l, d] = table[index]
    return Z


def qrope3d[
    TyIn, TyCoeff, TyAcc, TyOut, H, L, D
](
    X: "TyIn[H, L, D]",
    cos: "TyCoeff[L, D]",
    sin: "TyCoeff[L, D]",
    multiplier: "TyAcc",
    shift: int32,
    input_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[H, L, D]":
    """Integer rotary embedding with Q15 sine/cosine coefficients."""

    Z: TyOut[H, L, D]
    for h, l, d in dsl.grid(H, L, D, name="qrope"):
        partner: int32 = d + D // 2
        sign: int32 = -1
        if d >= D // 2:
            partner = d - D // 2
            sign = 1
        value: TyAcc = (X[h, l, d] - input_zero_point) * cos[l, d]
        value += sign * (X[h, l, partner] - input_zero_point) * sin[l, d]
        scaled: TyAcc = value * multiplier
        rounded: TyAcc = scaled
        if shift > 0:
            magnitude: TyAcc = scaled
            if magnitude < 0:
                magnitude = -magnitude
            quotient: TyAcc = magnitude >> shift
            remainder: TyAcc = magnitude - (quotient << shift)
            one: TyAcc = 1
            halfway: TyAcc = one << (shift - 1)
            increment: TyAcc = 0
            if remainder > halfway:
                increment = 1
            if remainder == halfway and (quotient & 1) == 1:
                increment = 1
            rounded = quotient + increment
            if scaled < 0:
                rounded = -rounded
        if shift < 0:
            rounded = scaled << (0 - shift)
        quantized: TyAcc = rounded + output_zero_point
        if quantized < qmin:
            quantized = qmin
        if quantized > qmax:
            quantized = qmax
        Z[h, l, d] = quantized
    return Z


def qpositioned_rope3d[
    TyIn, TyCoeff, TyAcc, TyOut, H, L, S, D
](
    X: "TyIn[H, L, D]",
    cos: "TyCoeff[S, D]",
    sin: "TyCoeff[S, D]",
    multiplier: "TyAcc",
    shift: int32,
    input_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
    position: int32,
) -> "TyOut[H, L, D]":
    """Integer RoPE with Q15 tables and a runtime starting position."""

    Z: TyOut[H, L, D]
    for h, l, d in dsl.grid(H, L, D, name="qpositioned_rope"):
        partner: int32 = d + D // 2
        sign: int32 = -1
        if d >= D // 2:
            partner = d - D // 2
            sign = 1
        value: TyAcc = (X[h, l, d] - input_zero_point) * cos[position + l, d]
        value += sign * (X[h, l, partner] - input_zero_point) * sin[
            position + l, d
        ]
        scaled: TyAcc = value * multiplier
        rounded: TyAcc = scaled
        if shift > 0:
            magnitude: TyAcc = scaled
            if magnitude < 0:
                magnitude = -magnitude
            quotient: TyAcc = magnitude >> shift
            remainder: TyAcc = magnitude - (quotient << shift)
            one: TyAcc = 1
            halfway: TyAcc = one << (shift - 1)
            increment: TyAcc = 0
            if remainder > halfway:
                increment = 1
            if remainder == halfway and (quotient & 1) == 1:
                increment = 1
            rounded = quotient + increment
            if scaled < 0:
                rounded = -rounded
        if shift < 0:
            rounded = scaled << (0 - shift)
        quantized: TyAcc = rounded + output_zero_point
        if quantized < qmin:
            quantized = qmin
        if quantized > qmax:
            quantized = qmax
        Z[h, l, d] = quantized
    return Z


def qcausal_softmax3d[
    TyIn, TyOut, H, L, S
](
    X: "TyIn[H, L, S]",
    exp_table: "int32[256]",
    output_multiplier: int64,
    output_shift: int32,
    causal_offset: int32,
    input_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[H, L, S]":
    """Integer causal softmax using an input-scale-specific exponential LUT."""

    Z: TyOut[H, L, S]
    E: int32[H, L, S] = 0
    row_max: int32[H, L] = -2147483647
    row_sum: int64[H, L] = 0
    for h, l, s in dsl.grid(H, L, S, name="qsoftmax_row_max"):
        if s < causal_offset + l + 1 and X[h, l, s] > row_max[h, l]:
            row_max[h, l] = X[h, l, s]
    for h, l, s in dsl.grid(H, L, S, name="qsoftmax_exp_sum"):
        if s < causal_offset + l + 1:
            delta: int32 = row_max[h, l] - X[h, l, s]
            E[h, l, s] = exp_table[delta]
            row_sum[h, l] += E[h, l, s]
    for h, l, s in dsl.grid(H, L, S, name="qsoftmax_normalize"):
        quantized: int64 = output_zero_point
        if s < causal_offset + l + 1 and row_sum[h, l] > 0:
            numerator: int64 = E[h, l, s] * output_multiplier
            denominator: int64 = row_sum[h, l] << output_shift
            quotient: int64 = numerator // denominator
            remainder: int64 = numerator - quotient * denominator
            if remainder * 2 > denominator:
                quotient += 1
            if remainder * 2 == denominator and (quotient & 1) == 1:
                quotient += 1
            quantized = quotient + output_zero_point
        if quantized < qmin:
            quantized = qmin
        if quantized > qmax:
            quantized = qmax
        Z[h, l, s] = quantized
    return Z


def qrms_norm3d[
    TyIn, TyW, TyAcc, TyOut, B, L, D
](
    X: "TyIn[B, L, D]",
    weight: "TyW[D]",
    factor_multiplier: "TyAcc",
    eps_codes: "TyAcc",
    input_zero_point: int32,
    weight_zero_point: int32,
    output_zero_point: int32,
    qmin: int32,
    qmax: int32,
) -> "TyOut[B, L, D]":
    """Integer RMSNorm using a widened sum of squares and integer square root."""

    Z: TyOut[B, L, D]
    for b, l in dsl.grid(B, L, name="qrms_rows"):
        sum_sq: TyAcc = 0
        for d_sum in range(D):
            centered: TyAcc = X[b, l, d_sum] - input_zero_point
            sum_sq += centered * centered
        radicand: TyAcc = (sum_sq + eps_codes) << 24
        low: TyAcc = 0
        high: TyAcc = radicand + 1
        for root_step in range(63):
            if low + 1 < high:
                midpoint: TyAcc = (low + high) // 2
                if midpoint == 0 or midpoint <= radicand // midpoint:
                    low = midpoint
                else:
                    high = midpoint
        root_q12: TyAcc = low
        if root_q12 < 1:
            root_q12 = 1
        denominator: TyAcc = root_q12 << 8
        for d in range(D):
            x_centered: TyAcc = X[b, l, d] - input_zero_point
            w_centered: TyAcc = weight[d] - weight_zero_point
            numerator: TyAcc = x_centered * w_centered * factor_multiplier
            magnitude: TyAcc = numerator
            if magnitude < 0:
                magnitude = -magnitude
            quotient: TyAcc = magnitude // denominator
            remainder: TyAcc = magnitude - quotient * denominator
            if remainder * 2 > denominator:
                quotient += 1
            if remainder * 2 == denominator and (quotient & 1) == 1:
                quotient += 1
            if numerator < 0:
                quotient = -quotient
            quantized: TyAcc = quotient + output_zero_point
            if quantized < qmin:
                quantized = qmin
            if quantized > qmax:
                quantized = qmax
            Z[b, l, d] = quantized
    return Z


def qembedding2d[
    TyW, B, L, V, D
](input_ids: "int32[B, L]", weight: "TyW[V, D]") -> "TyW[B, L, D]":
    """Integer token embedding lookup."""

    Z: TyW[B, L, D]
    for b, l, d in dsl.grid(B, L, D, name="qembedding_lookup"):
        Z[b, l, d] = weight[input_ids[b, l], d]
    return Z


def qkv_cache_update3d[
    TyValue, TyCache, TyAcc, H, L, S, D
](
    values: "TyValue[H, L, D]",
    cache: "TyCache[H, S, D]",
    multiplier: "TyAcc",
    shift: int32,
    value_zero_point: int32,
    cache_zero_point: int32,
    qmin: int32,
    qmax: int32,
    position: int32,
) -> "TyCache[H, S, D]":
    """Update an integer KV cache, reconciling the incoming tensor scale."""

    Z: TyCache[H, S, D]
    for h, s, d in dsl.grid(H, S, D, name="qkv_cache_copy"):
        Z[h, s, d] = cache[h, s, d]
    for h, l, d in dsl.grid(H, L, D, name="qkv_cache_update"):
        if position + l < S:
            centered: TyAcc = values[h, l, d] - value_zero_point
            scaled: TyAcc = centered * multiplier
            rounded: TyAcc = scaled
            if shift > 0:
                magnitude: TyAcc = scaled
                if magnitude < 0:
                    magnitude = -magnitude
                quotient: TyAcc = magnitude >> shift
                remainder: TyAcc = magnitude - (quotient << shift)
                one: TyAcc = 1
                halfway: TyAcc = one << (shift - 1)
                increment: TyAcc = 0
                if remainder > halfway:
                    increment = 1
                if remainder == halfway and (quotient & 1) == 1:
                    increment = 1
                rounded = quotient + increment
                if scaled < 0:
                    rounded = -rounded
            if shift < 0:
                rounded = scaled << (0 - shift)
            quantized: TyAcc = rounded + cache_zero_point
            if quantized < qmin:
                quantized = qmin
            if quantized > qmax:
                quantized = qmax
            Z[h, position + l, d] = quantized
    return Z


# END NATIVE INT8 QUANTIZATION: ADDED transformer kernels


def schedule_native_quantized(s):
    """Correctness-first schedule for native integer boundary kernels."""

    return s


# END NATIVE INT8 QUANTIZATION: ADDED rank-2/rank-3 integer kernels
