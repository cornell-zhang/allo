# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unsupported-assignment-operation

from .. import dsl
from .systolic import systolic


def softmax[Ty, L](X: "Ty[L, L]") -> "Ty[L, L]":
    Z: Ty[L, L]
    E: Ty[L, L]
    M: Ty[L] = -10000.0
    S: Ty[L] = 0.0

    for i, j in dsl.grid(L, L, name="row_max"):
        if X[i,j] > M[i]:
            M[i] = X[i, j]

    #compute exp and sum
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
