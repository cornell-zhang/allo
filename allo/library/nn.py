# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unsupported-assignment-operation

from .. import dsl


def softmax[Ty, D](X: "Ty[D, D]") -> "Ty[D, D]":
    Z: Ty[D, D]
    exp: Ty[D, D]
    row_sum: Ty[D] = 0.0

    for i, j in dsl.grid(D, D, name="exp_sum"):
        exp[i, j] = dsl.exp(X[i, j])
        row_sum[i] += exp[i, j]

    for i, j in dsl.grid(D, D, name="update"):
        Z[i, j] = exp[i, j] / row_sum[i]
    return Z


def layernorm[Ty, L, D](X: "Ty[L, D]", gamma: "Ty[D]", beta: "Ty[D]") -> "Ty[L, D]":
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
