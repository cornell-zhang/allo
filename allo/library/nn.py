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
