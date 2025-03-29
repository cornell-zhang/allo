# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32, index
import allo.ir.types as T


def ludcmp_np(A, b, x, y):
    N = A.shape[0]
    for i in range(N):
        for j in range(i):
            w = A[i, j]
            for k in range(j):
                w -= A[i, k] * A[k, j]
            A[i, j] = w / A[j, j]
        for j in range(i, N):
            w = A[i, j]
            for k in range(i):
                w -= A[i, k] * A[k, j]
            A[i, j] = w

    for i in range(N):
        w = b[i]
        for j in range(i):
            w -= A[i, j] * y[j]
        y[i] = w

    for i in range(N - 1, -1, -1):
        w = y[i]
        for j in range(i + 1, N):
            w -= A[i, j] * x[j]
        x[i] = w / A[i, i]
    return A, b, x, y


def kernel_ludcmp[
    T: (float32, int32), N: int32
](A: "T[N, N]", b: "T[N]", x: "T[N]", y: "T[N]"):
    # LU decomposition of A
    for i in range(N):
        for j in range(i):
            w: T = A[i, j]
            for k in range(j):
                w -= A[i, k] * A[k, j]
            A[i, j] = w / A[j, j]

        for j in range(i, N):
            w: T = A[i, j]
            for k in range(i):
                w -= A[i, k] * A[k, j]
            A[i, j] = w

    # Finding solution for LY = b
    for i in range(N):
        alpha: T = b[i]
        for j in range(i):
            alpha -= A[i, j] * y[j]
        y[i] = alpha

    # Finding solution for Ux = y
    # for i in range(N - 1, cst_neg_1, cst_neg_1):
    for i_inv in range(N):
        i: index = N - 1 - i_inv
        alpha: float32 = y[i]
        for j in range(i + 1, N):
            alpha -= A[i, j] * x[j]
        x[i] = alpha / A[i, i]


def ludcmp(concrete_type, n):
    s0 = allo.customize(kernel_ludcmp, instantiate=[concrete_type, n])
    return s0.build()


def test_ludcmp():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"

    # generate input data
    N = psize["ludcmp"][test_psize]["N"]
    A = np.random.rand(N, N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    x = np.zeros(N, dtype=np.float32)
    y = np.zeros(N, dtype=np.float32)

    # run reference
    A_ref = A.copy()
    b_ref = b.copy()
    x_ref = x.copy()
    y_ref = y.copy()
    A_ref, b_ref, x_ref, y_ref = ludcmp_np(A_ref, b_ref, x_ref, y_ref)

    # run allo
    A_opt = A.copy()
    b_opt = b.copy()
    x_opt = x.copy()
    y_opt = y.copy()
    ludcmp(float32, N)(A_opt, b_opt, x_opt, y_opt)

    np.testing.assert_allclose(A_ref, A_opt)
    np.testing.assert_allclose(b_ref, b_opt)
    np.testing.assert_allclose(x_ref, x_opt)
    np.testing.assert_allclose(y_ref, y_opt)


if __name__ == "__main__":
    pytest.main([__file__])
