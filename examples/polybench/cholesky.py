# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def cholesky_np(A):
    N = A.shape[0]
    for i in range(N):
        # j < i
        for j in range(i):
            for k in range(j):
                A[i, j] = A[i, j] - A[i, k] * A[j, k]
            A[i, j] = A[i, j] / A[j, j]
        # i == j
        for k in range(i):
            A[i, i] = A[i, i] - A[i, k] * A[i, k]
        A[i, i] = np.sqrt(A[i, i] * 1.0)
    return A


def kernel_cholesky[T: (int32, float32), N: int32](A: "T[N, N]"):
    for i in range(N):
        # Case: j < i
        for j in range(i):
            for k in range(j):
                A[i, j] = A[i, j] - A[i, k] * A[j, k]
            A[i, j] = A[i, j] / A[j, j]
        # Case: i == j
        for k in range(i):
            A[i, i] = A[i, i] - A[i, k] * A[i, k]
        A[i, i] = allo.sqrt(A[i, i] * 1.0)


def cholesky(concrete_type, n):
    s = allo.customize(kernel_cholesky, instantiate=[concrete_type, n])
    return s.build()


def test_cholesky():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"

    # generate input data
    N = psize["cholesky"][test_psize]["N"]
    A = np.random.rand(N, N).astype(np.float32)

    # run reference
    A_ref = A.copy()
    A_ref = cholesky_np(A_ref)

    # run allo
    A_opt = A.copy()
    s = cholesky(float32, N)
    s(A_opt)

    # verify
    np.testing.assert_allclose(A_ref, A_opt, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
