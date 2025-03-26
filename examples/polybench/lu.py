# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def lu_np(A):
    N = A.shape[0]
    for i in range(N):
        for j in range(i):
            for k in range(j):
                A[i, j] -= A[i, k] * A[k, j]
            A[i, j] /= A[j, j]

        for j in range(i, N):
            for k in range(i):
                A[i, j] -= A[i, k] * A[k, j]
    return A


def kernel_lu[T: (float32, int32), N: int32](A: "T[N, N]"):
    for i in range(N):
        for j in range(i):
            for k in range(j):
                A[i, j] -= A[i, k] * A[k, j]
            A[i, j] /= A[j, j]

        for j in range(i, N):
            for k in range(i):
                A[i, j] -= A[i, k] * A[k, j]


def lu(concrete_type, n):
    s0 = allo.customize(kernel_lu, instantiate=[concrete_type, n])
    return s0.build()


def test_lu():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"

    # generate input data
    N = psize["lu"][test_psize]["N"]
    A = np.random.rand(N, N).astype(np.float32)

    # run reference
    A_ref = A.copy()
    A_ref = lu_np(A_ref)

    # run allo
    A_opt = A.copy()
    lu(float32, N)(A_opt)

    # verify
    np.testing.assert_allclose(A_ref, A_opt, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
