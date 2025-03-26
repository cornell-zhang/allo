# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def jacobi_1d_np(A, B, TSTEPS, N):
    for m in range(TSTEPS):
        for i0 in range(1, N - 1):
            B[i0] = 0.33333 * (A[i0 - 1] + A[i0] + A[i0 + 1])

        for i1 in range(1, N - 1):
            A[i1] = 0.33333 * (B[i1 - 1] + B[i1] + B[i1 + 1])
    return A, B


def kernel_jacobi_1d[
    T: (float32, int32), TSTEPS: int32, N: int32
](A: "T[N]", B: "T[N]"):
    for m in range(TSTEPS):
        for i0 in range(1, N - 1):
            B[i0] = 0.33333 * (A[i0 - 1] + A[i0] + A[i0 + 1])

        for i1 in range(1, N - 1):
            A[i1] = 0.33333 * (B[i1 - 1] + B[i1] + B[i1 + 1])


def jacobi_1d(concrete_type, tsteps, nn):
    s = allo.customize(kernel_jacobi_1d, instantiate=[concrete_type, tsteps, nn])
    return s.build()


def test_jacobi_1d():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    N = psize["jacobi_1d"][test_psize]["N"]
    TSTEPS = psize["jacobi_1d"][test_psize]["TSTEPS"]
    concrete_type = float32
    mod = jacobi_1d(concrete_type, TSTEPS, N)
    # functional correctness test
    A = np.random.randint(10, size=(N,)).astype(np.float32)
    B = np.random.randint(10, size=(N,)).astype(np.float32)
    A_ref = A.copy()
    B_ref = B.copy()
    mod(A, B)
    A_ref, B_ref = jacobi_1d_np(A_ref, B_ref, TSTEPS, N)
    np.testing.assert_allclose(A, A_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(B, B_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
