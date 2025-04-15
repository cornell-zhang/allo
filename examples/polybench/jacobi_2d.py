# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def jacobi_2d_np(A, B, TSTEPS):
    for t in range(TSTEPS):
        for i in range(1, A.shape[0] - 1):
            for j in range(1, A.shape[1] - 1):
                B[i, j] = 0.2 * (
                    A[i, j + 1] + A[i, j - 1] + A[i - 1, j] + A[i + 1, j] + A[i, j]
                )
        for i in range(1, A.shape[0] - 1):
            for j in range(1, A.shape[1] - 1):
                A[i, j] = 0.2 * (
                    B[i, j + 1] + B[i, j - 1] + B[i - 1, j] + B[i + 1, j] + B[i, j]
                )
    return A, B


def compute_A[T: (float32, int32), N: int32](A0: "T[N, N]", B0: "T[N, N]"):
    for i0, j0 in allo.grid(N - 2, N - 2, name="A"):
        B0[i0 + 1, j0 + 1] = 0.2 * (
            A0[i0, j0 + 1]
            + A0[i0 + 1, j0]
            + A0[i0 + 1, j0 + 1]
            + A0[i0 + 1, j0 + 2]
            + A0[i0 + 2, j0 + 1]
        )


def compute_B[T: (float32, int32), N: int32](B1: "T[N, N]", A1: "T[N, N]"):
    for i1, j1 in allo.grid(N - 2, N - 2, name="B"):
        A1[i1 + 1, j1 + 1] = 0.2 * (
            B1[i1, j1 + 1]
            + B1[i1 + 1, j1]
            + B1[i1 + 1, j1 + 1]
            + B1[i1 + 1, j1 + 2]
            + B1[i1 + 2, j1 + 1]
        )


def kernel_jacobi_2d[T: (float32, int32), N: int32](A: "T[N, N]", B: "T[N, N]"):
    for m in range(TSTEPS):
        compute_A(A, B)
        compute_B(B, A)


def jacobi_2d(concrete_type, TSTEPS, N):
    sch0 = allo.customize(compute_A, instantiate=[concrete_type, N])
    lb0 = sch0.reuse_at(sch0.A0, "i0")
    wb0 = sch0.reuse_at(lb0, "j0")
    sch0.pipeline("i0")
    sch0.partition(lb0, dim=0)
    sch0.partition(wb0, dim=0)

    sch1 = allo.customize(compute_B, instantiate=[concrete_type, N])
    lb1 = sch1.reuse_at(sch1.B1, "i1")
    wb1 = sch1.reuse_at(lb1, "j1")
    sch1.pipeline("i1")
    sch1.partition(lb1, dim=0)
    sch1.partition(wb1, dim=0)

    sch = allo.customize(kernel_jacobi_2d, instantiate=[concrete_type, N])
    sch.compose(sch0)
    sch.compose(sch1)
    sch.partition(sch.A, dim=2)
    sch.partition(sch.B, dim=2)
    return sch


def test_jacobi_2d():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    N = psize["jacobi_2d"][test_psize]["N"]
    TSTEPS = psize["jacobi_2d"][test_psize]["TSTEPS"]
    concrete_type = float32
    sch = jacobi_2d(concrete_type, TSTEPS, N)
    mod = sch.build()
    # functional correctness test
    A = np.random.randint(10, size=(N, N)).astype(np.float32)
    B = np.random.randint(10, size=(N, N)).astype(np.float32)
    A_ref = A.copy()
    B_ref = B.copy()
    A_ref, B_ref = jacobi_2d_np(A_ref, B_ref, TSTEPS)
    mod = sch.build()
    mod(A, B)
    np.testing.assert_allclose(A, A_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(B, B_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
