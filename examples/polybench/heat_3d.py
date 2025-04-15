# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def heat_3d_np(A, B, TSTEPS, N):
    for m in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    B[i, j, k] = (
                        0.125 * (A[i + 1, j, k] - 2.0 * A[i, j, k] + A[i - 1, j, k])
                        + 0.125 * (A[i, j + 1, k] - 2.0 * A[i, j, k] + A[i, j - 1, k])
                        + 0.125 * (A[i, j, k + 1] - 2.0 * A[i, j, k] + A[i, j, k - 1])
                        + A[i, j, k]
                    )

                    A[i, j, k] = (
                        0.125 * (B[i + 1, j, k] - 2.0 * B[i, j, k] + B[i - 1, j, k])
                        + 0.125 * (B[i, j + 1, k] - 2.0 * B[i, j, k] + B[i, j - 1, k])
                        + 0.125 * (B[i, j, k + 1] - 2.0 * B[i, j, k] + B[i, j, k - 1])
                        + B[i, j, k]
                    )
    return A, B


def kernel_heat_3d[
    T: (float32, int32), TSTEPS: int32, N: int32
](A: "T[N, N, N]", B: "T[N, N, N]"):
    const0: float32 = 0.125
    const1: float32 = 2.0

    for m in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    B[i, j, k] = (
                        const0 * (A[i + 1, j, k] - const1 * A[i, j, k] + A[i - 1, j, k])
                        + const0
                        * (A[i, j + 1, k] - const1 * A[i, j, k] + A[i, j - 1, k])
                        + const0
                        * (A[i, j, k + 1] - const1 * A[i, j, k] + A[i, j, k - 1])
                        + A[i, j, k]
                    )

                    A[i, j, k] = (
                        const0 * (B[i + 1, j, k] - const1 * B[i, j, k] + B[i - 1, j, k])
                        + const0
                        * (B[i, j + 1, k] - const1 * B[i, j, k] + B[i, j - 1, k])
                        + const0
                        * (B[i, j, k + 1] - const1 * B[i, j, k] + B[i, j, k - 1])
                        + B[i, j, k]
                    )


def heat_3d(concrete_type, tsteps, nn):
    s = allo.customize(kernel_heat_3d, instantiate=[concrete_type, tsteps, nn])
    return s.build()


def test_heat_3d():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    TSTEPS = psize["heat_3d"][test_psize]["TSTEPS"]
    N = psize["heat_3d"][test_psize]["N"]
    A = np.random.randint(10, size=(N, N, N)).astype(np.float32)
    B = np.random.randint(10, size=(N, N, N)).astype(np.float32)
    A_ref = A.copy()
    B_ref = B.copy()
    mod = heat_3d(float32, TSTEPS, N)
    mod(A, B)
    A_ref, B_ref = heat_3d_np(A_ref, B_ref, TSTEPS, N)
    np.testing.assert_allclose(A, A_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(B, B_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
