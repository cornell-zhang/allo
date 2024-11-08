# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, float32
import allo.dataflow as df
import numpy as np


def _test_matrix_scalar_add():
    Ty = int32
    M, N = 64, 64
    P0 = 4

    @df.kernel(mapping=[P0])
    def core(A: Ty[M, N], B: Ty[M, N]):
        for i, j in allo.grid(M // P0, N):
            B[i, j] = A[i, j] + 1

    top = df.build(core, target="aie")
    A = np.random.randint(0, 100, (M, N)).astype(np.int32)
    B = np.zeros((M, N)).astype(np.int32)
    top(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")


def _test_matrix_matrix_add():
    Ty = int32
    M, N = 64, 64
    P0 = 4

    @df.kernel(mapping=[P0])
    def core(A: Ty[M, N], B: Ty[M, N], C: Ty[M, N]):
        for i, j in allo.grid(M // P0, N):
            C[i, j] = A[i, j] + B[i, j]

    top = df.build(core, target="aie")
    A = np.random.randint(0, 100, (M, N)).astype(np.int32)
    B = np.random.randint(0, 100, (M, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    top(A, B, C)
    np.testing.assert_allclose(C, A + B)
    print("PASSED!")


if __name__ == "__main__":
    _test_matrix_scalar_add()
    _test_matrix_matrix_add()
