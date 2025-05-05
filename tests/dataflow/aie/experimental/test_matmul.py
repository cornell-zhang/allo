# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")


def _test_gemm_1D():
    Ty = int32
    M, N, K = 16, 16, 16
    P0 = 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N], C: Ty[M, N] @ LyA):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_gemm_2D():
    TyI, TyO = int32, int32
    M, N, K = 16, 16, 16
    P0, P1 = 4, 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_gemm_1D()
    # failed
    # _test_gemm_2D()
