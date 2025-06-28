# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

def _test_matrix_scalar_add():
    LyA = Layout("S0R")
    Ty = int32
    M, N = 32, 32
    P0 = 1

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M, N] @ LyA, B: Ty[M, N] @ LyA):
            B[:, :] = allo.add(A, 1)

    A = np.random.randint(0, 100, (M, N)).astype(np.int32)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        B = np.zeros((M, N)).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A.T + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")


def _test_gemm_1D():
    Ty = int16
    M, N, K = 16, 16, 16
    P0 = 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N], C: Ty[M, N] @ LyA):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
    B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")

def _test_gemm_2D():
    TyI, TyO = int16, int16
    M, N, K = 32, 32, 32
    P0, P1 = 4, 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(
        top, target="aie-mlir"
        )
    A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
    B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")

if __name__ == "__main__":
    # _test_matrix_scalar_add()
    _test_gemm_1D()
    # _test_gemm_2D()
