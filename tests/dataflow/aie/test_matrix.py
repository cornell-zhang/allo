# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16, int32, float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

SpA = Layout("S0R")
SpB = Layout("RS1")
SpC = Layout("S0S1")


def _test_matrix_scalar_add():
    Ty = int32
    M, N = 64, 64
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M, N] @ SpA, B: Ty[M, N] @ SpA):
            B[:, :] = allo.add(A, 1)

    A = np.random.randint(0, 100, (M, N)).astype(np.int32)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        B = np.zeros((M, N)).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    # sim_mod = df.build(top, target="simulator")
    # B = np.zeros((M, N)).astype(np.int32)
    # sim_mod(A, B)
    # np.testing.assert_allclose(B, A + 1)
    # print("Dataflow Simulator Passed!")


def _test_matrix_matrix_add():
    Ty = int32
    M, N = 64, 64
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M, N] @ SpA, B: Ty[M, N] @ SpA, C: Ty[M, N] @ SpA):
            C[:, :] = allo.add(A, B)

    A = np.random.randint(0, 100, (M, N)).astype(np.int32)
    B = np.random.randint(0, 100, (M, N)).astype(np.int32)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        C = np.zeros((M, N)).astype(np.int32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A + B)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
    # sim_mod = df.build(top, target="simulator")
    # C = np.zeros((M, N)).astype(np.int32)
    # sim_mod(A, B, C)
    # np.testing.assert_allclose(C, A + B)
    # print("Dataflow Simulator Passed!")


def _test_gemm_1D():
    Ty = int16
    M, N, K = 16, 16, 16
    P0 = 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K] @ SpA, B: Ty[K, N], C: Ty[M, N] @ SpA):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(top, target="aie")
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_gemm_2D():
    TyI, TyO = int16, int32
    M, N, K = 16, 16, 16
    P0, P1 = 2, 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def gemm(A: TyI[M, K] @ SpA, B: TyI[K, N] @ SpB, C: TyO[M, N] @ SpC):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(top, target="aie")
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_matrix_scalar_add()
    _test_matrix_matrix_add()
    _test_gemm_1D()
    _test_gemm_2D()
