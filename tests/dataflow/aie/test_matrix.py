# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16, int32, float32
import allo.dataflow as df
import numpy as np


def _test_matrix_scalar_add():
    Ty = int32
    M, N = 64, 64
    P0 = 4
    Mt = M // P0

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M, N], B: Ty[M, N]):
            pi = df.get_pid()
            B[pi * Mt : (pi + 1) * Mt, :] = allo.add(A[pi * Mt : (pi + 1) * Mt, :], 1)

    A = np.random.randint(0, 100, (M, N)).astype(np.int32)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        B = np.zeros((M, N)).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    sim_mod = df.build(top, target="simulator")
    B = np.zeros((M, N)).astype(np.int32)
    sim_mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("Dataflow Simulator Passed!")


def _test_matrix_matrix_add():
    Ty = int32
    M, N = 64, 64
    P0 = 4
    Mt = M // P0

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M, N], B: Ty[M, N], C: Ty[M, N]):
            pi = df.get_pid()
            C[pi * Mt : (pi + 1) * Mt, :] = allo.add(
                A[pi * Mt : (pi + 1) * Mt, :], B[pi * Mt : (pi + 1) * Mt, :]
            )

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
    sim_mod = df.build(top, target="simulator")
    C = np.zeros((M, N)).astype(np.int32)
    sim_mod(A, B, C)
    np.testing.assert_allclose(C, A + B)
    print("Dataflow Simulator Passed!")


def _test_gemm_1D():
    Ty = int16
    M, N, K = 16, 16, 16
    P0 = 2
    Mt = M // P0

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
            pi = df.get_pid()
            C[pi * Mt : (pi + 1) * Mt, :] = allo.matmul(
                A[pi * Mt : (pi + 1) * Mt, :], B
            )

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
    Mt, Nt = M // P0, N // P1

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def gemm(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
            p0, p1 = df.get_pid()
            C[p0 * Mt : (p0 + 1) * Mt, p1 * Nt : (p1 + 1) * Nt] = allo.matmul(
                A[p0 * Mt : (p0 + 1) * Mt, :], B[:, p1 * Nt : (p1 + 1) * Nt]
            )

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
