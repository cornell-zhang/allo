# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import MemLayout
from allo.backend.aie import is_available

LyA = MemLayout("S0R")
LyB = MemLayout("RS1")
LyC = MemLayout("S0S1")


def test_matrix_scalar_add():
    Ty = int32
    M, N = 64, 64
    P0 = 4

    @df.region()
    def top(A: Ty[M, N], B: Ty[M, N]):
        @df.kernel(mapping=[P0], args=[A, B])
        def core(local_A: Ty[M, N] @ LyA, local_B: Ty[M, N] @ LyA):
            local_B[:, :] = allo.add(local_A, 1)

    A = np.random.randint(0, 100, (M, N)).astype(np.int32)

    if is_available():
        mod = df.build(top, target="aie")
        B = np.zeros((M, N)).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    # TODO: Fix AIE simulator
    # sim_mod = df.build(top, target="simulator")
    # B = np.zeros((M, N)).astype(np.int32)
    # sim_mod(A, B)
    # np.testing.assert_allclose(B, A + 1)
    # print("Dataflow Simulator Passed!")


def test_matrix_matrix_add():
    Ty = int32
    M, N = 64, 64
    P0 = 4

    @df.region()
    def top(A: Ty[M, N], B: Ty[M, N], C: Ty[M, N]):
        @df.kernel(mapping=[P0], args=[A, B, C])
        def core(
            local_A: Ty[M, N] @ LyA, local_B: Ty[M, N] @ LyA, local_C: Ty[M, N] @ LyA
        ):
            local_C[:, :] = allo.add(local_A, local_B)

    A = np.random.randint(0, 100, (M, N)).astype(np.int32)
    B = np.random.randint(0, 100, (M, N)).astype(np.int32)
    if is_available():
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


def test_gemm_1D():
    Ty = int16
    M, N, K = 16, 16, 16
    P0 = 2

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        @df.kernel(mapping=[P0], args=[A, B, C])
        def gemm(local_A: Ty[M, K] @ LyA, local_B: Ty[K, N], local_C: Ty[M, N] @ LyA):
            local_C[:, :] = allo.matmul(local_A, local_B)

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    if is_available():
        mod = df.build(top, target="aie")
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_gemm_1D_mixed():
    TyI = int16
    TyO = int32
    M, N, K = 16, 16, 16
    P0 = 2

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
        @df.kernel(mapping=[P0], args=[A, B, C])
        def gemm(
            local_A: TyI[M, K] @ LyA, local_B: TyI[K, N], local_C: TyO[M, N] @ LyA
        ):
            local_C_part: TyO[M // P0, N] = allo.matmul(local_A, local_B)
            local_C[:, :] = local_C_part

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int32)
    if is_available():
        mod = df.build(top, target="aie")
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_gemm_2D():
    TyI, TyO = int16, int32
    M, N, K = 64, 64, 32
    P0, P1 = 4, 4

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
        @df.kernel(mapping=[P0, P1], args=[A, B, C])
        def gemm(
            local_A: TyI[M, K] @ LyA, local_B: TyI[K, N] @ LyB, local_C: TyO[M, N] @ LyC
        ):
            local_C[:, :] = allo.matmul(local_A, local_B)

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(0, 64, (M, K)).astype(np.int16)
        B = np.random.randint(0, 64, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int32)
        mod(A, B, C)
        np_C = A.astype(np.int32) @ B.astype(np.int32)
        np.testing.assert_allclose(C, np_C, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_gemm_2D_mixed():
    TyI = int16
    TyO = int32
    M, N, K = 64, 64, 64
    P0, P1 = 4, 4

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
        @df.kernel(mapping=[P0, P1], args=[A, B, C])
        def gemm(
            local_A: TyI[M, K] @ LyA, local_B: TyI[K, N] @ LyB, local_C: TyO[M, N] @ LyC
        ):
            local_C_part: TyO[M // P0, N // P1] = allo.matmul(local_A, local_B)
            local_C[:, :] = local_C_part

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(-16, 16, (M, K)).astype(np.int16)
        B = np.random.randint(-16, 16, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_matrix_scalar_add()
    test_matrix_matrix_add()
    test_gemm_1D_mixed()
    test_gemm_2D_mixed()
    test_gemm_1D()

    # Allo flow
    test_gemm_2D()
    # allow modify
    TyI, TyO = int16, int32
    M, N, K = 64, 64, 32

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int32)
    allo.backend.aie._call_prj("top.prj", [TyI, TyI, TyO], 0, [0, 1], [2], A, B, C)
    np_C = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_allclose(C, np_C, atol=1e-5)
    print("PASSED!")
