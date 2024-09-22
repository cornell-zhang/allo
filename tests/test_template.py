# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int8, int32, float32


def kernel[
    TyA, TyB, TyC, M: int32, K: int32, N: int32
](A: "TyA[M, K]", B: "TyB[K, N]") -> "TyC[M, N]":
    C: TyC[M, N]
    for i in range(M):
        for j in range(N):
            acc: TyC = 0
            for k in range(K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc
    return C


def test_different_functions():
    def top_float(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
        return kernel[float32, float32, float32, 32, 32, 32](A, B)

    s = allo.customize(top_float)
    mod = s.build()
    print(s.module)
    np_A = np.random.randn(32, 32).astype(np.float32)
    np_B = np.random.randn(32, 32).astype(np.float32)
    np_C = np.matmul(np_A, np_B)
    allo_C = mod(np_A, np_B)
    np.testing.assert_allclose(np_C, allo_C, rtol=1e-4, atol=1e-4)

    def top_int(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        return kernel[int32, int32, int32, 32, 32, 32](A, B)

    s = allo.customize(top_int)
    mod = s.build()
    np_A = np.random.randn(32, 32).astype(np.int32)
    np_B = np.random.randn(32, 32).astype(np.int32)
    np_C = np.matmul(np_A, np_B)
    allo_C = mod(np_A, np_B)
    np.testing.assert_allclose(np_C, allo_C, rtol=1e-4, atol=1e-4)
    print("Passed!")


def test_same_function():
    M, K, N, P = 32, 64, 64, 32

    def top(A: float32[M, K], B: float32[K, N], W: float32[N, P]) -> float32[M, P]:
        C = kernel[float32, float32, float32, M, K, N](A, B)  # MxK x KxN = MxN
        D = kernel[float32, float32, float32, M, N, P](C, W)  # MxN x NxP = MxP
        return D

    s = allo.customize(top)
    print(s.module)
    mod = s.build()
    np_A = np.random.randn(M, K).astype(np.float32)
    np_B = np.random.randn(K, N).astype(np.float32)
    np_W = np.random.randn(N, P).astype(np.float32)
    np_C = (np_A @ np_B) @ np_W
    allo_C = mod(np_A, np_B, np_W)
    np.testing.assert_allclose(np_C, allo_C, rtol=1e-4, atol=1e-4)
    print("Passed!")


def test_expr_type():
    def kernel[Ty, M, N](A: "Ty[M + 1, N * 2]") -> "Ty[M, N]":
        B: Ty[M, N] = 0
        return B

    def top[Ty](X: "Ty[17, 64]") -> "Ty[16, 32]":
        return kernel[Ty, 16, 32](X)

    s = allo.customize(top, instantiate=[float32])
    print(s.module)


def test_expr_param():
    def kernel[Ty, M, N](A: "Ty[M + 1, N * 2]") -> "Ty[M, N]":
        B: Ty[M, N] = 0
        return B

    def top[Ty, M, N](X: "Ty[M + 3, N]") -> "Ty[M + 2, N // 2]":
        return kernel[Ty, M + 2, N // 2](X)

    s = allo.customize(top, instantiate=[float32, 16, 16])
    print(s.module)


def test_meta_if():
    def kernel_int8[M, N]() -> "int8[M, N]":
        B: int8[M, N] = 0
        return B

    def kernel_float32[M, N]() -> "float32[M, N]":
        B: float32[M, N] = 0
        return B

    def kernel_int32[M, N]() -> "int32[M, N]":
        B: int32[M, N] = 0
        return B

    def top[Ty, M, N]() -> "Ty[M, N]":
        with allo.meta_if(Ty == int8):
            return kernel_int8[M, N]()
        with allo.meta_elif(Ty == float32):
            return kernel_float32[M, N]()
        with allo.meta_else():
            with allo.meta_if(M + 2 == N + 2):
                A = kernel_int32[M * 2, N * 2]()
            B: int32[M, N] = 0
            return B

    s = allo.customize(top, instantiate=[int8, 16, 16])
    assert "16x16xi8" in str(s.module)
    print(s.module)
    s = allo.customize(top, instantiate=[float32, 20, 20])
    assert "20x20xf32" in str(s.module)
    print(s.module)
    s = allo.customize(top, instantiate=[int32, 32, 32])
    assert "64x64xi32" in str(s.module)
    print(s.module)


def test_double_meta_if():
    M, N = 32, 32

    def top(A: int32[M]):
        with allo.meta_if(M == 1):
            for i in range(M):
                A[i] = A[i] // 2
        with allo.meta_elif(M == 64):
            for i in range(M):
                A[i] = A[i] * 2
        with allo.meta_else():
            for i in range(M):
                A[i] = A[i] + 1
        with allo.meta_if(N == 32):
            for i in range(M):
                A[i] = A[i] - 1

    s = allo.customize(top)
    print(s.module)
    mod = s.build()
    np_A = np.random.randn(M).astype(np.int32)
    np_golden = np_A.copy()
    mod(np_A)
    np.testing.assert_allclose(np_A, np_golden, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
