# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
from allo.ir.types import int32, int64, float32, float16
from allo.spmw import kernel


def test_basic_template():
    @kernel
    def kernel1[Ty, M, N](A: "Ty[M, N]") -> "Ty[M, N]":
        B: Ty[M, N] = A
        return B

    s = process(kernel1, instantiate=[float32, 4, 4])
    np_A = np.random.rand(4, 4).astype(np.float32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A)

    s = process(kernel1, instantiate=[int32, 8, 4])
    np_A = np.random.randint(0, 10, size=(8, 4)).astype(np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A)

    @kernel
    def kernel2[Ty, M, N](A: "Ty[M, N]") -> "Ty[M, N]":
        B: Ty[M, N] = A + 1
        return B

    s = process(kernel2, instantiate=[float32, 4, 4])
    np_A = np.random.rand(4, 4).astype(np.float32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A + 1)

    s = process(kernel2, instantiate=[int32, 8, 4])
    np_A = np.random.randint(0, 10, size=(8, 4)).astype(np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A + 1)

    @kernel
    def kernel2[TyA, TyB, TyC, M, K, N](A: "TyA[M, K]", B: "TyB[K, N]") -> "TyC[M, N]":
        C: TyC[M, N]
        for i in range(M):
            for j in range(N):
                acc: TyC = 0
                for k in range(K):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc
        return C

    s = process(kernel2, instantiate=[float32, float32, float32, 32, 16, 32])
    np_A = np.random.randn(32, 16).astype(np.float32)
    np_B = np.random.randn(16, 32).astype(np.float32)
    np_C = np.matmul(np_A, np_B)
    allo_C = s(np_A, np_B)
    assert np.allclose(np_C, allo_C, rtol=1e-4, atol=1e-4)

    s = process(kernel2, instantiate=[int32, int32, int32, 8, 4, 8])
    np_A = np.random.randint(0, 10, size=(8, 4)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(4, 8)).astype(np.int32)
    np_C = np.matmul(np_A, np_B)
    allo_C = s(np_A, np_B)
    assert np.array_equal(np_C, allo_C)

    print("test_basic_template passed")


def test_call_instance():
    @kernel
    def GEMM[TyA, TyB, TyC, M, K, N](A: "TyA[M, K]", B: "TyB[K, N]") -> "TyC[M, N]":
        C: TyC[M, N]
        for i in range(M):
            for j in range(N):
                acc: TyC = 0
                for k in range(K):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc
        return C

    @kernel
    def kernel1(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
        return GEMM[float32, float32, float32, 32, 32, 32](A, B)

    s = process(kernel1)
    np_A = np.random.randn(32, 32).astype(np.float32)
    np_B = np.random.randn(32, 32).astype(np.float32)
    np_C = np.matmul(np_A, np_B)
    allo_C = s(np_A, np_B)
    assert np.allclose(np_C, allo_C, rtol=1e-4, atol=1e-4)

    @kernel
    def kernel2[TyA, TyB, TyC, M, K, N](A: "TyA[M, K]", B: "TyB[K, N]") -> "TyC[M, N]":
        return GEMM[TyA, TyB, TyC, M, K, N](A, B)

    s = process(kernel2, instantiate=[float32, float32, float32, 32, 32, 32])
    np_A = np.random.randn(32, 32).astype(np.float32)
    np_B = np.random.randn(32, 32).astype(np.float32)
    np_C = np.matmul(np_A, np_B)
    allo_C = s(np_A, np_B)
    assert np.allclose(np_C, allo_C, rtol=1e-4, atol=1e-4)

    s = process(kernel2, instantiate=[int32, int32, int32, 8, 4, 8])
    np_A = np.random.randint(0, 10, size=(8, 4)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(4, 8)).astype(np.int32)
    np_C = np.matmul(np_A, np_B)
    allo_C = s(np_A, np_B)
    assert np.array_equal(np_C, allo_C)

    @kernel
    def kernel3[TyA, TyB, TyC, M, K, N](A: "TyA[M, K]", B: "TyB[K, N]") -> "TyC[M, N]":
        tmp: TyC[M, N] = GEMM[TyA, TyB, TyC, M, K, N](A, B)  # call same instance
        tmp = GEMM[TyA, TyB, TyC, M, K, N](A, B)
        return tmp

    s = process(kernel3, instantiate=[float32, float32, float32, 32, 32, 32])
    np_A = np.random.randn(32, 32).astype(np.float32)
    np_B = np.random.randn(32, 32).astype(np.float32)
    np_C = np.matmul(np_A, np_B)
    allo_C = s(np_A, np_B)
    assert np.allclose(np_C, allo_C, rtol=1e-4, atol=1e-4)

    s = process(kernel3, instantiate=[int32, int32, int32, 8, 4, 8])
    np_A = np.random.randint(0, 10, size=(8, 4)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(4, 8)).astype(np.int32)
    np_C = np.matmul(np_A, np_B)
    allo_C = s(np_A, np_B)
    assert np.array_equal(np_C, allo_C)

    @kernel
    def kernel4[
        TyA, TyB, TyC1, TyC2, M, K, N
    ](A: "TyA[M, K]", B: "TyB[K, N]") -> "TyC2[M, N]":
        # call different instances
        tmp: TyC1[M, N] = GEMM[TyA, TyB, TyC1, M, K, N](A, B)
        tmp2: TyC2[M, N] = GEMM[TyA, TyB, TyC2, M, K, N](A, B)
        return tmp2

    s = process(kernel4, instantiate=[int32, int32, int32, int64, 8, 4, 8])
    np_A = np.random.randint(0, 10, size=(8, 4)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(4, 8)).astype(np.int32)
    np_C = np.matmul(np_A, np_B)
    allo_C = s(np_A, np_B)
    assert np.array_equal(np_C, allo_C)

    print("test_call_instance passed")


if __name__ == "__main__":
    test_basic_template()
    test_call_instance()
