# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int32, float32


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


if __name__ == "__main__":
    pytest.main([__file__])
