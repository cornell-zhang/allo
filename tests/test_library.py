# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int8, int16, int32, float32, index


def test_type_parameter():
    def kernel[
        TyA, TyB, TyC, M: int32, N: int32, K: int32
    ](A: "TyA[M, K]", B: "TyB[K, N]") -> "TyC[M, N]":
        C: TyC[M, N] = 0
        for i in range(M):
            for j in range(N):
                acc: TyC = 0
                for k in range(K):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc
        return C

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


if __name__ == "__main__":
    test_type_parameter()
