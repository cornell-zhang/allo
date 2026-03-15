# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
import allo
from allo.ir.types import int32, index, uint1, float32
from allo.dsl import grid
from allo.spmw import kernel


def test_grid_loop():
    @kernel
    def kernel1(A: int32[32], B: int32[32]) -> int32[32]:
        C: int32[32] = 0
        for i, _ in allo.grid(32, 2):
            C[i] = A[i] + B[i]
        return C

    s = process(kernel1)
    np_A = np.random.randint(0, 255, (32,), dtype=np.int32)
    np_B = np.random.randint(0, 255, (32,), dtype=np.int32)
    np_C = s(np_A, np_B)
    assert np.allclose(np_C, np_A + np_B)

    @kernel
    def kernel2(A: int32[32], B: int32[32]) -> int32[32]:
        C: int32[32] = 0
        for i, _ in grid(32, 2):
            C[i] = A[i] + B[i]
        return C

    s = process(kernel2)
    np_A = np.random.randint(0, 255, (32,), dtype=np.int32)
    np_B = np.random.randint(0, 255, (32,), dtype=np.int32)
    np_C = s(np_A, np_B)
    assert np.allclose(np_C, np_A + np_B)

    @kernel
    def kernel3(A: int32[32], B: int32[32]) -> int32[32]:
        C: int32[32] = 0
        for i, _ in allo.grid(32, 2):
            C[i] = C[i] + A[i] + B[i]
        return C

    s = process(kernel3)
    np_A = np.random.randint(0, 255, (32,), dtype=np.int32)
    np_B = np.random.randint(0, 255, (32,), dtype=np.int32)
    np_C = s(np_A, np_B)
    assert np.allclose(np_C, (np_A + np_B) * 2)

    @kernel
    def kernel4(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j in allo.grid(32, 32):
            v: int32 = 0
            for k in range(32):
                v += A[i, k] * B[k, j]
            C[i, j] = v
        return C

    s = process(kernel4)
    np_A = np.random.randint(0, 255, (32, 32), dtype=np.int32)
    np_B = np.random.randint(0, 255, (32, 32), dtype=np.int32)
    np_C = s(np_A, np_B)
    assert np.allclose(np_C, np.matmul(np_A, np_B))

    @kernel
    def kernel5(A: int32[32], factor: int32) -> int32[32]:
        B: int32[32] = 0
        for j, i in allo.grid(factor, 32):
            B[i] += A[i]
        for _, i in allo.grid(factor, 32):
            B[i] += A[i]
        return B

    s = process(kernel5)
    np_A = np.random.randint(0, 255, (32,), dtype=np.int32)
    np_factor = 8
    np_B = s(np_A, np_factor)
    assert np.allclose(np_B, np_A * np_factor * 2)

    @kernel
    def kernel6(A: int32[32], factor: int32) -> int32[32]:
        B: int32[32] = 0
        for i, j in allo.grid(32, factor):
            B[i] += A[i]
        return B

    s = process(kernel6)
    np_A = np.random.randint(0, 255, (32,), dtype=np.int32)
    np_factor = 8
    np_B = s(np_A, np_factor)
    assert np.allclose(np_B, np_A * np_factor)

    M = 10

    @kernel
    def kernel7(
        A: float32[M, M, M, M], B: float32[M, M, M, M], C: float32[M, M]
    ) -> (float32[M, M, M, M], float32[M, M, M, M], float32[M, M]):
        res0: float32[M, M, M, M] = 0
        res1: float32[M, M, M, M] = 0
        for i, j, k, l in allo.grid(M, M, M, M):
            res0[i, j, k, l] = A[i, j, k, l] + 1
            res1[i, j, k, l] = B[i, j, k, l] + 1
        res2: float32[M, M] = 0
        for i, j in allo.grid(M, M):
            res2[i, j] = C[i, j] + 1
        return res0, res1, res2

    s = process(kernel7)
    np_A = np.random.random((M, M, M, M)).astype(np.float32)
    np_B = np.random.random((M, M, M, M)).astype(np.float32)
    np_C = np.random.random((M, M)).astype(np.float32)
    np_res0, np_res1, np_res2 = s(np_A, np_B, np_C)
    np.testing.assert_allclose(np_res0, np_A + 1)
    np.testing.assert_allclose(np_res1, np_B + 1)
    np.testing.assert_allclose(np_res2, np_C + 1)

    print("test_grid_loop passed")


def test_reduction_loop():
    @kernel
    def kernel1(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j in allo.grid(32, 32):
            v: int32 = 0
            for k in allo.reduction(32):
                v += A[i, k] * B[k, j]
            C[i, j] = v
        return C

    s = process(kernel1)
    np_A = np.random.randint(0, 255, (32, 32), dtype=np.int32)
    np_B = np.random.randint(0, 255, (32, 32), dtype=np.int32)
    np_C = s(np_A, np_B)
    assert np.allclose(np_C, np.matmul(np_A, np_B))

    print("test_reduction_loop passed")


if __name__ == "__main__":
    test_grid_loop()
    test_reduction_loop()
