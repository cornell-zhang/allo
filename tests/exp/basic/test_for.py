# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
from allo.ir.types import int32, float32, float16
from allo.spmw import kernel


def test_single_affine_for():
    @kernel
    def kernel1(A: int32[20]) -> int32[20]:
        for i in range(10):
            A[i] = i
        return A

    s = process(kernel1)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        gold[i] = i
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel2(A: int32[20]) -> int32[20]:
        for i in range(10):
            A[i + 1] = i
        return A

    s = process(kernel2)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        gold[i + 1] = i
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel3(A: int32[20]) -> int32[20]:
        for i in range(10, 20):
            A[i - 1] = i
        return A

    s = process(kernel3)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10, 20):
        gold[i - 1] = i
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel4(A: int32[20]) -> int32[20]:
        for i in range(10):
            A[i * 2] = i
        return A

    s = process(kernel4)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        gold[i * 2] = i
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel5(A: int32[20]) -> int32[20]:
        for i in range(0, 20, 2):
            A[i / 2] = i
        return A

    s = process(kernel5)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(0, 20, 2):
        gold[i // 2] = i
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel6(A: int32[20]) -> int32[20]:
        for i in range(0, 20, 2):
            A[i // 2] = i
        return A

    s = process(kernel6)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(0, 20, 2):
        gold[i // 2] = i
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel7(A: int32[20]) -> int32[20]:
        for i in range(20):
            A[i % 10] = A[i % 10] + 1
        return A

    s = process(kernel7)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(20):
        gold[i % 10] = gold[i % 10] + 1
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel8(A: int32[20]):
        for i in range(10):
            A[i] = i
        for i in range(10, 20):
            A[i] = i
        for i in range(0, 20, 2):
            A[i] = i * 2

    s = process(kernel8)
    np_A = np.zeros((20,), dtype=np.int32)
    kernel8(np_A)
    np_B = np.zeros((20,), dtype=np.int32)
    s(np_B)
    np.testing.assert_allclose(np_A, np_B)

    print("pass test_single_affine_for")


def test_nested_affine_for():
    @kernel
    def kernel1(A: int32[20]) -> int32[20]:
        for i in range(10):
            for j in range(2):
                A[i * 2 + j] = i
        return A

    s = process(kernel1)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        for j in range(2):
            gold[i * 2 + j] = i
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel2(A: int32[20]) -> int32[20]:
        for i in range(10):
            for j in range(2):
                A[i * (1 + 1 + 0) + j] = i * 2 + j
        return A

    s = process(kernel2)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        for j in range(2):
            gold[i * (1 + 1 + 0) + j] = i * 2 + j
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel3(A: int32[20]) -> int32[20]:
        for i in range(5):
            for j in range(4):
                A[i * j] = A[i * j] + 1  # should not use affine load/store
        return A

    s = process(kernel3)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(5):
        for j in range(4):
            gold[i * j] = gold[i * j] + 1
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel4(A: int32[4, 5]) -> int32[4, 5]:
        for i in range(4):
            for j in range(5):
                A[i, j] = i * j
        return A

    s = process(kernel4)
    np_A = np.zeros((4, 5), dtype=np.int32)
    gold = np.zeros((4, 5), dtype=np.int32)
    for i in range(4):
        for j in range(5):
            gold[i, j] = i * j
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel5(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
        C: float32[32, 32]
        for i in range(32):
            for j in range(32):
                acc: float32 = 0
                for k in range(32):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc
        return C

    s = process(kernel5)
    np_A = np.random.randn(32, 32).astype(np.float32)
    np_B = np.random.randn(32, 32).astype(np.float32)
    np_C = np.matmul(np_A, np_B)
    allo_C = s(np_A, np_B)
    np.testing.assert_allclose(np_C, allo_C, rtol=1e-4, atol=1e-4)

    print("pass test_nested_affine_for")


def test_vadd():
    @kernel
    def vadd(A: float32[32], B: float32[32], C: float32[32]):
        for i in range(32):
            # syntax sugar for lib op "add"
            C[i] = A[i] + B[i]

    s = process(vadd)
    np_A = np.random.rand(32).astype(np.float32)
    np_B = np.random.rand(32).astype(np.float32)
    np_C = np.zeros((32,), dtype=np.float32)
    vadd(np_A, np_B, np_C)
    np_D = np.zeros((32,), dtype=np.float32)
    s(np_A, np_B, np_D)
    np.testing.assert_allclose(np_C, np_D)

    @kernel
    def vadd2(A: float32[32], B: float16[32], C: float32[32]):
        for i in range(32):
            C[i] = A[i] + B[i]

    s = process(vadd2)
    np_A = np.random.rand(32).astype(np.float32)
    np_B = np.random.rand(32).astype(np.float16)
    np_C = np.zeros((32,), dtype=np.float32)
    vadd2(np_A, np_B, np_C)
    np_D = np.zeros((32,), dtype=np.float32)
    s(np_A, np_B, np_D)
    np.testing.assert_allclose(np_C, np_D)

    @kernel
    def vadd3(A: float32[32, 4], B: float16[32, 4], C: float32[32, 4]):
        for i in range(32):
            for j in range(4):
                C[i, j] = A[i, j] + B[i, j]

    s = process(vadd3)
    np_A = np.random.rand(32, 4).astype(np.float32)
    np_B = np.random.rand(32, 4).astype(np.float16)
    np_C = np.zeros((32, 4), dtype=np.float32)
    vadd3(np_A, np_B, np_C)
    np_D = np.zeros((32, 4), dtype=np.float32)
    s(np_A, np_B, np_D)
    np.testing.assert_allclose(np_C, np_D)

    @kernel
    def vadd4(A: float32[32, 4], B: float16[32, 4], C: float32[32, 4]):
        for i in range(32):
            C[i] = A[i] + B[i]

    s = process(vadd4)
    np_A = np.random.rand(32, 4).astype(np.float32)
    np_B = np.random.rand(32, 4).astype(np.float16)
    np_C = np.zeros((32, 4), dtype=np.float32)
    vadd4(np_A, np_B, np_C)
    np_D = np.zeros((32, 4), dtype=np.float32)
    s(np_A, np_B, np_D)
    np.testing.assert_allclose(np_C, np_D)

    print("pass test_vadd")


def test_variable_bound_for():
    @kernel
    def kernel1(A: int32[10]):
        for i in range(10):
            for j in range(i + 1, 10):
                for k in range(j * 2, 10):
                    A[k] += i - j

    s = process(kernel1)
    np_A = np.zeros((10,), dtype=np.int32)
    kernel1(np_A)
    np_B = np.zeros((10,), dtype=np.int32)
    s(np_B)
    np.testing.assert_allclose(np_A, np_B)

    @kernel
    def kernel2(B: int32[10]):
        for i in range(10):
            for j in range(i, i + 1):
                B[i] += j

    s = process(kernel2)
    np_A = np.zeros((10,), dtype=np.int32)
    kernel2(np_A)
    np_B = np.zeros((10,), dtype=np.int32)
    s(np_B)
    np.testing.assert_allclose(np_A, np_B)

    @kernel
    def kernel3(A: int32[10], B: int32[10]):
        for i in range(5):
            for j in range(i, i + 5):
                A[1 + i - 1] = B[1 + i - 1]

    s = process(kernel3)
    B = np.random.randint(0, 10, (10,), dtype=np.int32)
    np_A = np.zeros((10,), dtype=np.int32)
    kernel3(np_A, B)
    np_B = np.zeros((10,), dtype=np.int32)
    s(np_B, B)
    np.testing.assert_allclose(np_A, np_B)

    print("pass test_variable_bound_for")


def test_scf_for():
    @kernel
    def kernel1(A: int32[10]):
        bound: int32 = 10
        for i in range(bound):
            A[i] = i

    s = process(kernel1)
    np_A = np.zeros((10,), dtype=np.int32)
    kernel1(np_A)
    np_B = np.zeros((10,), dtype=np.int32)
    s(np_B)
    np.testing.assert_allclose(np_A, np_B)

    @kernel
    def kernel2(A: int32[10], B: int32[10]):
        for i in range(10):
            for j in range(A[i], 10, A[i]):
                for k in range(A[i] - 1, A[i] + 2):
                    B[k] += i - j

    s = process(kernel2)
    np_A = np.zeros((10,), dtype=np.int32) + 1
    np_B = np.zeros((10,), dtype=np.int32)
    kernel2(np_A, np_B)
    np_C = np.zeros((10,), dtype=np.int32) + 1
    np_D = np.zeros((10,), dtype=np.int32)
    s(np_C, np_D)
    np.testing.assert_allclose(np_B, np_D)


if __name__ == "__main__":
    test_single_affine_for()
    test_nested_affine_for()
    test_vadd()
    test_variable_bound_for()
    test_scf_for()
