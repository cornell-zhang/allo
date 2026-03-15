# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
from allo.ir.types import int16, int32, float32
from allo.spmw import kernel


def test_multi_return():
    @kernel
    def kernel1(A: int32, B: int32) -> (int32, int32):
        res0: int32 = 0
        res1: int32 = 0
        return res0, res1

    s = process(kernel1)
    # [NOTE]: for llvm backend, When returning multiple values, we only support all tensors.

    @kernel
    def kernel2(A: int32[8], B: int32[8]) -> (int32[8], int32[8]):
        res0: int32[8] = 0
        res1: int32[8] = 0
        for i in range(8):
            res0[i] = A[i] + 1
            res1[i] = B[i] + 1
        return res0, res1

    s = process(kernel2)
    np_A = np.random.randint(0, 10, size=(8,)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(8,)).astype(np.int32)
    np_res0, np_res1 = s(np_A, np_B)
    assert np.array_equal(np_res0, np_A + 1)
    assert np.array_equal(np_res1, np_B + 1)

    print("test_multi_return passed!")


def test_call_multi_return():
    @kernel
    def helper(A: int32, B: int32) -> (int32, int32):
        return A + 1, B + 1

    @kernel
    def kernel1(A: int32, B: int32) -> (int32, int32):
        res0: int32 = 0
        res1: int32 = 0
        res0, res1 = helper(A, B)
        return res0, res1

    s = process(kernel1)
    # [NOTE]: for llvm backend, When returning multiple values, we only support all tensors.

    @kernel
    def kernel2(A: int32, B: int32) -> (int32[8], int32[8]):
        res0: int32[8] = 0
        res1: int32[8] = 0
        res0, res1 = helper(A, B)
        return res0, res1

    s = process(kernel2)
    np_res0, np_res1 = s(1, 2)
    assert np.array_equal(np_res0, np.ones(8).astype(np.int32) + 1)
    assert np.array_equal(np_res1, np.ones(8).astype(np.int32) + 2)

    @kernel
    def kernel3(A: int16, B: int16) -> (int32[8], int32[8]):
        res0: int32[8] = 0
        res1: int32[8] = 0
        res0, res1 = helper(A, B)
        return res0, res1

    s = process(kernel3)
    np_res0, np_res1 = s(-1, 2)
    assert np.array_equal(np_res0, np.ones(8).astype(np.int32) - 1)
    assert np.array_equal(np_res1, np.ones(8).astype(np.int32) + 2)

    @kernel
    def kernel4(A: int16, B: int16) -> (int16[8], int16[8]):
        res0: int16[8] = 0
        res1: int16[8] = 0
        res0, res1 = helper(A, B)
        return res0, res1

    s = process(kernel4)
    np_res0, np_res1 = s(-1, 2)
    assert np.array_equal(np_res0, np.ones(8).astype(np.int16) - 1)
    assert np.array_equal(np_res1, np.ones(8).astype(np.int16) + 2)

    @kernel
    def callee(a: float32, b: float32) -> (float32, float32):
        c: float32 = a + b
        d: float32 = a - b
        return c, d

    @kernel
    def kernel5(A: float32[10], B: float32[10]) -> (float32[10], float32[10]):
        C: float32[10] = 0
        D: float32[10] = 0
        for i in range(10):
            C[i], D[i] = callee(A[i], B[i])
        return C, D

    s = process(kernel5)
    np_A = np.random.random((10,)).astype(np.float32)
    np_B = np.random.random((10,)).astype(np.float32)
    np_C, np_D = s(np_A, np_B)
    np_C_ref = np.zeros((10,), dtype=np.float32)
    np_D_ref = np.zeros((10,), dtype=np.float32)
    for i in range(10):
        np_C_ref[i], np_D_ref[i] = callee(np_A[i], np_B[i])
    np.testing.assert_allclose(np_C, np_C_ref)
    np.testing.assert_allclose(np_D, np_D_ref)

    print("test_call_multi_return passed!")


if __name__ == "__main__":
    test_multi_return()
    test_call_multi_return()
