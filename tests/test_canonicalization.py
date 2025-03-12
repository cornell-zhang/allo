# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int32
from allo.autoscheduler.passes import dataflow_optimization_pass
from allo.autoscheduler.util import check_preprocess_ok


def test_single_producer_single_consumer():
    def producer() -> int32[10]:
        A: int32[10]
        for i in range(10):
            A[i] = i
        return A

    def consumer(A: int32[10]) -> int32[10]:
        B: int32[10]
        for i in range(10):
            B[i] = A[i] + 1
        return B

    def top() -> int32[10]:
        A = producer()
        return consumer(A)

    p = allo.customize(producer)
    c = allo.customize(consumer)

    s = allo.customize(top)
    s.compose([p, c])
    s = dataflow_optimization_pass(s, debugPoint="dataflow_canonicaliation")
    print(s.module)
    mod = s.build()
    res = mod()
    np.testing.assert_array_equal(res, np.arange(1, 11))
    assert check_preprocess_ok(s)


def test_single_producer_multiple_consumers():
    def producer() -> int32[10]:
        A: int32[10]
        for i in range(10):
            A[i] = i
        return A

    def consumer1(A: int32[10]) -> int32[10]:
        B: int32[10]
        for i in range(10):
            B[i] = A[i] - 1
        return B

    def consumer2(A: int32[10]) -> int32[10]:
        B: int32[10]
        for i in range(10):
            B[i] = A[i] + 1
        return B

    def top() -> int32[10]:
        A = producer()
        res1 = consumer1(A)
        res2 = consumer2(A)
        B: int32[10]
        for i in range(10):
            B[i] = res1[i] + res2[i]
        return B

    p = allo.customize(producer)
    c1 = allo.customize(consumer1)
    c2 = allo.customize(consumer2)

    s = allo.customize(top)
    print(s.module)
    s.compose([p, c1, c2])
    s = dataflow_optimization_pass(s, debugPoint="dataflow_canonicaliation")
    print(s.module)
    mod = s.build()
    res = mod()
    np.testing.assert_array_equal(res, 2 * np.arange(10))
    assert check_preprocess_ok(s)


def test_single_kernel():
    def producer() -> (int32[10], int32[10]):
        A: int32[10]
        for i in range(10):
            A[i] = i

        B: int32[10]
        C: int32[10]
        for i in range(10):
            B[i] = A[i] + 1
            C[i] = A[i] + 1
        return B, C

    s = allo.customize(producer)
    s = dataflow_optimization_pass(s, debugPoint="dataflow_canonicaliation")
    print(s.module)
    mod = s.build()
    res1, res2 = mod()
    np.testing.assert_array_equal(res1, np.arange(1, 11))
    np.testing.assert_array_equal(res2, np.arange(1, 11))
    assert check_preprocess_ok(s)


def test_nd_array():
    def producer() -> int32[10, 10]:
        A: int32[10, 10] = 0
        return A

    def consumer(A: int32[10, 10], B: int32[10, 10]) -> int32[10, 10]:
        sum: int32[10, 10]
        for i in range(10):
            for j in range(10):
                sum[i, j] = A[i, j] + B[i, j]
        return sum

    def top() -> int32[10, 10]:
        A = producer()
        return consumer(A, A)

    p = allo.customize(producer)
    c = allo.customize(consumer)

    s = allo.customize(top)
    s.compose([p, c])
    print(s.module)

    s = dataflow_optimization_pass(s, debugPoint="dataflow_canonicaliation")
    print(s.module)
    mod = s.build()
    res = mod()
    np.testing.assert_array_equal(res, 0)
    assert check_preprocess_ok(s)


def test_matmul_addition_condition1():
    """Checks that the matmul reduction loop is transformed correctly as described in the Stream-HLS paper (https://arxiv.org/pdf/2501.09118)."""

    def matmul_addition() -> int32[8, 8]:
        A: int32[8, 8] = 1
        B: int32[8, 8] = 2
        C: int32[8, 8] = 0
        D: int32[8, 8] = 3

        for i in range(8):
            for j in range(8):
                for k in range(8):
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        E: int32[8, 8]
        for i in range(8):
            for j in range(8):
                E[i, j] = C[i, j] + D[i, j]

        return E

    s = allo.customize(matmul_addition)
    print(s.module)
    s = dataflow_optimization_pass(s, debugPoint="dataflow_canonicaliation")
    print(s.module)

    mod = s.build()
    res = mod()

    expected = (
        np.full((8, 8), 1, dtype=np.int32) @ np.full((8, 8), 2, dtype=np.int32) + 3
    )

    np.testing.assert_allclose(res, expected, rtol=1e-5)
    assert check_preprocess_ok(s)


def matrix_multiply(A: int32[8, 8], B: int32[8, 8]) -> int32[8, 8]:
    C: int32[8, 8] = 0
    for i in range(8):
        for j in range(8):
            for k in range(8):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
    return C


def matrix_add(C: int32[8, 8], D: int32[8, 8]) -> int32[8, 8]:
    E: int32[8, 8]
    for i in range(8):
        for j in range(8):
            E[i, j] = C[i, j] + D[i, j]
    return E


def test_matmul_addition_nested_condition1():
    def top() -> int32[8, 8]:
        A: int32[8, 8] = 1
        B: int32[8, 8] = 2
        D: int32[8, 8] = 3

        C = matrix_multiply(A, B)
        E1 = matrix_add(C, D)

        return E1

    mm = allo.customize(matrix_multiply)
    ma = allo.customize(matrix_add)
    s = allo.customize(top)

    s.compose([mm, ma])

    print(s.module)
    s = dataflow_optimization_pass(s, debugPoint="dataflow_canonicaliation")
    print(s.module)

    mod = s.build()
    res = mod()
    expected = np.full((8, 8), 19, dtype=np.int32)
    np.testing.assert_allclose(res, expected, rtol=1e-5)
    assert check_preprocess_ok(s)


def test_nested_fn_inlining():
    def make_matrix(cst: int32) -> int32[8, 8]:
        A: int32[8, 8]
        for i in range(8):
            for j in range(8):
                A[i, j] = cst
        return A

    def add_constant(A: int32[8, 8], B: int32[8, 8], cst: int32) -> int32[8, 8]:
        const_mat = make_matrix(cst)
        return matrix_add(A, matrix_add(B, const_mat))

    def top() -> int32[8, 8]:
        A: int32[8, 8] = 1
        return add_constant(A, A, 2)

    ma = allo.customize(matrix_add)
    ac = allo.customize(add_constant)
    s = allo.customize(top)
    s.compose([ma, ac])

    print(s.module)
    s = dataflow_optimization_pass(s, debugPoint="dataflow_canonicaliation")
    print(s.module)

    mod = s.build()
    res = mod()
    expected = np.full((8, 8), 4, dtype=np.int32)
    np.testing.assert_array_equal(res, expected)
    assert check_preprocess_ok(s)


if __name__ == "__main__":
    pytest.main([__file__])
