# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
from allo.ir.types import int32, float32, Int, UInt
import numpy as np


def test_same():
    def same(A: int32[32, 32]) -> int32[32, 32]:
        return A

    s = allo.customize(same, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_A = np.zeros((32, 32)).astype(np.int32) + 1
    np_A_allo = mod(np_A)
    np.testing.assert_allclose(np_A, np_A_allo, rtol=1e-5)


def test_same_scalar():
    #  scalars are not transformed into tensors even if enable_tensor=True
    def same_scalar(A: float32) -> float32:
        return A

    s = allo.customize(same_scalar, enable_tensor=True)
    print(s.module)

    mod = s.build()
    assert mod(0.0) == 0.0


def test_outzero():
    def outzero() -> float32[32, 32]:
        C: float32[32, 32] = 0.0
        return C

    s = allo.customize(outzero, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_C = np.zeros((32, 32)).astype(np.float32)
    np_C_allo = mod()
    np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)


def test_outzero_scalar():
    #  scalars are not transformed into tensors even if enable_tensor=True
    def outzero_scalar() -> int32:
        C: int32 = 0
        return C

    s = allo.customize(outzero_scalar, enable_tensor=True)
    print(s.module)

    mod = s.build()
    assert mod() == 0


def test_extract():
    def extract(A: int32[6, 6]) -> int32[1, 2]:
        return A[1:2, 1:3]

    s = allo.customize(extract, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_A = np.random.randint(0, 10, size=(6, 6)).astype(np.int32)
    np.testing.assert_allclose(extract(np_A), mod(np_A), rtol=1e-5)


def test_extract_ele():
    def extract_ele(A: int32[6, 6]) -> int32:
        return A[1, 2]

    s = allo.customize(extract_ele, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_A = np.random.randint(0, 10, size=(6, 6)).astype(np.int32)
    np.testing.assert_allclose(extract_ele(np_A), mod(np_A), rtol=1e-5)


def test_insert():
    def insert(A: int32[3, 4, 5], B: int32[1, 2, 1]) -> int32[3, 4, 5]:
        A[1:2, 1:3, 0:1] = B
        return A

    s = allo.customize(insert, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_A = np.random.randint(0, 10, size=(3, 4, 5)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(1, 2, 1)).astype(np.int32)
    np.testing.assert_allclose(insert(np_A, np_B), mod(np_A, np_B), rtol=1e-5)


def test_insert_ele():
    def insert_ele(A: int32[6, 6], b: int32) -> int32[6, 6]:
        A[1, 2] = b
        return A

    s = allo.customize(insert_ele, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_A = np.random.randint(0, 10, size=(6, 6)).astype(np.int32)
    np_b = np.random.randint(0, 10)
    np.testing.assert_allclose(insert_ele(np_A, np_b), mod(np_A, np_b), rtol=1e-5)


def test_insert_def():
    #  define a scalar inside the function
    def insert_def(A: int32[6, 6]) -> int32[6, 6]:
        B: int32[2, 3] = 0
        A[0:2, 0:3] = B
        return A

    s = allo.customize(insert_def, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_A = np.random.randint(0, 10, size=(6, 6)).astype(np.int32)
    np.testing.assert_allclose(insert_def(np_A), mod(np_A), rtol=1e-5)


def test_insert_ele_def():
    #  define a scalar inside the function
    def insert_ele_def(A: int32[6, 6]) -> int32[6, 6]:
        B: int32 = 0
        A[0, 0] = B
        return A

    s = allo.customize(insert_ele_def, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_A = np.random.randint(0, 10, size=(6, 6)).astype(np.int32)
    np.testing.assert_allclose(insert_ele_def(np_A), mod(np_A), rtol=1e-5)


def test_slice():
    def slice(A: int32[6, 6]) -> int32[6, 6]:
        B: int32[2, 3] = 0
        B[0, 0] = 1
        A[0:2, 0:3] = B
        return A

    s = allo.customize(slice, enable_tensor=True)
    print(s.module)

    np_A = np.random.randint(0, 10, size=(6, 6)).astype(np.int32)
    np_A_slice = np_A.copy()
    np_B = np.zeros((2, 3)).astype(np.int32)
    np_B[0, 0] = 1
    np_A_slice[0:2, 0:3] = np_B
    mod = s.build()
    np.testing.assert_allclose(np_A_slice, mod(np_A), rtol=1e-5)


def test_rank_reducing():
    M, K, N = 6, 3, 6

    def kernel(A: int32[M, N]) -> int32[M, K, N]:
        B: int32[M, K, N] = 2
        B[:M, 0, :N] = A[:M, :N]
        return B

    s = allo.customize(kernel, verbose=True, enable_tensor=True)
    print(s.module)

    np_A = np.random.randint(0, 10, size=(M, N)).astype(np.int32)
    np_B = np.zeros((M, K, N)).astype(np.int32) + 2
    np_B[:M, 0, :N] = np_A
    mod = s.build()
    np.testing.assert_allclose(np_B, mod(np_A), rtol=1e-5)


def test_nested_func_slicing_arg():
    M, N = 6, 6

    def foo(A: int32[3, 3]) -> int32[2, 2]:
        return A[1:3, 1:3]

    def kernel(A: int32[M, N]) -> int32[2, 2]:
        B = foo(A[2:5, 1:4])
        return B

    s = allo.customize(kernel, verbose=True, enable_tensor=True)
    print(s.module)

    np_A = np.random.randint(0, 10, size=(M, N)).astype(np.int32)
    mod = s.build()
    np.testing.assert_allclose(kernel(np_A), mod(np_A), rtol=1e-5)


def test_slicing_broadcast():
    M, N = 6, 6

    np_A = np.zeros((M, N, M, N, M, N)).astype(np.int32)

    def kernel() -> int32[M, N - 1, 2, 2, 1, 1]:
        A: int32[M, N, M, N, M, N] = np_A
        A[:, 1:, :2, 1:3, 0, 2:4:2] = 2
        return A[:, 1:, :2, 1:3, 0, 2:4:2]

    s = allo.customize(kernel, verbose=True, enable_tensor=True)
    print(s.module)

    mod = s.build()
    golden = np.zeros((M, N - 1, 2, 2, 1, 1)).astype(np.int32) + 2
    np.testing.assert_allclose(mod(), golden, rtol=1e-5)


def test_linalg_math_tensor():
    M = 10
    K = 15
    A = np.float32(np.random.uniform(size=(M, K)))
    B = np.float32(np.random.uniform(size=(K, M)))

    def kernel(A: float32[M, K], B: float32[K, M]) -> float32[M, M]:
        D = allo.matmul(A, B)
        C = (allo.add(allo.exp(D), allo.abs(D)) - allo.log(D)) / D
        return C

    s = allo.customize(kernel, enable_tensor=True)
    f = s.build()
    print(s.module)
    outs = f(A, B)
    np_outs = kernel(A, B)
    np.testing.assert_allclose(outs, np_outs, atol=1e-3)


def test_linalg_matmul():
    M = 10
    K = 15
    A = np.random.uniform(size=(M, K))
    B = np.random.uniform(size=(K, M))

    def kernel() -> float32[M, K]:
        A1: float32[M, K] = A
        B1: float32[K, M] = B
        D = allo.matmul(allo.matmul(A1, B1), A1)
        return D

    s = allo.customize(kernel, enable_tensor=True)
    print(s.module)
    f = s.build()
    np.testing.assert_allclose(f(), kernel(), atol=1e-4)


def test_broadcast_int():
    T_IN, T_OUT = Int(4), Int(15)

    def kernel(A: T_IN[16, 16]) -> T_OUT[16, 16]:
        B = A + 3
        C = B - 5
        return C

    for T_IN, T_OUT in [
        (Int(4), Int(15)),
        (Int(5), Int(16)),
        (Int(6), Int(17)),
        (Int(9), Int(31)),
        (Int(8), Int(32)),
        (Int(7), Int(33)),
        (Int(15), Int(34)),
        (Int(16), Int(32)),
        (Int(32), Int(17)),
    ]:
        s = allo.customize(kernel, enable_tensor=True)
        print(s.module)
        np_A = np.random.randint(-8, 8, size=(16, 16)).astype(np.int32)
        f = s.build()
        np.testing.assert_allclose(f(np_A), kernel(np_A), atol=1e-4)
        print(f"Passed {T_IN}, {T_OUT}!")


def test_broadcast_uint():
    T_IN, T_OUT = UInt(3), UInt(7)

    def kernel(A: T_IN[16, 16]) -> T_OUT[16, 16]:
        B = A + 6
        C = B - 4
        return C

    for T_IN, T_OUT in [
        (UInt(3), UInt(7)),
        (UInt(4), UInt(8)),
        (UInt(5), UInt(9)),
        (UInt(7), UInt(16)),
        (UInt(16), UInt(8)),
    ]:
        s = allo.customize(kernel, enable_tensor=True)
        np_A = np.random.randint(0, 8, size=(16, 16)).astype(np.int32)
        f = s.build()
        np.testing.assert_allclose(f(np_A), kernel(np_A), atol=1e-4)
        print(f"Passed {T_IN}, {T_OUT}!")


if __name__ == "__main__":
    pytest.main([__file__])
