# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
from allo.ir.types import int32, float32
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


if __name__ == "__main__":
    pytest.main([__file__])
