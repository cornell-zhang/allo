# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process
from allo.ir.types import int32, ConstExpr, index
import numpy as np
from allo.spmw import kernel


def test_element_access():
    @kernel
    def kernel1() -> int32[2]:
        b: int32[2]
        idx: index = 0
        b[idx] = 1
        b[1] = 0
        return b

    s = process(kernel1)
    assert np.array_equal(s(), np.array([1, 0], dtype=np.int32))

    @kernel
    def kernel2() -> int32[2]:
        b: int32[2]
        b[0] = 1
        b[1] = 0
        return b

    s = process(kernel2)
    assert np.array_equal(s(), np.array([1, 0], dtype=np.int32))

    @kernel
    def kernel3() -> int32[2]:
        b: int32[2]
        idx: index = 0
        b[idx] = 1
        b[idx + 1] = 0
        return b

    s = process(kernel3)
    assert np.array_equal(s(), np.array([1, 0], dtype=np.int32))

    @kernel
    def kernel4() -> int32[2, 2]:
        b: int32[2, 2]
        idx: index = 0
        b[idx, idx] = 1
        b[idx, idx + 1] = 2
        b[idx + 1, idx] = 3
        b[idx + 1, idx + 1] = 4
        return b

    s = process(kernel4)
    assert np.array_equal(s(), np.array([[1, 2], [3, 4]], dtype=np.int32))

    @kernel
    def kernel5(a: int32[2, 2]) -> int32[2, 2]:
        b: int32[2, 2]
        idx: index = 0
        b[idx, idx] = a[idx, idx] + 1
        b[idx, idx + 1] = a[idx, idx + 1] + 1
        b[idx + 1, idx] = a[idx + 1, idx] + 1
        b[idx + 1, idx + 1] = a[idx + 1, idx + 1] + 1
        return b

    s = process(kernel5)
    np_A = np.random.randint(0, 10, (2, 2), dtype=np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A + 1)

    print("test_element_access passed")


def test_slice():
    @kernel
    def kernel1() -> int32[2, 4]:
        b: int32[2, 4]
        idx: index = 0
        b[idx] = 1
        b[1] = 0
        return b

    s = process(kernel1)
    assert np.array_equal(s(), np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int32))

    print("test_slice passed")

    @kernel
    def kernel2(a: int32[2, 4]) -> int32[2, 4]:
        b: int32[2, 4]
        idx: index = 0
        b[idx] = a[idx] + 1
        b[1] = a[1] + 1
        return b

    s = process(kernel2)
    np_A = np.random.randint(0, 10, (2, 4), dtype=np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A + 1)

    @kernel
    def kernel3(a: int32[2, 2, 4]) -> int32[2, 2, 4]:
        b: int32[2, 2, 4]
        idx: index = 0
        b[idx] = a[idx] + 1
        b[idx + 1] = a[idx + 1] + 1
        return b

    s = process(kernel3)
    np_A = np.random.randint(0, 10, (2, 2, 4), dtype=np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A + 1)

    @kernel
    def kernel4(a: int32[2, 2, 4]) -> int32[2, 2, 4]:
        b: int32[2, 2, 4]
        idx: index = 0
        b[idx, :] = a[idx, :, :] + 1
        b[idx + 1, :, :] = a[idx + 1] + 1
        return b

    s = process(kernel4)
    np_A = np.random.randint(0, 10, (2, 2, 4), dtype=np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A + 1)

    print("test_slice passed")


if __name__ == "__main__":
    test_element_access()
    test_slice()
