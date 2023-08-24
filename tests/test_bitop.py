# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import allo
from allo.ir.types import uint1, uint2, int32, uint8


def test_scalar():
    def kernel(a: int32) -> int32:
        return a[28:32]

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    assert mod(0xABCD0123) == 0xA


def test_get_bit():
    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] + 1)[0]
        return B

    s = allo.customize(kernel)
    print(s.module)
    np_A = np.random.randint(10, size=(10,))
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), (np_A + 1) & 1, rtol=1e-5, atol=1e-5)


def test_get_bit_slice():
    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] + 1)[0:2]
        return B

    s = allo.customize(kernel)
    print(s.module)
    np_A = np.random.randint(10, size=(10,))
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), (np_A + 1) & 0b11, rtol=1e-5, atol=1e-5)


def test_reverse():
    def kernel(A: uint8[10]) -> uint8[10]:
        B: uint8[10] = 0
        for i in range(10):
            B[i] = (A[i][0:8]).reverse()
        return B

    s = allo.customize(kernel, verbose=True)
    print(s.module)
    np_A = np.random.randint(10, size=(10,)).astype(np.uint8)
    golden = (np_A & 0xFF).astype(np.uint8)
    mod = s.build()
    ret = mod(np_A)
    for i in range(0, 10):
        x = np.unpackbits(golden[i])
        x = np.flip(x)
        y = np.unpackbits(ret[i])
        assert np.array_equal(x, y)


def test_set_bit_tensor():
    def kernel1(A: uint1[10], B: int32[10]):
        for i in range(10):
            b: int32 = B[i]
            b[0] = A[i]
            B[i] = b

    def kernel2(A: uint1[10], B: int32[10]):
        for i in range(10):
            B[i][0] = A[i]

    for kernel in [kernel1, kernel2]:
        s = allo.customize(kernel)
        print(s.module)
        np_A = np.random.randint(2, size=(10,))
        np_B = np.random.randint(10, size=(10,))
        golden = np_B & 0b1110 | np_A
        mod = s.build()
        mod(np_A, np_B)
        assert np.array_equal(golden, np_B)


def test_set_slice():
    def kernel(A: uint2[10], B: int32[10]):
        for i in range(10):
            B[i][0:2] = A[i]

    s = allo.customize(kernel)
    np_A = np.random.randint(2, size=(10,))
    np_B = np.random.randint(7, size=(10,))
    golden = (np_B & 0b1100) | np_A
    mod = s.build()
    mod(np_A, np_B)
    assert np.array_equal(golden, np_B)


if __name__ == "__main__":
    pytest.main([__file__])
