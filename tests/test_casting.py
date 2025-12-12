# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np
import allo
from allo.ir.types import (
    int64,
    int32,
    int16,
    uint64,
    uint32,
    uint16,
    uint8,
    float32,
    float64,
)


@pytest.mark.parametrize("flag", [True, False])
def test_casting_int(flag: bool):
    if flag:
        os.environ["USE_LESS_CASTING"] = "1"

    # int32 + int16 -> int32
    def kernel1(A: int32[1], B: int16[1]) -> int32:
        return A[0] + B[0]

    s = allo.customize(kernel1)
    print(s.module)
    if flag:
        module = str(s.module)
        assert "i33" not in module
    mod = s.build()
    a = np.array(40000, dtype=np.int32)
    b = np.array(100, dtype=np.int16)
    assert mod(a, b) == 40000 + 100
    a = np.array(-40000, dtype=np.int32)
    assert mod(a, b) == -40000 + 100

    # uint8 + uint16 -> uint16
    def kernel2(A: uint8[1], B: uint16[1]) -> uint16:
        return A[0] + B[0]

    s = allo.customize(kernel2)
    print(s.module)
    if flag:
        module = str(s.module)
        assert "i17" not in module
    mod = s.build()
    a = np.array(255, dtype=np.uint8)
    b = np.array(1, dtype=np.uint16)
    assert mod(a, b) == 256

    # int16 + uint16 -> uint16
    # casting rule: same rank, unsigned wins
    def kernel3(A: int16[1], B: uint16[1]) -> uint16:
        return A[0] + B[0]

    s = allo.customize(kernel3)
    mod = s.build()
    print(s.module)
    if flag:
        module = str(s.module)
        assert "i18" not in module
    a = np.array(-1, dtype=np.int16)
    b = np.array(1, dtype=np.uint16)
    # casting rule: -1 casted to uint16 = 65535
    assert mod(a, b) == 65536 % (1 << 16)

    # int32 + uint16 -> int32
    # casting rule: signed can represent all unsigned -> stay signed
    def kernel4(A: int32[1], B: uint16[1]) -> int32:
        return A[0] + B[0]

    s = allo.customize(kernel4)
    mod = s.build()
    print(s.module)
    if flag:
        module = str(s.module)
        assert "i33" not in module
    a = np.array(-40000, dtype=np.int32)
    b = np.array(100, dtype=np.uint16)
    assert mod(a, b) == -39900

    # int16 + uint32 -> uint32
    # casting rule: unsigned has higher rank -> convert signed to unsigned
    def kernel5(A: int16[1], B: uint32[1]) -> uint32:
        return A[0] + B[0]

    s = allo.customize(kernel5)
    mod = s.build()
    print(s.module)
    if flag:
        module = str(s.module)
        assert "i34" not in module
    a = np.array(-1, dtype=np.int16)
    b = np.array(2, dtype=np.uint32)
    # -1 casted to uint32 = 0xFFFFFFFF
    assert mod(a, b) == (0xFFFFFFFF + 2) % (1 << 32)

    # int32 + uint64 -> uint64
    def kernel6(A: int32[1], B: uint64[1]) -> uint64:
        return A[0] + B[0]

    s = allo.customize(kernel6)
    mod = s.build()
    print(s.module)
    if flag:
        module = str(s.module)
        assert "i66" not in module
    a = np.array(-1, dtype=np.int32)
    b = np.array(1, dtype=np.uint64)
    assert mod(a, b) == 0

    if flag:
        del os.environ["USE_LESS_CASTING"]


@pytest.mark.parametrize("flag", [True, False])
def test_casting_int_float(flag: bool):

    if flag:
        os.environ["USE_LESS_CASTING"] = "1"

    # float32 + int16 -> float32
    # casting rule: integer is converted to float32
    def kernel1(A: float32[1], B: int16[1]) -> float32:
        return A[0] + B[0]

    s = allo.customize(kernel1)
    mod = s.build()
    print(s.module)
    a = np.array(1.5, dtype=np.float32)
    b = np.array(2, dtype=np.int16)
    np.testing.assert_allclose(mod(a, b), 3.5, rtol=1e-5, atol=1e-5)

    # int32 + float32 -> float32
    # casting rule: integer promoted to float
    def kernel2(A: int32[1], B: float32[1]) -> float32:
        return A[0] + B[0]

    s = allo.customize(kernel2)
    mod = s.build()
    print(s.module)
    a = np.array(-2, dtype=np.int32)
    b = np.array(0.5, dtype=np.float32)
    np.testing.assert_allclose(mod(a, b), -1.5, rtol=1e-5, atol=1e-5)

    # int32 + float64 -> float64
    def kernel3(A: int32[1], B: float64[1]) -> float64:
        return A[0] + B[0]

    s = allo.customize(kernel3)
    mod = s.build()
    print(s.module)
    a = np.array(1000000, dtype=np.int32)
    b = np.array(0.25, dtype=np.float64)
    np.testing.assert_allclose(mod(a, b), 1000000.25, rtol=1e-5, atol=1e-5)

    if flag:
        del os.environ["USE_LESS_CASTING"]


@pytest.mark.parametrize("flag", [True, False])
def test_casting_index(flag: bool):
    if flag:
        os.environ["USE_LESS_CASTING"] = "1"

    def kernel1(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = A[i] + i
        return B

    s = allo.customize(kernel1)
    print(s.module)
    if flag:
        module = str(s.module)
        assert "i34" not in module
    np_A = np.random.randint(10, size=(10,)).astype(np.int32)
    mod = s.build()
    np_B = np_A + np.arange(10, dtype=np.int32)
    np.testing.assert_allclose(mod(np_A), np_B, rtol=1e-5, atol=1e-5)

    # int16 + index -> int32
    def kernel2(A: int16[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = A[i] + i
        return B

    s = allo.customize(kernel2)
    print(s.module)
    if flag:
        module = str(s.module)
        assert "i34" not in module
    mod = s.build()
    np_A = np.random.randint(10, size=(10,)).astype(np.int16)
    np_B = np_A.astype(np.int32) + np.arange(10, dtype=np.int32)
    np.testing.assert_allclose(mod(np_A), np_B, rtol=1e-5, atol=1e-5)

    def kernel3(A: int64[10]) -> int64[10]:
        B: int64[10] = 0
        for i in range(10):
            B[i] = A[i] + i
        return B

    s = allo.customize(kernel3)
    print(s.module)
    if flag:
        module = str(s.module)
        assert "i65" not in module
    mod = s.build()
    np_A = np.random.randint(2294967296, 4294967296, size=(10,)).astype(np.int64)
    np_B = np_A + np.arange(10, dtype=np.int64)
    np.testing.assert_allclose(mod(np_A), np_B, rtol=1e-5, atol=1e-5)

    def kernel4(A: float32[10]) -> float32[10]:
        B: float32[10] = 0
        for i in range(10):
            B[i] = A[i] * i
        return B

    s = allo.customize(kernel4)
    print(s.module)
    mod = s.build()
    np_A = np.random.rand(10).astype(np.float32)
    np_B = np_A * np.arange(10, dtype=np.float32)
    np.testing.assert_allclose(mod(np_A), np_B, rtol=1e-5, atol=1e-5)

    if flag:
        del os.environ["USE_LESS_CASTING"]


@pytest.mark.parametrize("flag", [True, False])
def test_get_bit(flag: bool):
    if flag:
        os.environ["USE_LESS_CASTING"] = "1"

    def kernel_add(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] + 1)[0]
        return B

    s = allo.customize(kernel_add)
    print(s.module)
    np_A = np.random.randint(10, size=(10,))
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), (np_A + 1) & 1, rtol=1e-5, atol=1e-5)

    def kernel_mul(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] * 2)[0]
        return B

    s = allo.customize(kernel_mul)
    print(s.module)
    np_A = np.random.randint(10, size=(10,))
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), (np_A * 2) & 1, rtol=1e-5, atol=1e-5)

    def kernel_div(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] // 2)[0]
        return B

    s = allo.customize(kernel_div)
    print(s.module)
    np_A = np.random.randint(10, size=(10,))
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), (np_A // 2) & 1, rtol=1e-5, atol=1e-5)

    if flag:
        del os.environ["USE_LESS_CASTING"]


@pytest.mark.parametrize("flag", [True, False])
def test_get_bit_slice(flag: bool):
    if flag:
        os.environ["USE_LESS_CASTING"] = "1"

    def kernel_add(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] + 1)[0:2]
        return B

    s = allo.customize(kernel_add)
    print(s.module)
    np_A = np.random.randint(10, size=(10,))
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), (np_A + 1) & 0b11, rtol=1e-5, atol=1e-5)

    def kernel_mul(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] * 2)[0:2]
        return B

    s = allo.customize(kernel_mul)
    print(s.module)
    np_A = np.random.randint(10, size=(10,))
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), (np_A * 2) & 0b11, rtol=1e-5, atol=1e-5)

    if flag:
        del os.environ["USE_LESS_CASTING"]


@pytest.mark.parametrize("flag", [True, False])
def test_dynamic_index(flag: bool):
    def kernel(A: int32, B: int32[11]):
        for i in range(1, 12):
            B[i - 1] = A[i - 1]

    if flag:
        os.environ["USE_LESS_CASTING"] = "1"
    s = allo.customize(kernel)
    np_B = np.zeros((11,), dtype=np.int32)
    mod = s.build()
    mod(1234, np_B)
    assert bin(1234) == "0b" + "".join([str(np_B[i]) for i in range(10, -1, -1)])
    if flag:
        del os.environ["USE_LESS_CASTING"]


@pytest.mark.parametrize("flag", [True, False])
def test_dynamic_slice(flag: bool):
    def kernel(A: int32, B: int32[11]):
        for i in range(1, 12):
            B[i - 1] = A[i - 1 : i]

    if flag:
        os.environ["USE_LESS_CASTING"] = "1"
    s = allo.customize(kernel)
    np_B = np.zeros((11,), dtype=np.int32)
    mod = s.build()
    mod(1234, np_B)
    assert bin(1234) == "0b" + "".join([str(np_B[i]) for i in range(10, -1, -1)])
    if flag:
        del os.environ["USE_LESS_CASTING"]


if __name__ == "__main__":
    test_casting_int(True)
    test_casting_int(False)
    test_casting_int_float(True)
    test_casting_int_float(False)
    test_casting_index(True)
    test_casting_index(False)
    test_get_bit(True)
    test_get_bit(False)
    test_get_bit_slice(True)
    test_get_bit_slice(False)
    test_dynamic_index(True)
    test_dynamic_index(False)
    test_dynamic_slice(True)
    test_dynamic_slice(False)
