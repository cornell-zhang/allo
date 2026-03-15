# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from src.main import process
from allo.ir.types import (
    int16,
    int32,
    uint16,
    uint32,
    float16,
    float32,
    float64,
    bfloat16,
    Fixed,
    index,
    UFixed,
)
from allo.spmw import kernel


def test_int_index_cast():
    @kernel
    def kernel1() -> int32:
        b: index = 1
        a: int32 = b
        return a

    s = process(kernel1)
    assert s() == 1

    @kernel
    def kernel2(A: int16) -> int16:
        b: index = A
        a: int16 = b
        return a

    s = process(kernel2)
    assert s(1) == 1
    assert s(1000) == 1000
    assert s(-1) == -1
    assert s(-1000) == -1000
    assert s(32767) == 32767
    assert s(-32768) == -32768

    @kernel
    def kernel3() -> uint16:
        b: index = 1
        a: uint16 = b
        return a

    s = process(kernel3)
    assert s() == 1

    @kernel
    def kernel4(A: int16) -> uint16:
        b: index = A
        a: uint16 = b
        return a

    s = process(kernel4)
    assert s(1) == 1
    assert s(1000) == 1000
    assert s(32767) == 32767

    print("test_int_index_cast passed")


def test_int_float_cast():
    @kernel
    def kernel1(a: int16) -> float32:
        b: float32 = a
        return b

    s = process(kernel1)
    assert s(1) == 1.0
    assert s(1000) == 1000.0
    assert s(-1) == -1.0
    assert s(-1000) == -1000.0
    assert s(32767) == 32767.0
    assert s(-32768) == -32768.0

    @kernel
    def kernel2(a: int32) -> float32:
        b: float32 = a
        return b

    s = process(kernel2)
    assert s(1) == 1.0
    assert s(1000) == 1000.0
    assert s(-1) == -1.0
    assert s(-1000) == -1000.0
    assert s(32767) == 32767.0
    assert s(-32768) == -32768.0

    @kernel
    def kernel3(a: float32) -> int32:
        b: int32 = a
        return b

    s = process(kernel3)
    assert s(1.0) == 1
    assert s(1000.0) == 1000
    assert s(-1.0) == -1
    assert s(-1000.0) == -1000
    assert s(32767.0) == 32767
    assert s(-32768.0) == -32768

    @kernel
    def kernel4(a: float32) -> int16:
        b: int16 = a
        return b

    s = process(kernel4)
    assert s(1.0) == 1
    assert s(1000.0) == 1000
    assert s(-1.0) == -1
    assert s(-1000.0) == -1000
    assert s(32767.0) == 32767
    assert s(-32768.0) == -32768

    @kernel
    def kernel5(a: float32) -> uint16:
        b: uint16 = a
        return b

    s = process(kernel5)
    assert s(1.0) == 1
    assert s(1000.0) == 1000
    assert s(32767.0) == 32767

    print("test_int_float_cast passed")


def test_fixed_cast():
    # [NOTE]: fixed point not supported for llvm backend

    @kernel
    def kernel1(a: float32) -> int32:
        b: Fixed(12, 4) = a
        c: int32 = b
        return c

    s = process(kernel1)

    @kernel
    def kernel2(a: float32) -> int32:
        b: Fixed(20, 12) = a
        c: int32 = b
        return c

    s = process(kernel2)

    @kernel
    def kernel3(a: float32) -> int32:
        b: UFixed(12, 4) = a
        c: int32 = b
        return c

    s = process(kernel3)

    @kernel
    def kernel4(a: float32) -> int32:
        b: UFixed(20, 12) = a
        c: int32 = b
        return c

    s = process(kernel4)

    @kernel
    def kernel5(a: int32) -> float32:
        b: Fixed(12, 4) = a
        c: float32 = b
        return c

    s = process(kernel5)

    @kernel
    def kernel6(a: int32) -> float32:
        b: UFixed(20, 12) = a
        c: float32 = b
        return c

    s = process(kernel6)

    @kernel
    def kernel7(a: int32) -> int16:
        b: Fixed(12, 4) = a
        c: int16 = b
        return c

    s = process(kernel7)

    @kernel
    def kernel8(a: int32) -> int32:
        b: UFixed(20, 12) = a
        c: int32 = b
        return c

    s = process(kernel8)

    @kernel
    def kernel9(a: int16) -> uint16:
        b: Fixed(12, 4) = a
        c: uint16 = b
        return c

    s = process(kernel9)

    @kernel
    def kernel10(a: int16) -> uint16:
        b: UFixed(20, 12) = a
        c: uint16 = b
        return c

    s = process(kernel10)

    @kernel
    def kernel11(a: int16) -> uint16:
        b: Fixed(12, 4) = a
        b_: Fixed(20, 12) = b
        c: uint16 = b_
        return c

    s = process(kernel11)

    @kernel
    def kernel12(a: int16) -> uint16:
        b: UFixed(20, 12) = a
        b_: UFixed(12, 4) = b
        c: uint16 = b_
        return c

    s = process(kernel12)

    @kernel
    def kernel13(a: int16) -> uint16:
        b: Fixed(12, 4) = a
        b_: UFixed(12, 4) = b
        c: uint16 = b_
        return c

    s = process(kernel13)

    @kernel
    def kernel14(a: int16) -> uint16:
        b: UFixed(20, 12) = a
        b_: Fixed(20, 12) = b
        c: uint16 = b_
        return c

    s = process(kernel14)

    print("test_fixed_cast passed")


def test_int_cast():
    # [NOTE]: llvm ir use signless int, the result may not be expected

    @kernel
    def kernel1(a: int16) -> int32:
        b: int32 = a
        return b

    s = process(kernel1)
    assert s(1) == 1
    assert s(1000) == 1000
    assert s(-1) == -1
    assert s(-1000) == -1000
    assert s(32767) == 32767
    assert s(-32768) == -32768

    @kernel
    def kernel2(a: uint16) -> int32:
        b: int32 = a
        return b

    s = process(kernel2)
    assert s(1) == 1
    assert s(1000) == 1000
    assert s(32767) == 32767

    @kernel
    def kernel3(a: int32) -> int16:
        b: int16 = a
        return b

    s = process(kernel3)
    assert s(1) == 1
    assert s(1000) == 1000
    assert s(-1) == -1
    assert s(-1000) == -1000
    assert s(32767) == 32767
    assert s(-32768) == -32768
    assert s(32768) == -32768
    assert s(65535) == -1

    @kernel
    def kernel4(a: int32) -> uint16:
        b: uint16 = a
        return b

    s = process(kernel4)
    assert s(1) == 1
    assert s(1000) == 1000
    assert s(32767) == 32767

    @kernel
    def kernel5(a: uint16) -> int16:
        b: int16 = a
        return b

    s = process(kernel5)
    assert s(1) == 1
    assert s(1000) == 1000
    assert s(32767) == 32767

    @kernel
    def kernel6(a: int16) -> uint32:
        b: uint32 = a
        return b

    s = process(kernel6)
    assert s(1) == 1
    assert s(1000) == 1000
    assert s(32767) == 32767

    @kernel
    def kernel7(a: uint32) -> int16:
        b: int16 = a
        return b

    s = process(kernel7)
    assert s(1) == 1
    assert s(1000) == 1000
    assert s(32767) == 32767

    print("test_int_cast passed")


def test_float_cast():
    @kernel
    def kernel1(a: float32) -> float16:
        b: float16 = a
        return b

    s = process(kernel1)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel2(a: float32) -> float64:
        b: float64 = a
        return b

    s = process(kernel2)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel3(a: float32) -> bfloat16:
        b: bfloat16 = a
        return b

    s = process(kernel3)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel4(a: float16) -> float32:
        b: float32 = a
        return b

    s = process(kernel4)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel5(a: float16) -> float64:
        b: float64 = a
        return b

    s = process(kernel5)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    # [NOTE]: bfloat16 -> float16 not supported
    # def kernel6(a: float16) -> bfloat16:
    #     b: bfloat16 = a
    #     return b

    # s = process(kernel6)
    # assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    # assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    # assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    # assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel7(a: float64) -> float16:
        b: float16 = a
        return b

    s = process(kernel7)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel8(a: float64) -> float32:
        b: float32 = a
        return b

    s = process(kernel8)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel9(a: float64) -> bfloat16:
        b: bfloat16 = a
        return b

    s = process(kernel9)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    # [NOTE]: bfloat16 -> float16 not supported
    # def kernel10(a: bfloat16) -> float16:
    #     b: float16 = a
    #     return b

    # s = process(kernel10)
    # assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    # assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    # assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    # assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel11(a: bfloat16) -> float32:
        b: float32 = a
        return b

    s = process(kernel11)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel12(a: bfloat16) -> float64:
        b: float64 = a
        return b

    s = process(kernel12)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000.0), 1000.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1.0), -1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(-1000.0), -1000.0, rel_tol=1e-3, abs_tol=1e-3)

    print("test_float_cast passed")


def test_index_float_cast():
    @kernel
    def kernel1(a: index) -> float32:
        b: float32 = a
        return b

    s = process(kernel1)
    assert math.isclose(s(1), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000), 1000.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel2(a: index) -> float64:
        b: float64 = a
        return b

    s = process(kernel2)
    assert math.isclose(s(1), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(1000), 1000.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel3(a: index) -> bfloat16:
        b: bfloat16 = a
        return b

    s = process(kernel3)
    assert math.isclose(s(1), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(10), 10.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel4(a: float16) -> float32:
        b: index = a
        c: float32 = b
        return c

    s = process(kernel4)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(10.0), 10.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel5(a: float16) -> float64:
        b: index = a
        c: float64 = b
        return c

    s = process(kernel5)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(10.0), 10.0, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel6(a: float16) -> bfloat16:
        b: index = a
        c: bfloat16 = b
        return c

    s = process(kernel6)
    assert math.isclose(s(1.0), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(s(10.0), 10.0, rel_tol=1e-3, abs_tol=1e-3)

    print("test_index_float_cast passed")


def test_index_fixed_cast():
    # [NOTE]: fixed point not supported for llvm backend
    @kernel
    def kernel1(a: index) -> int32:
        b: Fixed(16, 16) = a
        c: int32 = b
        return c

    s = process(kernel1)

    @kernel
    def kernel2(a: index) -> int16:
        b: UFixed(16, 16) = a
        c: int16 = b
        return c

    s = process(kernel2)

    @kernel
    def kernel3(a: index) -> index:
        b: Fixed(20, 12) = a
        c: index = b
        return c

    s = process(kernel3)

    print("test_index_fixed_cast passed")


if __name__ == "__main__":
    test_int_index_cast()
    test_int_float_cast()
    test_fixed_cast()
    test_int_cast()
    test_float_cast()
    test_index_float_cast()
    test_index_fixed_cast()
