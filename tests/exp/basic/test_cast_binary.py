# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from src.main import process
from allo.ir.types import int16, int32, bool, float16, float32
from allo.spmw import kernel


def test_cast_binary_arith():
    @kernel
    def kernel1(a: int16, b: int16) -> int32:
        c: int32 = a + b
        return c

    s = process(kernel1)
    assert s(1, 2) == 3
    assert s(1000, 2000) == 3000
    assert s(-1, -2) == -3
    assert s(-1000, -2000) == -3000

    @kernel
    def kernel2(a: int16, b: int32) -> int32:
        c: int32 = a - b
        return c

    s = process(kernel2)
    assert s(1, 2) == -1
    assert s(1000, 2000) == -1000
    assert s(-1, -2) == 1
    assert s(-1000, -2000) == 1000
    assert s(32767, 1) == 32766
    assert s(-32768, -1) == -32767

    @kernel
    def kernel3(a: int16, b: int32) -> int32:
        c: int32 = a * b
        return c

    s = process(kernel3)
    assert s(1, 2) == 2
    assert s(1000, 2000) == 2000000
    assert s(-1, -2) == 2
    assert s(-1000, -2000) == 2000000

    @kernel
    def kernel4(a: int16, b: int32) -> int32:
        c: int32 = a / b
        return c

    s = process(kernel4)
    assert s(1, 2) == 0
    assert s(2000, 1000) == 2
    assert s(-1, -2) == 0
    assert s(-2000, -1000) == 2

    @kernel
    def kernel5(a: float16, b: float32) -> float32:
        c: float32 = a + b
        return c

    s = process(kernel5)
    assert math.isclose(s(1.0, 2.0), 3.0)
    assert math.isclose(s(1000.0, 2000.0), 3000.0)
    assert math.isclose(s(-1.0, -2.0), -3.0)
    assert math.isclose(s(-1000.0, -2000.0), -3000.0)

    @kernel
    def kernel6(a: float16, b: float32) -> float32:
        c: float32 = a - b
        return c

    s = process(kernel6)
    assert math.isclose(s(1.0, 2.0), -1.0)
    assert math.isclose(s(1000.0, 2000.0), -1000.0)
    assert math.isclose(s(-1.0, -2.0), 1.0)
    assert math.isclose(s(-1000.0, -2000.0), 1000.0)

    @kernel
    def kernel7(a: int16, b: float32) -> float32:
        c: float32 = a * b
        return c + 1.0

    s = process(kernel7)
    assert math.isclose(s(1, 2.0), 3.0)
    assert math.isclose(s(1000, 2000.0), 2000001.0)
    assert math.isclose(s(-1, -2.0), 3.0)
    assert math.isclose(s(-1000, -2000.0), 2000001.0)

    print("test_cast_binary_arith passed")


def test_cast_binary_compare():
    @kernel
    def kernel1(a: int16, b: int32) -> bool:
        c: bool = a < b
        return c

    s = process(kernel1)
    assert s(1, 2) == True
    assert s(1000, 2000) == True
    assert s(-1, -2) == False
    assert s(-1000, -2000) == False

    print("test_cast_binary_compare passed")

    @kernel
    def kernel2(a: int16, b: float32) -> bool:
        c: bool = a > b
        return c

    s = process(kernel2)
    assert s(1, 2.0) == False
    assert s(1000, 2000.0) == False
    assert s(-1, -2.0) == True
    assert s(-1000, -2000.0) == True

    @kernel
    def kernel3(a: float16, b: int32) -> bool:
        c: bool = a <= b
        return c

    s = process(kernel3)
    assert s(1.0, 1) == True
    assert s(1.0, 2) == True
    assert s(1000.0, 2000) == True
    assert s(-1.0, -2) == False
    assert s(-1000.0, -2000) == False

    @kernel
    def kernel4(a: float16, b: float32) -> bool:
        c: bool = a >= b
        return c

    s = process(kernel4)
    assert s(1.0, 1.0) == True
    assert s(1.0, 2.0) == False
    assert s(1000.0, 2000.0) == False
    assert s(-1.0, -2.0) == True
    assert s(-1000.0, -2000.0) == True

    print("test_cast_binary_compare passed")


if __name__ == "__main__":
    test_cast_binary_arith()
    test_cast_binary_compare()
