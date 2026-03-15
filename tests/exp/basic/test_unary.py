# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process
import numpy as np
from allo.ir.types import int32, bool
from allo.spmw import kernel


def test_unary():
    @kernel
    def kernel1() -> int32:
        A: int32 = 1
        B: int32 = +A
        return B

    s = process(kernel1)
    assert s() == 1

    @kernel
    def kernel2() -> int32:
        A: int32 = 1
        B: int32 = -A
        return B

    s = process(kernel2)
    assert s() == -1

    @kernel
    def kernel3() -> int32[10]:
        A: int32[10] = 1
        B: int32[10] = -A
        return B

    s = process(kernel3)
    assert np.array_equal(s(), np.full((10,), -1))

    @kernel
    def kernel3(A: int32[10]) -> int32[10]:
        B: int32[10] = -A
        return B

    s = process(kernel3)
    np_A = np.random.randint(0, 10, 10)
    assert np.array_equal(s(np_A), -np_A)

    @kernel
    def kernel4() -> int32[10]:
        A: int32[10] = 1
        B: int32[10] = +A
        return B

    s = process(kernel4)
    assert np.array_equal(s(), np.full((10,), 1))

    @kernel
    def kernel5() -> int32:
        A: int32 = +1
        return A

    s = process(kernel5)
    assert s() == 1

    @kernel
    def kernel6() -> int32:
        A: int32 = -1
        return A

    s = process(kernel6)
    assert s() == -1

    @kernel
    def kernel7() -> int32:
        A: int32 = 1
        B: int32 = 1
        C: int32 = 1
        A, B, C = -A, +B, -C
        return A + B + C

    s = process(kernel7)
    assert s() == -1

    @kernel
    def kernel8() -> int32[10]:
        A: int32[10] = +1
        B: int32[10] = -1
        A, B = -A, +B
        return A + B

    s = process(kernel8)
    assert np.array_equal(s(), np.full((10,), -2))

    print("pass test_unary")


def test_unary_not():
    @kernel
    def kernel1() -> bool:
        A: bool = 1 == 1
        B: bool = not A
        return B

    s = process(kernel1)
    assert s() == False

    @kernel
    def kernel2() -> bool:
        A: bool = 1 == 1
        B: bool = not A == True
        return B

    s = process(kernel2)
    assert s() == False

    @kernel
    def kernel3() -> bool:
        A: bool = 1 == 1
        B: bool = not True
        return B

    s = process(kernel3)
    assert s() == False

    @kernel
    def kernel4() -> bool:
        A: bool = 1 == 1
        B: bool = not False
        return B

    s = process(kernel4)
    assert s() == True

    print("pass test_unary_not")


if __name__ == "__main__":
    test_unary()
    test_unary_not()
