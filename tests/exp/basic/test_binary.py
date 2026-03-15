# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
from allo.ir.types import int32, uint16, bool, float32
from allo.spmw import kernel


def test_arith():
    @kernel
    def kernel1() -> int32:
        A: int32 = 0
        B: int32 = 1
        A = 0 - 1
        A = B + B
        return A

    s = process(kernel1)
    assert s() == 2

    @kernel
    def kernel1() -> uint16:
        A: uint16 = 0
        B: uint16 = 1
        A = B + B
        return A

    s = process(kernel1)
    assert s() == 2

    @kernel
    def kernel2() -> int32:
        A: int32 = 0
        B: int32 = 0
        A = B - 1
        return A

    s = process(kernel2)
    assert s() == -1

    @kernel
    def kernel3() -> int32:
        A: int32 = 0
        B: int32 = 1
        C: int32 = 0
        A, C = B * 2, 1 + B
        return A

    s = process(kernel3)
    assert s() == 2

    @kernel
    def kernel3() -> uint16:
        A: uint16 = 0
        B: uint16 = 1
        C: uint16 = 2
        A, C = B * C, 1 + B
        return A

    s = process(kernel3)
    assert s() == 2

    @kernel
    def kernel4() -> int32:
        A: int32 = 4
        B: int32 = 2
        A = B // 2
        return A

    s = process(kernel4)
    assert s() == 1

    @kernel
    def kernel5() -> int32:
        A: int32 = 5
        B: int32 = 2
        A = A % B
        return A

    s = process(kernel5)
    assert s() == 1

    # TODO: ast.Pow not supported for now
    # def kernel6() -> int32:
    #     A: int32 = 0
    #     B: int32 = 0
    #     A = B**2
    #     return A

    # s = process(kernel6)

    @kernel
    def kernel7() -> int32:
        A: int32 = 0
        B: int32 = 0
        A = 1 + B + B + 2
        return A

    s = process(kernel7)
    assert s() == 3

    @kernel
    def kernel8() -> uint16:
        A: uint16 = 0
        B: uint16 = 0
        A = 1 + B + B + 2
        return A

    s = process(kernel8)
    assert s() == 3

    @kernel
    def kernel9(a: int32, b: int32) -> int32:
        A: int32 = a
        B: int32 = b
        A = 1 + B + B + 2
        return A

    s = process(kernel9)
    assert s(1, 2) == 7
    assert s(2, 3) == 9

    print("pass test_arith")


def test_broadcast():
    @kernel
    def kernel1() -> int32[10]:
        A: int32[10] = 0
        B: int32 = 1
        C: int32[10] = A + B
        return C

    s = process(kernel1)
    np_A = np.zeros((10,), dtype=np.int32)
    np_B = np_A + 1
    assert np.array_equal(s(), np_B)

    @kernel
    def kernel1(A: int32[10]) -> int32[10]:
        B: int32 = 1
        C: int32[10] = A + B
        return C

    s = process(kernel1)
    np_A = np.random.randint(0, 10, (10,), dtype=np.int32)
    np_B = np_A + 1
    assert np.array_equal(s(np_A), np_B)

    @kernel
    def kernel2() -> int32[10]:
        A: int32[10] = 0
        B: int32[1] = 1
        C: int32[10] = A + B[0]
        return C

    s = process(kernel2)
    np_A = np.zeros((10,), dtype=np.int32)
    np_B = np_A + 1
    assert np.array_equal(s(), np_B)

    @kernel
    def kernel3(A: int32[10]) -> int32[10]:
        C: int32[10] = A + 1
        return C

    s = process(kernel3)
    np_A = np.random.randint(0, 10, (10,), dtype=np.int32)
    np_B = np_A + 1
    assert np.array_equal(s(np_A), np_B)

    print("pass test_broadcast")


def test_compare():
    @kernel
    def kernel1() -> bool:
        A: int32 = 0
        B: int32 = 0
        C: bool = A == B
        return C

    s = process(kernel1)
    assert s() == True

    @kernel
    def kernel2() -> bool:
        A: uint16 = 0
        B: uint16 = 1
        C: bool = A == B
        return C

    s = process(kernel2)
    assert s() == False

    @kernel
    def kernel3() -> bool:
        A: float32 = 1.0
        B: float32 = 0
        B += 1
        return A == B

    s = process(kernel3)
    assert s() == True

    @kernel
    def kernel2() -> bool:
        A: int32 = 0
        B: int32 = 0
        C: bool = A != B
        return C

    s = process(kernel2)
    assert s() == False

    @kernel
    def kernel2() -> bool:
        A: uint16 = 0
        B: uint16 = 1
        C: bool = A != B
        return C

    s = process(kernel2)
    assert s() == True

    @kernel
    def kernel3() -> bool:
        A: float32 = 1.0
        B: float32 = 0
        B -= 1
        return A != B

    s = process(kernel3)
    assert s() == True

    @kernel
    def kernel4() -> bool:
        A: int32 = 0
        B: int32 = 0
        C: bool = A > B
        return C

    s = process(kernel4)
    assert s() == False

    @kernel
    def kernel4(A: uint16, B: uint16) -> bool:
        return A > B

    s = process(kernel4)
    assert s(1, 0) == True
    assert s(0, 1) == False
    assert s(1, 1) == False

    @kernel
    def kernel4(A: float32, B: float32) -> bool:
        return A > B

    s = process(kernel4)
    assert s(1.0, 0.0) == True
    assert s(0.0, 1.0) == False
    assert s(1.0, 1.0) == False

    @kernel
    def kernel4(A: int32, B: int32) -> bool:
        return A >= B

    s = process(kernel4)
    assert s(1, 0) == True
    assert s(0, 1) == False
    assert s(1, 1) == True

    @kernel
    def kernel5(A: uint16, B: uint16) -> bool:
        return A >= B

    s = process(kernel5)
    assert s(1, 0) == True
    assert s(0, 1) == False
    assert s(1, 1) == True

    @kernel
    def kernel5(A: float32, B: float32) -> bool:
        return A >= B

    s = process(kernel5)
    assert s(1.0, 0.0) == True
    assert s(0.0, 1.0) == False
    assert s(1.0, 1.0) == True

    @kernel
    def kernel6(A: int32, B: int32) -> bool:
        return A <= B

    s = process(kernel6)
    assert s(1, 0) == False
    assert s(0, 1) == True
    assert s(1, 1) == True

    @kernel
    def kernel6(A: uint16, B: uint16) -> bool:
        return A <= B

    s = process(kernel6)
    assert s(1, 0) == False
    assert s(0, 1) == True
    assert s(1, 1) == True

    @kernel
    def kernel6(A: float32, B: float32) -> bool:
        return A <= B

    s = process(kernel6)
    assert s(1.0, 0.0) == False
    assert s(0.0, 1.0) == True
    assert s(1.0, 1.0) == True

    @kernel
    def kernel7(A: int32, B: int32) -> bool:
        return A < B

    s = process(kernel7)
    assert s(1, 0) == False
    assert s(0, 1) == True
    assert s(1, 1) == False
    assert s(-1, 0) == True
    assert s(0, -1) == False
    assert s(-1, -1) == False

    @kernel
    def kernel7(A: uint16, B: uint16) -> bool:
        return A < B

    s = process(kernel7)
    assert s(1, 0) == False
    assert s(0, 1) == True
    assert s(1, 1) == False

    @kernel
    def kernel7(A: float32, B: float32) -> bool:
        return A < B

    s = process(kernel7)
    assert s(1.0, 0.0) == False
    assert s(0.0, 1.0) == True
    assert s(1.0, 1.0) == False
    assert s(-1.0, 0.0) == True
    assert s(0.0, -1.0) == False
    assert s(-1.0, -1.0) == False

    @kernel
    def kernel7() -> bool:
        A: int32 = 0
        B: bool = 0 <= A
        C: bool = A > 0
        return B == C

    s = process(kernel7)
    assert s() == False

    @kernel
    def kernel7() -> bool:
        A: int32 = 0
        B: bool = 0 <= A
        C: bool = A > 0
        return B != C

    s = process(kernel7)
    assert s() == True

    @kernel
    def kernel8() -> bool:
        C: bool = 1 >= 0
        return C

    s = process(kernel8)
    assert s() == True

    @kernel
    def kernel9() -> bool:
        C: bool = 1 < 0
        return C

    s = process(kernel9)
    assert s() == False

    print("pass test_compare")


if __name__ == "__main__":
    test_arith()
    test_broadcast()
    test_compare()
