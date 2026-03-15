# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from src.main import process
import numpy as np
from allo.ir.types import int32, bool
from allo.spmw import kernel


def test_branch():
    @kernel
    def kernel1() -> int32:
        A: int32 = 0
        B: int32 = 0
        if A > B:
            B = A
        if True:
            B = 1
        return B

    s = process(kernel1)
    assert s() == 1

    @kernel
    def kernel1(A: int32, B: int32[1]) -> int32:
        if A > B[0]:
            B[0] = A
        if False:
            B[0] = 1
        return B[0]

    s = process(kernel1)
    assert s(1, np.array([0], dtype=np.int32)) == 1
    assert s(0, np.array([1], dtype=np.int32)) == 1
    assert s(1, np.array([2], dtype=np.int32)) == 2

    @kernel
    def kernel2() -> int32:
        A: int32 = 0
        B: int32 = 1
        if A > B:
            B = A
        else:
            B = B
        return B

    s = process(kernel2)
    assert s() == 1

    @kernel
    def kernel2(A: int32, B: int32[1]) -> int32:
        if A > B[0]:
            B[0] = A
        else:
            B[0] = B[0]
        return B[0]

    s = process(kernel2)
    assert s(1, np.array([0], dtype=np.int32)) == 1
    assert s(0, np.array([1], dtype=np.int32)) == 1
    assert s(1, np.array([2], dtype=np.int32)) == 2

    @kernel
    def kernel3() -> int32:
        A: int32 = 1
        B: int32 = 0
        if A > B:
            B = A
        elif A < B:
            B = B
        else:
            B = 0
        return B

    s = process(kernel3)
    assert s() == 1

    @kernel
    def kernel3(A: int32, B: int32[1]) -> int32:
        if A > B[0]:
            B[0] = A
        elif A < B[0]:
            B[0] = B[0]
        else:
            B[0] = 0
        return B[0]

    s = process(kernel3)
    assert s(1, np.array([0], dtype=np.int32)) == 1
    assert s(0, np.array([1], dtype=np.int32)) == 1
    assert s(1, np.array([2], dtype=np.int32)) == 2

    @kernel
    def kernel4() -> bool:
        A: bool = 1 == 1
        B: bool
        if not A:
            B: bool = not False
        else:
            B: bool = not True
        return B

    s = process(kernel4)
    assert s() == False

    @kernel
    def kernel4(A: bool) -> bool:
        B: bool
        if not A:
            B: bool = not False
        else:
            B: bool = not True
        return B

    s = process(kernel4)
    assert s(True) == False
    assert s(False) == True

    print("pass test_branch")


def test_branch_complicate():
    @kernel
    def kernel1() -> int32:
        A: int32 = 2
        B: int32 = 0
        if A > B and 1 == 1:
            B = A
        if True:
            B = 1
        return B

    s = process(kernel1)
    assert s() == 1

    @kernel
    def kernel2() -> int32:
        A: int32 = 0
        B: int32 = -1
        if A > B:
            B = A
        elif A < B or B == 0:
            B = B
        else:
            B = 1
        return B

    s = process(kernel2)
    assert s() == 0

    @kernel
    def kernel3() -> int32:
        A: int32 = -1
        B: int32 = 0
        if True and A > B:
            B = A
        elif A < B or B == 0 and False:
            B = B + 1
        else:
            B = 0
        return B

    s = process(kernel3)
    assert s() == 1

    print("pass test_branch_complicate")


if __name__ == "__main__":
    test_branch()
    test_branch_complicate()
