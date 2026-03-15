# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp import process
from allo.ir.types import int32, ConstExpr
from allo.spmw import kernel
import numpy as np


def test_annassign():
    """
    Test the annotated assignment.
    """
    zero = 0
    one = 1 + zero

    @kernel
    def kernel1() -> int32:
        """
        Initialize variables with constants.
        """
        A: int32 = 0
        B: int32 = zero
        C: ConstExpr[int32] = one
        D: ConstExpr[int32] = C + 2
        E: int32 = C + 1
        F: int32 = C
        return E

    s = process(kernel1)
    assert s() == one + 1

    @kernel
    def kernel2(A: int32) -> int32:
        """
        Initialize variables with arguments or varibales.
        """
        B: int32 = A
        C: int32 = B
        return C

    s = process(kernel2)
    assert s(1) == 1
    assert s(2) == 2
    assert s(-1) == -1

    @kernel
    def kernel3(a: int32[8]) -> int32[8]:
        b: int32[8] = a
        return b

    s = process(kernel3)
    np_A = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_A, np_B)

    print("test_annassign passed!")


def test_assign():
    @kernel
    def kernel1() -> int32:
        B: int32 = 3
        A = B
        return A

    s = process(kernel1)
    assert s() == 3

    @kernel
    def kernel2() -> int32:
        A: int32 = 0
        B: int32 = 0
        C: int32 = 0
        A, C = B, B
        return A

    s = process(kernel2)
    assert s() == 0

    @kernel
    def kernel3() -> int32[2]:
        b: int32[2]
        b[0] = 1
        b[1] = 0
        return b

    s = process(kernel3)
    assert np.array_equal(s(), np.array([1, 0], dtype=np.int32))

    @kernel
    def kernel4() -> int32[2]:
        b: int32[2]
        b[0] = 1
        b[1] = 0
        a: int32[2, 2]
        a[0] = b
        a[1, :] = b[:]
        return b

    s = process(kernel4)
    assert np.array_equal(s(), np.array([1, 0], dtype=np.int32))

    @kernel
    def kernel5() -> int32[4, 3, 2]:
        b: int32[2] = 1
        a: int32[4, 3, 2]
        a[0, :] = 0
        a[0] = b[:]
        b[0] = 2
        b_: int32[2] = b[:]
        a[1, :] = b_[:]
        a[2, :, :] = b_
        return a

    s = process(kernel5)
    assert np.array_equal(
        s(),
        np.array(
            [
                [[1, 1], [1, 1], [1, 1]],
                [[2, 1], [2, 1], [2, 1]],
                [[2, 1], [2, 1], [2, 1]],
                [[0, 0], [0, 0], [0, 0]],
            ],
            dtype=np.int32,
        ),
    )

    @kernel
    def kernel6(b: int32[2]) -> int32[4, 3, 2]:
        a: int32[4, 3, 2]
        a[0, :] = 0
        a[0] = b[:]
        b[0] = 2
        b_: int32[2] = b[:]
        a[1, :] = b_[:]
        a[2, :, :] = b_
        return a

    s = process(kernel6)
    assert np.array_equal(
        s(np.array([4, 3], dtype=np.int32)),
        np.array(
            [
                [[4, 3], [4, 3], [4, 3]],
                [[2, 3], [2, 3], [2, 3]],
                [[2, 3], [2, 3], [2, 3]],
                [[0, 0], [0, 0], [0, 0]],
            ],
            dtype=np.int32,
        ),
    )

    print("test_assign passed!")


def test_augassign():
    @kernel
    def kernel1() -> int32:
        A: int32 = 0
        B: int32 = 1
        A += B
        return A

    s = process(kernel1)
    assert s() == 1

    @kernel
    def kernel2() -> int32:
        A: int32[2] = 0
        B: int32 = 1
        A[0] += B
        return A[0]

    s = process(kernel2)
    assert s() == 1

    @kernel
    def kernel3() -> int32:
        A: int32 = 2
        B: int32 = 1
        A -= B
        return A

    s = process(kernel3)
    assert s() == 1

    @kernel
    def kernel4() -> int32:
        A: int32 = 2
        B: int32 = 2
        A *= B
        return A

    s = process(kernel4)
    assert s() == 4

    @kernel
    def kernel5() -> int32:
        A: int32[2] = 2
        A[0] -= 1
        return A[0]

    s = process(kernel5)
    assert s() == 1

    @kernel
    def kernel6() -> int32:
        A: int32[2, 2] = 0
        A[0] -= 1
        return A[0, 0]

    s = process(kernel6)
    assert s() == -1

    @kernel
    def kernel7() -> int32:
        A: int32[2, 2] = 2
        A[0] *= 2
        return A[0, 0]

    s = process(kernel7)
    assert s() == 4

    @kernel
    def kernel8() -> int32:
        A: int32[2, 2] = 2
        A[0] /= 2
        return A[0, 0]

    s = process(kernel8)
    assert s() == 1

    print("test_augassign passed!")


def test_broadcast_init():
    @kernel
    def kernel1() -> int32[2]:
        a: int32[2] = 1
        b: int32[32, 32] = 0
        return a

    s = process(kernel1)
    assert np.array_equal(s(), np.array([1, 1], dtype=np.int32))

    @kernel
    def kernel2() -> int32:
        a: int32 = 1
        b: int32[32, 32] = a
        return b[0, 0]

    s = process(kernel2)
    assert s() == 1

    @kernel
    def kernel3() -> int32:
        a: int32[32] = 1
        b: int32[4, 32] = a
        return b[0, 0]

    s = process(kernel3)
    assert s() == 1

    @kernel
    def kernel4(a: int32[32]) -> int32[32]:
        b: int32[4, 32] = a
        return b[1]

    s = process(kernel4)
    np_A = np.random.randint(0, 10, size=32, dtype=np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A)

    @kernel
    def kernel5(a: int32[32]) -> int32[32]:
        b: int32[4, 32] = a
        return b[2, :]

    s = process(kernel5)
    np_A = np.random.randint(0, 10, size=32, dtype=np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A)

    @kernel
    def kernel6(a: int32[32]) -> int32[2, 32]:
        b: int32[4, 2, 32] = a
        return b[3, :]

    s = process(kernel6)
    np_A = np.random.randint(0, 10, size=32, dtype=np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_B[0], np_A)
    assert np.array_equal(np_B[1], np_A)

    print("test_broadcast_init passed!")


if __name__ == "__main__":
    test_annassign()
    test_assign()
    test_augassign()
    test_broadcast_init()
