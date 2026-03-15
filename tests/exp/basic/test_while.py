# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
from allo.ir.types import int32, uint16
from allo.spmw import kernel


def test_while():
    @kernel
    def kernel1() -> int32:
        A: int32 = 0
        while A < 10:
            A += 1
        return A

    s = process(kernel1)
    assert s() == 10

    @kernel
    def kernel2() -> int32:
        A: int32 = 0
        while A:
            A += 1
        return A

    s = process(kernel2)
    assert s() == 0

    @kernel
    def kernel3(A: int32[10], B: int32[10]):
        idx: uint16 = 0
        while idx < 10:
            A[idx] = B[idx]
            idx += 1

    s = process(kernel3)
    np_A = np.zeros((10,), dtype=np.int32)
    np_B = np.random.randint(0, 10, (10,), dtype=np.int32)
    s(np_A, np_B)
    assert np.array_equal(np_A, np_B)

    @kernel
    def kernel4(A: int32[4, 10], B: int32[4, 10]):
        for i in range(4):
            idx: uint16 = 0
            while idx < 10:
                A[i][idx] = B[i][idx]
                idx += 1

    s = process(kernel4)
    np_A = np.zeros((4, 10), dtype=np.int32)
    np_B = np.random.randint(0, 10, (4, 10), dtype=np.int32)
    s(np_A, np_B)
    assert np.array_equal(np_A, np_B)


if __name__ == "__main__":
    test_while()
