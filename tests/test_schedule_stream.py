# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import pytest
from allo.ir.types import Fixed


def test_two_bands():
    T = Fixed(12, 4)

    def kernel(A: T[10, 20, 30]) -> T[10, 20, 30]:
        B: T[10, 20, 30] = 0
        for i, j, m in allo.grid(10, 20, 30):
            B[i, j, m] = A[i, j, m] * 2

        C: T[10, 20, 30] = 0
        for ii, jj, mm in allo.grid(10, 20, 30, name="C"):
            C[ii, jj, mm] = B[ii, jj, mm] + 1
        return C

    s = allo.customize(kernel)
    s.to(s.B, "C")
    print(s.module)
    print(s.build(target="vhls"))


if __name__ == "__main__":
    pytest.main([__file__])
