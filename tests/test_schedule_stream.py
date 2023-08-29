# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import pytest
from allo.ir.types import Fixed


def test_two_bands():
    T = Fixed(12, 4)

    def kernel(A: T[10, 20, 30]) -> T[10, 20, 30]:
        B: T[10, 20, 30]
        for i, j, m in allo.grid(10, 20, 30):
            B[i, j, m] = A[i, j, m] * 2

        C: T[10, 20, 30]
        for ii, jj, mm in allo.grid(10, 20, 30, name="C"):
            C[ii, jj, mm] = B[ii, jj, mm] + 1
        return C

    s = allo.customize(kernel)
    s.to(s.B, "C")
    print(s.module)
    print(s.build(target="vhls"))


def test_fork_join():
    T = Fixed(15, 13)

    def kernel(A: T[10, 20], B: T[10, 20]) -> T[10, 20]:
        C: T[10, 20]
        D: T[10, 20]
        E: T[10, 20]
        F: T[10, 20]
        for i, j in allo.grid(10, 20):
            C[i, j] = A[i, j] + B[i, j]
        for ii, jj in allo.grid(10, 20, name="D"):
            D[ii, jj] = C[ii, jj] + 1
        for ii, jj in allo.grid(10, 20, name="E"):
            E[ii, jj] = C[ii, jj] + 1
        for ii, jj in allo.grid(10, 20, name="F"):
            F[ii, jj] = D[ii, jj] * E[ii, jj]
        return F

    s = allo.customize(kernel)
    s.to(s.C, "D")
    s.to(s.C, "E")
    s.to(s.D, "F")
    s.to(s.E, "F")
    print(s.module)
    print(s.build(target="vhls"))


if __name__ == "__main__":
    pytest.main([__file__])
