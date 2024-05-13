# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32
from allo.ir.transform import find_loop_in_bands
import pytest


def test_get_loops():
    def kernel(A: int32[32]):
        for i in range(32):
            A[i] = i
        for ii in range(32):
            for jj in range(32):
                A[ii] = A[ii] * 2
        for i0 in allo.grid(32, name="imperfect"):
            A[i0] = A[i0] + 1
            for i1 in range(32):
                A[i1] = A[i1] * 2
            A[i0] = A[i0] + 1
            for i2 in range(32):
                A[i2] = A[i2] + 1
                for i3 in range(32):
                    A[i3] = A[i3] * 2

    s = allo.customize(kernel)
    loops = s.get_loops("kernel")
    assert hasattr(loops["S_i_0"], "i")
    assert hasattr(loops["S_ii_1"], "jj")
    assert hasattr(loops["imperfect"], "i0")
    assert hasattr(loops["imperfect"], "i1")
    assert hasattr(loops["imperfect"], "i2")


def test_func_call():
    def kernel(A: int32[32]):
        for i in range(32):
            A[i] = i

    def top(A: int32[32]):
        for i in range(32):
            kernel(A)
            for j in range(32):
                A[j] = A[j] * 2

    s = allo.customize(top)
    print(s.module)
    band_name, axis_name = find_loop_in_bands(s.top_func, "j")
    assert band_name == "S_i_0"
    assert axis_name == "j"


if __name__ == "__main__":
    pytest.main([__file__])
