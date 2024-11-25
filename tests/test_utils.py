# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32
from allo.ir.transform import find_loop_in_bands, find_func_in_module
from allo.passes import analyze_arg_load_store_in_func, analyze_arg_load_store
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


def test_analyze_load_store():
    def kernel(A: int32[32], B: int32[32], C: int32[32]):
        for i in range(32):
            C[i] = A[i] + B[i]

    s = allo.customize(kernel)
    res = analyze_arg_load_store_in_func(
        s.top_func, arg_names=s.func_args[s.top_func_name]
    )
    assert res["A"] == "in"
    assert res["B"] == "in"
    assert res["C"] == "out"

    def rw_kernel(D: int32[32]):
        for i in range(32):
            D[i] = D[i] + 1

    def top(A: int32[32], B: int32[32], C: int32[32], D: int32[32]):
        kernel(A, B, C)
        rw_kernel(D)

    s = allo.customize(top)
    func = find_func_in_module(s.module, "rw_kernel")
    res = analyze_arg_load_store_in_func(func, arg_names=s.func_args["rw_kernel"])
    assert res["D"] == "both"

    def write_kernel(A: int32[32]):
        for i in range(32):
            A[i] = i

    def top2(A: int32[32], B: int32[32], C: int32[32], D: int32[32]):
        kernel(A, B, C)
        write_kernel(A)
        rw_kernel(D)

    s = allo.customize(top2)
    res = analyze_arg_load_store(s.module, s.func_args)
    assert res["A"] == "both"
    assert res["B"] == "in"
    assert res["C"] == "out"
    assert res["D"] == "both"


if __name__ == "__main__":
    pytest.main([__file__])
