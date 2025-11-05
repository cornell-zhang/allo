# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import re
import pytest
from allo.ir.types import Fixed, int32


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
    code = s.build(target="vhls").hls_code
    assert "B.write" in code
    assert "B.read" in code


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
    code = s.build(target="vhls").hls_code
    assert "C.write" in code
    assert "D.write" in code
    assert "E.write" in code
    assert "D.read" in code
    assert "E.read" in code


def test_nested_function():
    T = int32

    def func1(A: T[10, 20], B: T[10, 20]):
        for i, j in allo.grid(10, 20):
            B[i, j] = A[i, j] + 1

    def func2(A: T[10, 20], B: T[10, 20]):
        for i, j in allo.grid(10, 20):
            B[i, j] = A[i, j] * 2

    def top(A: T[10, 20]) -> T[10, 20]:
        B: T[10, 20]
        C: T[10, 20]
        func1(A, B)
        func2(B, C)
        return C

    s = allo.customize(top)
    s.to(s.B, "func2", depth=1)
    print(s.module)
    code = s.build(target="vhls").hls_code
    assert re.search(r"#pragma HLS stream variable=\w+ depth=1", code)


def test_fork_join_function():
    T = int32

    def func1(A: T[10, 20], B: T[10, 20]):
        for i, j in allo.grid(10, 20):
            B[i, j] = A[i, j] + 1

    def func2(A: T[10, 20], B: T[10, 20]):
        for i, j in allo.grid(10, 20):
            B[i, j] = A[i, j] * 2

    def func3(A: T[10, 20], B: T[10, 20], C: T[10, 20]):
        for i, j in allo.grid(10, 20):
            C[i, j] = A[i, j] + B[i, j]

    def top(A: T[10, 20]) -> T[10, 20]:
        B: T[10, 20]
        C: T[10, 20]
        D: T[10, 20]
        func1(A, B)
        func2(A, C)
        func3(B, C, D)
        return D

    s = allo.customize(top)
    s.to(s.B, "func3")
    s.to(s.C, "func3")
    print(s.module)
    code = s.build(target="vhls").hls_code
    assert re.search(r"#pragma HLS stream variable=\w+ depth=200", code)


def test_dataflow_function():
    T = int32

    def func1(A: T[10, 20], B: T[10, 20]):
        for i, j in allo.grid(10, 20):
            B[i, j] = A[i, j] + 1

    def func2(A: T[10, 20], B: T[10, 20]):
        for i, j in allo.grid(10, 20):
            B[i, j] = A[i, j] - 1
        for i, j in allo.grid(10, 20):
            B[i, j] = A[i, j] * 2

    def top(A: T[10, 20], B: T[10, 20], C: T[10, 20], D: T[10, 20]):
        func1(A, B)
        func2(C, D)

    s = allo.customize(top, instantiate=[8])
    s.dataflow("func2")
    code = s.build(target="vhls").hls_code
    assert "#pragma HLS dataflow" in code


def test_dataflow_loop():
    def case1[N: int32](A: int32, B: int32, C: int32[N]):
        times_out: int32 = A + B
        for i1 in range(times_out, name="out1"):
            for i2 in range(A, name="out2"):
                for j in range(B, name="target"):
                    for k in range(N, name="in1"):
                        C[k] = C[k] + 1
                    for k in range(N, name="in2"):
                        C[k] = C[k] - 1

    s1 = allo.customize(case1, instantiate=[8])
    target_loop = s1.get_loops(s1.top_func_name)["out1"]["j"]
    s1.dataflow(target_loop)
    code = s1.build(target="vhls").hls_code
    assert "#pragma HLS dataflow" in code


if __name__ == "__main__":
    pytest.main([__file__])
