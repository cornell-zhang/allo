# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
import pytest
from allo.ir.types import int32, float32


def test_pipeline():
    def pipeline(A: int32[10, 20], B: int32[10, 20]) -> int32[10, 20]:
        C: int32[10, 20] = 0
        for i, j in allo.grid(10, 20):
            C[i, j] = A[i, j] + B[i, j]
        return C

    s = allo.customize(pipeline)
    s.pipeline("i", initiation_interval=4)
    print(s.module)


def test_unroll():
    def unroll(A: int32[10, 20], B: int32[10, 20]) -> int32[10, 20]:
        C: int32[10, 20] = 0
        for i, j in allo.grid(10, 20):
            C[i, j] = A[i, j] + B[i, j]
        return C

    s = allo.customize(unroll)
    s.unroll("i", factor=4)
    print(s.module)


def test_fuse():
    def fuse(
        A: int32[10, 20, 30, 40], B: int32[10, 20, 30, 40]
    ) -> int32[10, 20, 30, 40]:
        C: int32[10, 20, 30, 40] = 0
        for i, j, k, l in allo.grid(10, 20, 30, 40):
            C[i, j, k, l] = A[i, j, k, l] + B[i, j, k, l]
        return C

    s = allo.customize(fuse)
    s.fuse("j", "k")
    print(s.module)


def test_reorder():
    def reorder(
        A: int32[10, 20, 30, 40], B: int32[10, 20, 30, 40]
    ) -> int32[10, 20, 30, 40]:
        C: int32[10, 20, 30, 40] = 0
        for i, j, k, l in allo.grid(10, 20, 30, 40):
            C[i, j, k, l] = A[i, j, k, l] + B[i, j, k, l]
        return C

    # axes are consecutive
    def test_case_1():
        s = allo.customize(reorder)
        s.reorder("k", "j")
        print(s.module)

    # axes are not consecutive
    def test_case_2():
        s = allo.customize(reorder)
        s.reorder("l", "i")
        print(s.module)

    test_case_1()
    test_case_2()


def test_split():
    def split(A: int32[10, 20], B: int32[10, 20]) -> int32[10, 20]:
        C: int32[10, 20] = 0
        for i, j in allo.grid(10, 20):
            C[i, j] = A[i, j] + B[i, j]
        return C

    def test_transform_mode_1():
        s = allo.customize(split)
        s.split("j", factor=4)
        print(s.module)

    def test_transform_mode_2():
        s = allo.customize(split)
        s.split("j", factor=3)
        print(s.module)

    test_transform_mode_1()
    test_transform_mode_2()


def test_split_reorder():
    def split_reorder(A: int32[10, 20], B: int32[10, 20]) -> int32[10, 20]:
        C: int32[10, 20] = 0
        for i, j in allo.grid(10, 20):
            C[i, j] = A[i, j] + B[i, j]
        return C

    def test_case_1():
        s = allo.customize(split_reorder)
        s.split("i", factor=2)
        s.split("j", factor=5)
        s.reorder("j.outer", "i.outer", "j.inner", "i.inner")
        print(s.module)

    def test_case_2():
        s = allo.customize(split_reorder)
        s.split("i", factor=3)
        s.split("j", factor=3)
        s.reorder("j.outer", "j.inner", "i.outer", "i.inner")
        print(s.module)

    test_case_1()
    test_case_2()


def test_compute_at():
    def kernel(A: int32[10, 20, 30]) -> int32[10, 20, 30]:
        B: int32[10, 20, 30] = 0
        for i, j, m in allo.grid(10, 20, 30):
            B[i, j, m] = A[i, j, m] * 2

        C: int32[10, 20, 30] = 0
        for ii, jj, mm in allo.grid(10, 20, 30):
            C[ii, jj, mm] = B[ii, jj, mm] + 1
        return C

    def test_case_1():
        # axis 0
        s0 = allo.customize(kernel)
        s0.compute_at("i", "ii")
        print(s0.module)

        # axis 1
        s1 = allo.customize(kernel)
        s1.compute_at("j", "jj")
        print(s1.module)

        # axis 2
        s2 = allo.customize(kernel)
        loops = s2.get_loops()
        s2.compute_at("m", "mm")
        print(s2.module)

    def test_case_2():
        s = allo.customize(kernel)
        s.compute_at("m", "mm")
        s.fuse("ii", "jj")
        print(s.module)

    def test_case_3():
        s = allo.customize(kernel)
        s.compute_at("m", "mm")
        s.split("ii", factor=3)
        s.split("jj", factor=3)
        print(s.module)

    # compute_at and reorder, compute at an axis that is not reordered
    # check both directions of reorder and compute_at
    def test_case_4():
        s = allo.customize(kernel)
        s.compute_at("m", "mm")
        s.reorder("jj", "ii")
        print(s.module)

    # compute_at and reorder, compute at an axis that has been reordered
    # note that the results will be different
    def test_case_5():
        s = allo.customize(kernel)
        s.compute_at("j", "jj")
        s.reorder("jj", "ii")
        print(s.module)

    def test_case_6():
        s = allo.customize(kernel)
        s.compute_at("m", "mm")
        s.split("ii", factor=3)
        s.split("jj", factor=3)
        s.reorder("ii.outer", "jj.outer", "ii.inner", "jj.inner")
        print(s.module)

    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()


def test_compute_at_complex():
    def compute_at_complex(A: int32[10, 20, 30]) -> int32[10, 20, 30]:
        B: int32[10, 20, 30] = 0
        for i, j, m in allo.grid(10, 20, 30):
            B[i, j, m] = A[i, j, m] * 2

        C: int32[10, 20, 30] = 0
        for ii, jj, mm in allo.grid(10, 20, 30):
            C[ii, jj, mm] = B[ii, jj, mm] + 1

        D: int32[10, 20, 30] = 0
        for iii, jjj, mmm in allo.grid(10, 20, 30):
            D[iii, jjj, mmm] = C[iii, jjj, mmm] % 3
        return D

    s = allo.customize(compute_at_complex)
    s.compute_at("j", "jj")
    s.compute_at("m", "mmm")
    print(s.module)


def test_multi_stage():
    def multi_stage(A: int32[10, 10]) -> int32[10]:
        B: int32[10] = 0
        for x in allo.grid(10):
            v: int32 = 0
            for r in allo.reduction(10):
                v += A[x, r]
            B[x] = v
        return B

    s = allo.customize(multi_stage)
    s.split("x", 5)
    print(s.module)


if __name__ == "__main__":
    test_pipeline()
    test_unroll()
    test_fuse()
    test_reorder()
    test_split()
    test_split_reorder()
    test_compute_at()
    test_compute_at_complex()
    test_multi_stage()
