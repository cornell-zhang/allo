# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
import pytest
from allo.ir.types import int32, float32


def test_gemm_grid_for():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    # transformations are applied immediately
    s.split("i", 8)
    s.split("j", 8)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)


def test_schedule():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    # Some meaningless schedules, not used for execution
    s.fuse("i", "j", "k")
    s.unroll("i_j_k_fused")
    s.reshape(s.A, (32, 4, 8))
    s.parallel("i_j_k_fused")
    print(s.module)


def test_pipeline():
    def pipeline(A: int32[10, 20], B: int32[10, 20]) -> int32[10, 20]:
        C: int32[10, 20] = 0
        for i, j in allo.grid(10, 20):
            C[i, j] = A[i, j] + B[i, j]
        return C

    s = allo.customize(pipeline)
    s.pipeline("i", initiation_interval=4)
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


def test_split_imperfect():
    def kernel(A: int32[10, 10]) -> int32[10]:
        B: int32[10] = 0
        for x in allo.grid(10):
            v: int32 = 0
            for r in allo.reduction(10):
                v += A[x, r]
            B[x] = v
        return B

    s = allo.customize(kernel)
    s.split("x", 5)
    print(s.module)


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


def test_multiband():
    def kernel(A: int32[32, 32]) -> int32[32, 32]:
        B: int32[32, 32] = 0
        for i, j in allo.grid(32, 32, name="B"):
            B[i, j] = A[i, j] + 1
        C: int32[32, 32] = 0
        for i, j in allo.grid(32, 32, name="C"):
            C[i, j] = B[i, j] * 2
        return C

    s = allo.customize(kernel)
    loops = s.get_loops()
    print(loops)
    s.compute_at(loops.B.j, loops.C.j)
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
    np_C = (np_A + 1) * 2
    np_B = mod(np_A)
    assert np.array_equal(np_B, np_C)


def test_compute_at():
    def kernel(A: int32[10, 20, 30]) -> int32[10, 20, 30]:
        B: int32[10, 20, 30] = 0
        for i, j, m in allo.grid(10, 20, 30):
            B[i, j, m] = A[i, j, m] * 2

        C: int32[10, 20, 30] = 0
        for ii, jj, mm in allo.grid(10, 20, 30):
            C[ii, jj, mm] = B[ii, jj, mm] + 1
        return C

    # axis 2
    s2 = allo.customize(kernel)
    loops = s2.get_loops()
    s2.compute_at(loops.S_i_j_m_0.m, loops.S_ii_jj_mm_1.mm)
    # _verify_build(s2)
    print(s2.module)
    print(s2.build("vhls"))


def test_access_imperfect_loops():
    M, K, N = 32, 32, 32

    def gemm(inp: float32[M, K], W: float32[K, N], B: float32[N]) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i in range(M):
            for j in range(N):
                for k in allo.reduction(K):
                    outp[i, j] += inp[i, k] * W[k, j]
                for j0 in range(N):
                    outp[i, j0] = B[j0]
        return outp

    s = allo.customize(gemm)
    loops = s.get_loops()
    s.split(loops.S_i_0.i, 8)
    print(s.module)
    loops = s.get_loops()
    s.reorder(loops.S_i_0["i.outer"], loops.S_i_0.j, loops.S_i_0["i.inner"])
    s.unroll(loops.S_i_0.k)
    print(s.module)
    s.fuse(loops.S_i_0["i.outer"], loops.S_i_0.j)
    s.pipeline(loops.S_i_0.j0)
    print(s.build("vhls"))

    s1 = allo.customize(gemm)
    s1.pipeline("j0")
    s1.unroll("j0")
    print(s1.build("vhls"))


if __name__ == "__main__":
    pytest.main([__file__])
