# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp import process_spmw
from allo.ir.types import int32, ConstExpr, index
from allo import spmw


def test_shard_1D():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[4])
        def core():
            x = spmw.axes()
            local_A = A.shard([x])
            local_B: int32[256] = B.shard([x])
            local_B[:] = local_A + 1

    s = process_spmw(top)

    @spmw.unit()
    def top(A: int32[256], B: int32[1024]):
        @spmw.work(grid=[4])
        def core():
            x = spmw.axes()
            local_A = A.shard([None])
            local_B: int32[256] = B.shard([x])
            local_B[:] = local_A + 1

    s = process_spmw(top)

    @spmw.unit()
    def top(A: int32[256], B: int32[1024]):
        @spmw.work(grid=[4])
        def core():
            x = spmw.axes()
            local_B: int32[256] = B.shard([x])
            local_B[:] = A + 1

    s = process_spmw(top)


def test_shard_2D():
    M, N = 64, 64

    @spmw.unit()
    def top(A: int32[M, N], B: int32[M, N]):
        @spmw.work(grid=[4])
        def core():
            x = spmw.axes()
            local_A = A.shard([x, None])
            local_B = B.shard([x, None])
            local_B[:, :] = local_A + 1

    s = process_spmw(top)

    @spmw.unit()
    def top(A: int32[64, 64], B: int32[64, 64]):
        @spmw.work(grid=[2, 2])
        def core():
            x, y = spmw.axes()
            local_A = A.shard([x, y])
            local_B = B.shard([x, y])
            local_B[:, :] = local_A + 1

    s = process_spmw(top)


def test_get_wid_1D_1():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            for i in range(1024):
                B[i] = A[i] + 1

    mod = process_spmw(top)


def test_get_wid_1D_2():
    vlen = 1024
    P = 4
    tlen = vlen // P

    @spmw.unit()
    def top(A: int32[vlen], B: int32[vlen]):
        @spmw.work(grid=[P])
        def core():
            x = spmw.axes()
            pi: ConstExpr[index] = x.id
            for i in range(tlen * pi, tlen * (pi + 1)):
                B[i] = A[i] + 1

    mod = process_spmw(top)

    @spmw.unit()
    def top(A: int32[vlen], B: int32[vlen]):
        @spmw.work(grid=[P])
        def core():
            x = spmw.axes()
            pi = x.id
            for i in range(tlen * pi, tlen * (pi + 1)):
                B[i] = A[i] + 1

    mod = process_spmw(top)

    @spmw.unit()
    def top(A: int32[vlen], B: int32[vlen]):
        @spmw.work(grid=[P])
        def core():
            x = spmw.axes()
            for i in range(tlen * x.id, tlen * (x.id + 1)):
                B[i] = A[i] + 1

    mod = process_spmw(top)


if __name__ == "__main__":
    test_shard_1D()
    test_shard_2D()
    test_get_wid_1D_1()
    test_get_wid_1D_2()
