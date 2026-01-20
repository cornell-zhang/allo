# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
from allo.ir.types import int32, Stream, ConstExpr
import allo.dataflow as df
import numpy as np


def test_meta_if():
    Ty = int32
    M, N = 16, 16

    @df.region()
    def top(A: Ty[M, N], B: Ty[M, N]):
        @df.kernel(mapping=[1], args=[A, B])
        def core(local_A: Ty[M, N], local_B: Ty[M, N]):
            pid = df.get_pid()
            with allo.meta_if(pid > 0):
                local_B[:, :] = local_A
            with allo.meta_else():
                local_B[:, :] = local_A

    df.build(top, target="simulator")

    @df.region()
    def top(A: Ty[M, N], B: Ty[M, N]):
        @df.kernel(mapping=[1], args=[A, B])
        def core(local_A: Ty[M, N], local_B: Ty[M, N]):
            a: ConstExpr[int32] = 2
            pid = df.get_pid()
            with allo.meta_if(pid + a > 0):
                local_B[:, :] = local_A
            with allo.meta_else():
                local_B[:, :] = local_A

    df.build(top, target="simulator")

    # @df.region()
    # def top(A: Ty[M, N], B: Ty[M, N]):
    #     @df.kernel(mapping=[1], args=[A, B])
    #     def core(local_A: Ty[M, N], local_B: Ty[M, N]):
    #         a: int32 = 2
    #         pid = df.get_pid()
    #         with allo.meta_if(pid + a > 0): # invalid condition
    #             local_B[:, :] = local_A
    #         with allo.meta_else():
    #             local_B[:, :] = local_A

    # with pytest.raises(SystemExit):
    #     df.build(top, target="simulator")


def test_stream():
    Ty = int32
    M, N, K = 16, 16, 16

    @df.region()
    def top(A: Ty[M, N], B: Ty[M, N]):
        pipe: Stream[Ty, 1][1]

        @df.kernel(mapping=[2], args=[A, B])
        def core(local_A: Ty[M, N], local_B: Ty[M, N]):
            pid = df.get_pid()
            with allo.meta_if(pid == 0):
                pipe[pid].put(local_A)
            with allo.meta_else():
                local_B[:, :] = pipe[pid - 1].get()

    df.build(top, target="simulator")

    @df.region()
    def top(A: Ty[M, N], B: Ty[M, N]):
        pipe: Stream[Ty[N], 1][M]

        @df.kernel(mapping=[2], args=[A, B])
        def core(local_A: Ty[M, N], local_B: Ty[M, N]):
            pid = df.get_pid()
            with allo.meta_for(M) as i:
                with allo.meta_if(pid == 0):
                    pipe[i].put(local_A[i])
                with allo.meta_else():
                    local_B[i, :] = pipe[i].get()

    df.build(top, target="simulator")

    @df.region()
    def top(A: Ty[M, N], B: Ty[M, N]):
        pipe: Stream[Ty, 4]

        @df.kernel(mapping=[1], args=[A])
        def producer(local_A: Ty[M, N]):
            for i, j in allo.grid(M, N):
                # load data
                out: Ty = local_A[i, j]
                # send data
                pipe.put(out)

        @df.kernel(mapping=[1], args=[B])
        def consumer(local_B: Ty[M, N]):
            for i, j in allo.grid(M, N):
                # receive data
                data = pipe.get()
                # computation
                local_B[i, j] = data + 1

    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.zeros((M, N), dtype=np.int32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("Dataflow Simulator Passed!")


if __name__ == "__main__":
    # test_meta_if()
    test_stream()
