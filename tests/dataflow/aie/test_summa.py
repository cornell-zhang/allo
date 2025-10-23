# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import allo.dataflow as df
from allo.ir.types import int32, Stream
from allo.memory import Layout
import numpy as np


def _test_summa_2x2():
    Ty = int32
    M, K, N = 8, 8, 8
    P0, P1 = 2, 2

    La = Layout("RS1")
    Lb = Layout("S1S0")

    @df.region()
    def top():
        row_fifo: Stream[Ty[M, N // 2], 1][P0]
        final_fifo: Stream[Ty[M, N], P0][P0]

        @df.kernel(mapping=[P0, P1])
        def summa(A: Ty[M, K] @ La, B: Ty[K, N] @ Lb):
            i, j = df.get_pid()

            P_tile: Ty[M, N // 2] = allo.matmul(A, B)

            with allo.meta_if(j == 1):
                row_fifo[i].put(P_tile)
            with allo.meta_else():
                F_tile: Ty[M, N] = 0
                right_half: Ty[M, N // 2] = row_fifo[i].get()
                with allo.meta_for(M) as m:
                    with allo.meta_for(N // 2) as n:
                        F_tile[m, n] = P_tile[m, n]
                        F_tile[m, n + N // 2] = right_half[m, n]
                final_fifo[i].put(F_tile)

        @df.kernel(mapping=[1])
        def write_c(C: Ty[M, N]):
            agg: Ty[M, N] = 0
            with allo.meta_for(P0) as i:
                agg[:, :] += final_fifo[i].get()
            C[:, :] = agg

    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)

    mod = df.build(top, target="aie")
    mod(A, B, C)

    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_summa():
    Ty = int32
    M, K, N = 32, 32, 32
    P0, P1 = 4, 4

    La = Layout("RS0")
    Lb = Layout("S0S1")
    Lc = Layout("RS1")

    @df.region()
    def top():
        column_fifo: Stream[Ty[M, N // P0], 1][P0, P1 - 1]

        @df.kernel(mapping=[P0, P1])
        def summa(B: Ty[K, N] @ Lb, A: Ty[M, K] @ La, C: Ty[M, N] @ Lc):
            i, j = df.get_pid()

            P_tile: Ty[M, N // P0] = allo.matmul(A, B)

            with allo.meta_if(j == 0):
                P_tile[:, :] += column_fifo[i, j].get()
                C[:, :] = P_tile
            with allo.meta_elif(j == P1 - 1):
                column_fifo[i, j - 1].put(P_tile)
            with allo.meta_else():
                P_tile[:, :] += column_fifo[i, j].get()
                column_fifo[i, j - 1].put(P_tile)

    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)

    mod = df.build(top, target="aie")
    mod(B, A, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_summa_2x2()
    _test_summa()
