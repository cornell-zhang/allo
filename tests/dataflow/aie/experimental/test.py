# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout


def _test_vector_scalar_add():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024
    LyA = Layout("R")
    LyB = Layout("S0")

    @df.region()
    def top():
        @df.kernel(mapping=[4])
        def core(A: Ty[M // 4] @ LyA, B: Ty[M] @ LyB):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, M // 4).astype(np.int32)
    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[("bundle", ["core_0", "core_1", "core_2", "core_3"])],
    )
    B = np.zeros(M).astype(np.int32)
    mod(A, B)
    print("PASSED!")


LyW1 = Layout("RS0")
LyW2 = Layout("S0R")


def test_producer_consumer():

    Ty = int32
    M, N, K = 16, 16, 16

    @df.region()
    def top():
        pipe = df.pipe(dtype=Ty, shape=(M, N), depth=1)
        pipe_rev = df.pipe(dtype=Ty, shape=(M, N), depth=1)

        @df.kernel(mapping=[1])
        def consumer(B: Ty[M, N]):
            data = pipe.get()
            tmp: Ty[M, N] = 0
            for i, j in allo.grid(M, N):
                # computation
                tmp[i, j] = data[i, j] + B[i, j]
            pipe_rev.put(tmp)

        @df.kernel(mapping=[1])
        def producer(A: Ty[M, N], C: Ty[M, N]):
            pipe.put(A)
            data = pipe_rev.get()
            for i, j in allo.grid(M, N):
                C[i, j] = data[i, j]

    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.zeros((M, N), dtype=np.int32)

    mod = df.build(top, target="aie-mlir")
    mod(A, A, B)
    np.testing.assert_allclose(A + A, B, atol=1e-5)
    print("Passed!")


def _test_tensor_parallelism():
    Ty = int32
    M, K, N, L = 8, 8, 8, 8
    P0 = 2
    Nt = N // P0

    @df.region()
    def top():
        Y = df.array(df.pipe(dtype=Ty, shape=(M, Nt), depth=2), shape=(P0,))
        part_Z = df.array(df.pipe(dtype=Ty, shape=(M, L), depth=2), shape=(P0,))

        @df.kernel(mapping=[P0])
        def gemm0(X: Ty[M, K], W1: Ty[K, N] @ LyW1):
            pn = df.get_pid()
            Y[pn].put(allo.matmul(X, W1))

        @df.kernel(mapping=[P0])
        def gemm1(W2: Ty[N, L] @ LyW2):
            pn = df.get_pid()
            part_Z[pn].put(allo.matmul(Y[pn].get(), W2))

        @df.kernel(mapping=[1])
        def acc(Z: Ty[M, L]):
            Z_out: Ty[M, L] = 0
            with allo.meta_for(P0) as i:
                Z_out[:, :] += part_Z[i].get()
            Z[:, :] = Z_out

    mod = df.build(top, target="aie-mlir")


LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")


def _test_gemm_1D():
    Ty = int32
    M, N, K = 16, 16, 16
    P0 = 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N], C: Ty[M, N] @ LyA):
            C[:, :] = allo.matmul(A, B)

    # df.build(top, target="aie-mlir")
    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_gemm_2D():
    TyI, TyO = int32, int32
    M, N, K = 16, 16, 16
    P0, P1 = 4, 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            C[:, :] = allo.matmul(A, B)

    # df.build(top, target="aie-mlir")
    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_summa_2x2():
    Ty = int32
    M, K, N = 8, 8, 8
    P0, P1 = 2, 2

    La = Layout("RS1")
    Lb = Layout("S1S0")

    @df.region()
    def top():
        row_fifo = df.array(df.pipe(dtype=Ty, shape=(M, N // 2), depth=1), shape=(P0,))
        final_fifo = df.array(df.pipe(dtype=Ty, shape=(M, N), depth=P0), shape=(P0,))

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

    mod = df.build(top, target="aie-mlir")


def _test_summa():
    Ty = int32
    M, K, N = 32, 32, 32
    P0, P1 = 4, 4

    La = Layout("RS0")
    Lb = Layout("S0S1")
    Lc = Layout("RS1")

    @df.region()
    def top():
        column_fifo = df.array(
            df.pipe(dtype=Ty, shape=(M, N // P0), depth=1), shape=(P0, P1 - 1)
        )

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

    mod = df.build(top, target="aie-mlir")


if __name__ == "__main__":
    # _test_vector_scalar_add()
    test_producer_consumer()
    # _test_summa()
    # _test_summa_2x2()
    # _test_tensor_parallelism()
    # _test_gemm_1D()
    # _test_gemm_2D()
