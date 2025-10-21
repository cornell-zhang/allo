# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 4, 4, 4
P0, P1 = K, N + 2


@df.region()
def top():
    fifo_A = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(P0, P1))
    fifo_B = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(P0, P1))

    @df.kernel(mapping=[P0, P1])
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        # Weight stationary GEMM systolic array
        # B is the matrix that contains the stationary weights
        i, j = df.get_pid()

        # periperals kernels
        with allo.meta_if(j == 0):
            for m in range(M):
                fifo_A[i, j + 1].put(A[m, i])

        # drain
        with allo.meta_elif(j == N + 1):
            for m in range(M):
                fifo_A[i, j].get()

        # compute
        # There are three cases: i == 0, i == K - 1, and the rest
        with allo.meta_elif(i == 0):
            # Does not take partial sum from the previous PE
            b: float32 = B[i, j - 1]
            for m in range(M):
                a = fifo_A[i, j].get()
                fifo_A[i, j + 1].put(a)
                fifo_B[i + 1, j].put(a * b)
        with allo.meta_elif(i == K - 1):
            # Does not keep passing the partial sum to the next PE
            # Concludes the computation and writes to the output
            b: float32 = B[i, j - 1]
            for m in range(M):
                partial_sum = fifo_B[i, j].get()
                a = fifo_A[i, j].get()
                C[m, j - 1] = partial_sum + a * b
                fifo_A[i, j + 1].put(a)
        with allo.meta_else():
            # Continues the computation
            b: float32 = B[i, j - 1]
            for m in range(M):
                partial_sum = fifo_B[i, j].get()
                a = fifo_A[i, j].get()
                fifo_A[i, j + 1].put(a)
                fifo_B[i + 1, j].put(partial_sum + a * b)


def test_systolic():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    llvm_mod = df.build(top, target="simulator")
    llvm_mod(A, B, C)
    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
    print("Dataflow Simulator Passed!")

    mod = df.build(top)
    if hls.is_available("vitis_hls"):
        C = np.zeros((M, N), dtype=np.float32)
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_systolic()
