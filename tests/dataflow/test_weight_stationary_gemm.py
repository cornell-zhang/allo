# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 4, 4, 4
P0, P1 = M + 2, K


@df.region()
def top():
    fifo_A = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(P0, P1))
    fifo_B = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(P0, P1))

    @df.kernel(mapping=[P0, P1])
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        # Weight stationary GEMM systolic array
        # A is contains the stationary weights
        i, j = df.get_pid()
        # periperals kernels
        with allo.meta_if(i == 0):
            for n in range(N):
                fifo_B[i + 1, j].put(B[j, n])

        # drain
        with allo.meta_elif(i == M + 1):
            for n in range(N):
                fifo_B[i, j].get()

        with allo.meta_elif(j == 0):
            for n in range(N):
                a: float32 = A[i - 1, j]
                b = fifo_B[i, j].get()
                fifo_A[i, j + 1].put(a * b)
                fifo_B[i + 1, j].put(b)
        with allo.meta_elif(j == K - 1):
            for n in range(N):
                partial_sum = fifo_A[i, j].get()
                a: float32 = A[i - 1, j]
                b = fifo_B[i, j].get()
                C[i - 1, n] = partial_sum + a * b
                fifo_B[i + 1, j].put(b)
        with allo.meta_else():
            for n in range(N):
                partial_sum = fifo_A[i, j].get()
                a: float32 = A[i - 1, j]
                b = fifo_B[i, j].get()
                fifo_A[i, j + 1].put(partial_sum + a * b)
                fifo_B[i + 1, j].put(b)


def test_systolic():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    mod = df.build(top)
    if hls.is_available("vitis_hls"):
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_systolic()
