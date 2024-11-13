# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 16, 16, 16
P0 = K + 2


@df.region()
def top():
    fifo_A = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(2, P0))
    fifo_B = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(2, P0))

    @df.kernel(mapping=[2, P0])
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        i, j = df.get_pid()
        with allo.meta_if(i == 0 and (j == 0 or j == P0 - 1)):
            pass
        # input
        with allo.meta_elif(i == 0):
            for _ in range(N):
                for k in range(K):
                    fifo_B[i + 1, j].put(B[k, j - 1])
        with allo.meta_elif(j == 0):
            for m in range(M):
                for k in range(K):
                    fifo_A[i, j + 1].put(A[m, k])
        # drain
        with allo.meta_elif(j == P0 - 1):
            for m in range(M):
                for _ in range(K):
                    a: float32 = fifo_A[i, j].get()
        # compute
        with allo.meta_else():
            for m in range(M):
                c: float32 = 0
                for _ in range(K):
                    a: float32 = fifo_A[i, j].get()
                    b: float32 = fifo_B[i, j].get()
                    c += a * b
                    fifo_A[i, j + 1].put(a)
                C[m, j - 1] = c


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
