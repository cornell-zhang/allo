# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

# M, N, K = 512, 512, 512
# Mt, Nt = 16, 16
M, N, K = 16, 16, 16
Mt, Nt = 4, 4
# M, N, K = 4, 4, 4
# Mt, Nt = 1, 1
P0, P1 = Mt + 2, Nt + 2


@df.region()
def top():
    fifo_A = df.array(df.pipe(dtype=int32, shape=(), depth=4), shape=(P0, P1))
    fifo_B = df.array(df.pipe(dtype=int32, shape=(), depth=4), shape=(P0, P1))

    @df.kernel(mapping=[P0, P1])
    def gemm(A: int32[M, K], B: int32[K, N], C: int32[M, N]):
        # A[Mt, K] * B[K, Nt] = C[Mt, Nt]
        i, j = df.get_pid()
        for m in range(M // Mt):
            for n in range(N // Nt):
                # periperals kernels
                with allo.meta_if(i in {0, Mt + 1} and j in {0, Nt + 1}):
                    pass
                with allo.meta_elif(j == 0):
                    # i > 0
                    for k in range(K):
                        fifo_A[i, j + 1].put(A[m * Mt + i - 1, k])
                with allo.meta_elif(i == 0):
                    # j > 0
                    for k in range(K):
                        fifo_B[i + 1, j].put(B[k, n * Nt + j - 1])
                # drain
                with allo.meta_elif(i == Mt + 1):
                    for k in range(K):
                        b: int32 = fifo_B[i, j].get()
                with allo.meta_elif(j == Nt + 1):
                    for k in range(K):
                        a: int32 = fifo_A[i, j].get()
                # main body
                with allo.meta_else():
                    c: int32 = 0
                    for k in range(K):
                        a: int32 = fifo_A[i, j].get()
                        b: int32 = fifo_B[i, j].get()
                        c += a * b
                        fifo_A[i, j + 1].put(a)
                        fifo_B[i + 1, j].put(b)
                    C[m * Mt + i - 1, n * Nt + j - 1] = c


def test_tiled_systolic():
    A = np.random.randint(0, 10, (M, K)).astype(np.int32)
    B = np.random.randint(0, 10, (K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    mod = df.build(top)
    if hls.is_available("vitis_hls"):
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_tiled_systolic()
