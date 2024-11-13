# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int8, int16, UInt
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 4, 4, 4
P0, P1 = M + 1, N + 1


@df.region()
def top():
    L2_A = df.array(df.pipe(dtype=UInt(M * 8), shape=(), depth=4), shape=(P0,))
    L2_B = df.array(df.pipe(dtype=UInt(N * 8), shape=(), depth=4), shape=(P1,))
    fifo_A = df.array(df.pipe(dtype=int8, shape=(), depth=4), shape=(M, N))
    fifo_B = df.array(df.pipe(dtype=int8, shape=(), depth=4), shape=(M, N))

    @df.kernel(mapping=[P0, P1])
    def gemm(A: int8[M, K], B: int8[K, N], C: int16[M, N]):
        i, j = df.get_pid()
        # periperals kernels
        with allo.meta_if(i == 0 and j == 0):
            for k in range(K):
                # pack data A
                packed_A: UInt(M * 8) = 0
                for m in range(M):
                    packed_A[m * 8 : (m + 1) * 8] = A[m, k]
                L2_A[1].put(packed_A)
                # pack data B
                packed_B: UInt(N * 8) = 0
                for n in range(N):
                    packed_B[n * 8 : (n + 1) * 8] = B[k, n]
                L2_B[1].put(packed_B)
        with allo.meta_elif(j == 0):
            # i > 0, the first column
            for k in range(K):
                a = L2_A[i].get()
                # unpack data
                fifo_A[i - 1, 0].put(a[8 * (i - 1) : 8 * i])
                with allo.meta_if(i < M):
                    L2_A[i + 1].put(a)
                # TODO: Fix meta matching
                with allo.meta_else():
                    pass
        with allo.meta_elif(i == 0):
            # j > 0, the first row
            for k in range(K):
                b = L2_B[j].get()
                fifo_B[0, j - 1].put(b[8 * (j - 1) : 8 * j])
                with allo.meta_if(j < N):
                    L2_B[j + 1].put(b)
                with allo.meta_else():
                    pass
        # main body
        with allo.meta_else():
            c: int16 = 0
            for k in range(K):
                a: int8 = fifo_A[i - 1, j - 1].get()
                b: int8 = fifo_B[i - 1, j - 1].get()
                c += a * b
                with allo.meta_if(j < N):
                    fifo_A[i - 1, j].put(a)
                with allo.meta_else():
                    pass
                with allo.meta_if(i < M):
                    fifo_B[i, j - 1].put(b)
                with allo.meta_else():
                    pass
            C[i - 1, j - 1] = c


def test_systolic():
    A = np.random.randint(0, 8, (M, K))
    B = np.random.randint(0, 8, (K, N))
    C = np.zeros((M, N), dtype=np.int16)
    mod = df.build(top)
    if hls.is_available("vitis_hls"):
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_systolic()
