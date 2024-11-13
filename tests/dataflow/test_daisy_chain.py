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
    fifo_A = df.array(df.pipe(dtype=UInt(M * 8), shape=(), depth=4), shape=(P0, P1))
    fifo_B = df.array(df.pipe(dtype=UInt(N * 8), shape=(), depth=4), shape=(P0, P1))

    @df.kernel(mapping=[P0, P1])
    def gemm(A: int8[M, K], B: int8[K, N], C: int16[M, N]):
        i, j = df.get_pid()
        # periperals kernels
        with allo.meta_if(i == 0 and j == 0):
            for k in range(K):
                packed_A: UInt(M * 8) = 0
                for m in range(M):
                    packed_A[m * 8 : (m + 1) * 8] = A[m, k]
                fifo_A[1, 0].put(packed_A)
                packed_B: UInt(N * 8) = 0
                for n in range(N):
                    packed_B[n * 8 : (n + 1) * 8] = B[k, n]
                fifo_B[0, 1].put(packed_B)
        with allo.meta_elif(j == 0):
            # i > 0
            for k in range(K):
                a = fifo_A[i, j].get()
                fifo_A[i, j + 1].put(a)
                with allo.meta_if(i < M):
                    fifo_A[i + 1, j].put(a)
                # TODO: Fix meta matching
                with allo.meta_else():
                    pass
        with allo.meta_elif(i == 0):
            # j > 0
            for k in range(K):
                b = fifo_B[i, j].get()
                fifo_B[i + 1, j].put(b)
                with allo.meta_if(j < N):
                    fifo_B[i, j + 1].put(b)
                with allo.meta_else():
                    pass
        # main body
        with allo.meta_else():
            c: int8 = 0
            for k in range(K):
                a: UInt(M * 8) = fifo_A[i, j].get()
                b: UInt(N * 8) = fifo_B[i, j].get()
                local_a: int8 = a[(i - 1) * 8 : i * 8]
                local_b: int8 = b[(j - 1) * 8 : j * 8]
                c += local_a * local_b
                with allo.meta_if(j < N):
                    fifo_A[i, j + 1].put(a)
                with allo.meta_else():
                    pass
                with allo.meta_if(i < M):
                    fifo_B[i + 1, j].put(b)
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
