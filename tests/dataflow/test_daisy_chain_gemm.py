# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int8, int16, UInt
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 4, 4, 4
P0, P1 = M + 2, N + 2


@df.region()
def top():
    L2_A = df.array(df.pipe(dtype=UInt(M * 16), shape=(), depth=4), shape=(P0 - 1,))
    L2_B = df.array(df.pipe(dtype=UInt(N * 16), shape=(), depth=4), shape=(P1 - 1,))

    L1_C = df.array(df.pipe(dtype=UInt(M * 16), shape=(), depth=4), shape=(M, N))
    L2_C = df.array(df.pipe(dtype=UInt(M * 16), shape=(), depth=4), shape=(N,))

    fifo_A = df.array(df.pipe(dtype=int16, shape=(), depth=4), shape=(M, N))
    fifo_B = df.array(df.pipe(dtype=int16, shape=(), depth=4), shape=(M, N))

    @df.kernel(mapping=[P0, P1])
    def gemm(A: int16[M, K], B: int16[K, N], C: int16[M, N]):
        i, j = df.get_pid()
        # periperals kernels
        with allo.meta_if(i == 0 and j == 0):
            for k in range(K):
                # pack data A
                packed_A: UInt(M * 16) = 0
                for m in range(M):
                    packed_A[m * 16 : (m + 1) * 16] = A[m, k]
                L2_A[1].put(packed_A)
                # pack data B
                packed_B: UInt(N * 16) = 0
                for n in range(N):
                    packed_B[n * 16 : (n + 1) * 16] = B[k, n]
                L2_B[1].put(packed_B)

        with allo.meta_elif(i == P0 - 1 and j == P1 - 1):
            for n in range(N):
                packed_C = L2_C[N - 1].get()
                for m in range(M):
                    C[m, n] = packed_C[m * 16 : (m + 1) * 16]

        with allo.meta_elif(i in {0, P0 - 1} and j in {0, P1 - 1}):
            pass

        with allo.meta_elif(j == 0):
            # i > 0, the first column
            for k in range(K):
                a = L2_A[i].get()
                # unpack data
                fifo_A[i - 1, 0].put(a[16 * (i - 1) : 16 * i])
                with allo.meta_if(i < M):
                    L2_A[i + 1].put(a)
                # TODO: Fix meta matching
                with allo.meta_else():
                    pass

        with allo.meta_elif(i == 0):
            # j > 0, the first row
            for k in range(K):
                b = L2_B[j].get()
                fifo_B[0, j - 1].put(b[16 * (j - 1) : 16 * j])
                with allo.meta_if(j < N):
                    L2_B[j + 1].put(b)
                with allo.meta_else():
                    pass

        with allo.meta_elif(i == P0 - 1):
            c_C = L1_C[i - 2, N - j].get()
            L2_C[j - 1].put(c_C)
            with allo.meta_if(j != 1):
                for ind in range(j - 1):
                    L2_C[j - 1].put(L2_C[j - 2].get())
            with allo.meta_else():
                pass

        with allo.meta_elif(j == P1 - 1):
            pass

        # main body
        with allo.meta_else():
            c: int16 = 0
            for k in range(K):
                a: int16 = fifo_A[i - 1, j - 1].get()
                b: int16 = fifo_B[i - 1, j - 1].get()
                c += a * b
                with allo.meta_if(j < N):
                    fifo_A[i - 1, j].put(a)
                with allo.meta_if(i < M):
                    fifo_B[i, j - 1].put(b)
                with allo.meta_else():
                    pass

            with allo.meta_if(i == 1):
                packed_tmp: UInt(M * 16) = 0
            with allo.meta_else():
                packed_tmp: UInt(M * 16) = L1_C[i - 2, j - 1].get()

            packed_c: UInt(M * 16) = 0
            for m in range(M):
                if m == i - 1:
                    packed_c[m * 16 : (m + 1) * 16] = c
                else:
                    packed_c[m * 16 : (m + 1) * 16] = packed_tmp[m * 16 : (m + 1) * 16]
            L1_C[i - 1, j - 1].put(packed_c)


def test_systolic():
    A = np.random.randint(0, 8, (M, K), dtype=np.int16)
    B = np.random.randint(0, 8, (K, N), dtype=np.int16)
    C = np.zeros((M, N), dtype=np.int16)
    mod = df.build(top)
    if hls.is_available("vitis_hls"):
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")

    mod = df.build(top, target="vitis_hls", mode="hw_emu", project="df-gemm-daisy.prj")
    if hls.is_available("vitis_hls"):
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_systolic()
