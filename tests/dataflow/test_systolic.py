# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 2, 2, 2
P0, P1 = M + 2, N + 2


@df.region()
def top():
    fifo_A: Stream[float32, 4][P0, P1]
    fifo_B: Stream[float32, 4][P0, P1]

    @df.kernel(mapping=[P0, P1])
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        i, j = df.get_pid()
        # periperals kernels
        with allo.meta_if(i in {0, M + 1} and j in {0, N + 1}):
            pass
        with allo.meta_elif(j == 0):
            # i > 0
            for k in range(K):
                fifo_A[i, j + 1].put(A[i - 1, k])
        with allo.meta_elif(i == 0):
            # j > 0
            for k in range(K):
                fifo_B[i + 1, j].put(B[k, j - 1])
        # drain
        with allo.meta_elif(i == M + 1 and j > 0):
            for k in range(K):
                b: float32 = fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(K):
                a: float32 = fifo_A[i, j].get()
        # main body
        with allo.meta_else():
            c: float32 = 0
            for k in range(K):
                a: float32 = fifo_A[i, j].get()
                b: float32 = fifo_B[i, j].get()
                c += a * b
                fifo_A[i, j + 1].put(a)
                fifo_B[i + 1, j].put(b)
            C[i - 1, j - 1] = c


def test_systolic():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B, C)
    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
    print("Dataflow Simulator Passed!")

    if hls.is_available("vitis_hls"):
        s = df.customize(top)
        s.partition("top:A", dim=1, factor=2)
        s.partition("top:B", dim=2, factor=2)
        s.partition("top:C", dim=0, factor=2)
        mod = s.build(target="vitis_hls", mode="hw_emu", project="systolic.prj")
        C = np.zeros((M, N), dtype=np.float32)
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_systolic()
