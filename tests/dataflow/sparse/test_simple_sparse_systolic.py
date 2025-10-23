# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 4, 4, 4  # feel free to change the dimensions of the matrices!
P0, P1 = M + 2, N + 2


@df.region()
def top():
    fifo_A: Stream[int32, 4][P0, P1]
    fifo_B: Stream[int32, 4][P0, P1]

    @df.kernel(mapping=[P0, P1])
    def semm(A: int32[M, K], B: int32[K, N], C: int32[M, N]):
        """
        This kernel `semm` is a gemm implemented with a systolic array with an additional `if`
        check before multiply accumulating.
        """
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
                b: int32 = fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(K):
                a: int32 = fifo_A[i, j].get()
        # main body
        with allo.meta_else():
            c: int32 = 0
            for k in range(K):
                a: int32 = fifo_A[i, j].get()
                b: int32 = fifo_B[i, j].get()
                if a != 0:
                    c += a * b
                fifo_A[i, j + 1].put(a)
                fifo_B[i + 1, j].put(b)
            C[i - 1, j - 1] = c


def test_sparse_systolic():
    A_dense = np.random.rand(M, K).astype(np.int32)

    # create sparse pattern mask for A (2:4 sparsity)
    # for each block of 4 elements, randomly select 2 positions to be zero
    A = A_dense.copy()
    total_elements = M * K
    for block_start in range(0, total_elements, 4):
        # get the indices for current block
        block_indices = np.array(
            [
                (i // K, i % K)
                for i in range(block_start, min(block_start + 4, total_elements))
            ]
        )
        # r select 2 positions to set to zero
        zero_positions = np.random.choice(
            len(block_indices), size=min(2, len(block_indices) // 2), replace=False
        )
        for pos in zero_positions:
            i, j = block_indices[pos]
            A[i, j] = 0

    B = np.random.rand(K, N).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B, C)
    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
    print("Dataflow Simulator Passed!")

    if hls.is_available("vitis_hls"):
        s = df.customize(top)
        s.partition("top:A", dim=1, factor=2)
        s.partition("top:B", dim=2, factor=2)
        s.partition("top:C", dim=0, factor=2)

        mod = s.build(target="vitis_hls", mode="hw_emu", project="simple_semm.prj")
        C = np.zeros((M, N), dtype=np.int32)
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_sparse_systolic()
