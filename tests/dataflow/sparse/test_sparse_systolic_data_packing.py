# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, int128, index, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np
import random

M, N, K = 4, 4, 4  # feel free to change the dimensions of the matrices!
P0, P1 = M + 2, N + 2

NZ = int(K // 2)


@df.region()
def top():
    fifo_A: Stream[int32, 4][P0, P1]
    fifo_idx: Stream[int32, 4][P0, P1]
    fifo_B: Stream[int128, 4][P0, P1]

    @df.kernel(mapping=[P0, P1])
    def semm(A_nz: int32[M, NZ], A_in: int32[M, NZ], B: int32[K, N], C: int32[M, N]):
        """
        This kernel `semm` takes in the original sparse matrix A in a compressed format, with
        `A_nz` being only the nonzero values, and `A_in` being the column indices of the nonzero values.
        B is the dense matrix, and C is the desired solution of A * B, initialized to zeros.

        We do a row-wise multiply accumulate across the tiles of the systolic array, and modify and return C.
        """
        i, j = df.get_pid()
        # periperals kernels
        with allo.meta_if(i in {0, M + 1} and j in {0, N + 1}):
            pass
        with allo.meta_elif(j == 0):
            # i > 0
            for knz in range(NZ):
                fifo_A[i, j + 1].put(A_nz[i - 1, knz])
                fifo_idx[i, j + 1].put(A_in[i - 1, knz])
        with allo.meta_elif(i == 0):
            # pack the columns of B
            pack: int128 = 0
            for k in range(K):
                msb: index = (k + 1) * 32 - 1
                lsb: index = k * 32
                b: int32 = B[k, j - 1]
                pack[lsb:msb] = b
            fifo_B[i + 1, j].put(pack)
        # drain
        with allo.meta_elif(i == M + 1 and j > 0):
            b: int128 = fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(NZ):
                a: int32 = fifo_A[i, j].get()
                idx: int32 = fifo_idx[i, j].get()
        # main body
        with allo.meta_else():
            # multiply accumulate
            c: int32 = 0
            # get packed data only once
            b_packed: int128 = fifo_B[i, j].get()
            # process nonzero elements
            for k in range(NZ):
                a: int32 = fifo_A[i, j].get()
                idx: int32 = fifo_idx[i, j].get()
                # unpacking column B to get
                # corresponding values to the sparse matrix
                msb: index = (idx + 1) * 32 - 1
                lsb: index = idx * 32
                b: int32 = b_packed[lsb:msb]
                c += a * b
                # forward A data
                fifo_A[i, j + 1].put(a)
                fifo_idx[i, j + 1].put(idx)
            # forward column B data once row A is processed
            fifo_B[i + 1, j].put(b_packed)
            C[i - 1, j - 1] = c


def create_sparse_matrix(m, k, sparsity_ratio):
    """Create a sparse matrix with the given sparsity ratio."""
    A = np.zeros((m, k), dtype=np.int32)  # original sparse matrix
    A_nz = []  # nonzero values in A
    A_in = []  # indices for each row of nonzero values in A

    for i in range(m):
        # randomly select indices for non-zeros
        nnz_indices = random.sample(range(k), int(k * sparsity_ratio))
        row_vals = []
        row_inds = []
        for j in nnz_indices:
            val = random.randint(1, 10)
            A[i, j] = val
            row_vals.append(val)
            row_inds.append(j)
        A_nz.append(row_vals)
        A_in.append(row_inds)

    # input print statements
    print("\nOriginal matrix A:")
    print(A)
    print("\nCompressed representation:")
    print("A values:", A_nz)
    print("A indices:", A_in)

    return A, A_nz, A_in


def test_sparse_systolic():
    # A is 2:4 sparsity, A_nz is the nonzero values,
    # and A_in are the column indices indicating non-zero values
    A, A_nz, A_in = create_sparse_matrix(M, K, 0.5)

    A_nz = np.array(A_nz, dtype=np.int32)
    A_in = np.array(A_in, dtype=np.int32)

    B = np.random.randint(1, 10, size=(K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)

    # dataflow print statements
    print("\n=== Test Matrices ===")
    print("Matrix B:")
    print(B)
    print("\nInitial Matrix C:")
    print(C)

    print("\n=== Running Simulator ===")
    sim_mod = df.build(top, target="simulator")
    sim_mod(A_nz, A_in, B, C)

    print("\nFinal Result Matrix C:")
    print(C)
    print("\nExpected Result (numpy.dot):")
    expected = np.dot(A, B)
    print(expected)

    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
    print("Dataflow Simulator Passed!")

    mod = df.build(top)
    if hls.is_available("vitis_hls"):
        s = df.customize(top)

        s.partition("top:A_nz", dim=1, factor=2)
        s.partition("top:A_in", dim=1, factor=2)
        s.partition("top:B", dim=2, factor=2)
        s.partition("top:C", dim=0, factor=2)

        mod = s.build(target="vitis_hls", mode="hw_emu", project="systolic_rw_semm.prj")
        C = np.zeros((M, N), dtype=np.int32)
        mod(A_nz, A_in, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_sparse_systolic()
