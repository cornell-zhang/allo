# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, int32, int128, index
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np
import random

M, N, K = 4, 4, 4
P0, P1 = M + 2, N + 2

NZ = int(K // 2)

@df.region()
def top():
    fifo_A = df.array(df.pipe(dtype=int32, shape=(), depth=4), shape=(P0, P1))
    fifo_idx = df.array(df.pipe(dtype=int32, shape=(), depth=4), shape=(P0, P1))
    fifo_B = df.array(df.pipe(dtype=int128, shape=(), depth=4), shape=(P0, P1))
    
    @df.kernel(mapping=[P0, P1])
    def semm(A: int32[M, NZ], A_in: int32[M, NZ], B: int32[K, N], C: int32[M, N]):
        i, j = df.get_pid()
        # periperals kernels
        with allo.meta_if(i in {0, M + 1} and j in {0, N + 1}):
            pass
        with allo.meta_elif(j == 0):
            # i > 0
            for knz in range(NZ):
                fifo_A[i, j + 1].put(A[i - 1, knz])
                fifo_idx[i, j + 1].put(A_in[i - 1, knz])
        with allo.meta_elif(i == 0):
            # pass
            pack: int128 = 0
            for k in range(K):
                msb: index = (k + 1) * 32 - 1
                lsb: index = k * 32
                b: int32 = B[j - 1, k]
                pack[lsb : msb] = b
            fifo_B[i + 1, j].put(pack)
        # drain
        with allo.meta_elif(i == M + 1 and j > 0):
            for k in range(K):
                b: int128 = fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(NZ):
                a: int32 = fifo_A[i, j].get()
                idx: int32 = fifo_idx[i, j].get()
        # main body
        with allo.meta_else():
            c: int32 = 0
            for k in range(NZ):
                a: int32 = fifo_A[i, j].get()
                idx: int32 = fifo_idx[i, j].get()
                # unpacking
                b_packed: int128 = fifo_B[i, j].get()
                msb: index = (idx + 1) * 32 - 1
                lsb: index = idx * 32
                b: int32 = b_packed[lsb : msb]
                c += a * b
                fifo_A[i, j + 1].put(a)
                fifo_idx[i, j + 1].put(idx)
                fifo_B[i + 1, j].put(b_packed)
            C[i - 1, j - 1] = c


def create_sparse_matrix(m, k, sparsity_ratio):
    """Create a sparse matrix with the given sparsity ratio."""
    A_dense = np.zeros((m, k), dtype=np.int32)
    A = [] # nonzero values in A
    A_in = [] # indices for each row of nonzero values in A
    
    for i in range(m):
        # randomly select indices for non-zeros
        nnz_indices = random.sample(range(k), int(k * sparsity_ratio))
        row_vals = []
        row_inds = []
        for j in nnz_indices:
            val = random.randint(1, 10)  # Changed to generate integers
            A_dense[i, j] = val
            row_vals.append(val)
            row_inds.append(j)
        A.append(row_vals)
        A_in.append(row_inds)

    # input print statements
    print("\nDense matrix A:")
    print(A_dense)
    print("\nSparse representation:")
    print("A values:", A)
    print("A indices:", A_in)

    return A_dense, A, A_in

def test_sparse_systolic():
    A_dense, A, Ain = create_sparse_matrix(M, K, 0.5)
    # A_dense is 2:4 sparsity, A is the nonzero values, 
    # and Ain are the column indices indicating non-zero values

    A = np.array(A, dtype=np.int32)
    Ain = np.array(Ain, dtype=np.int32)

    B = np.random.randint(1, 10, size=(K, N)).astype(np.int32)  # Changed to generate integers
    C = np.zeros((M, N), dtype=np.int32)

    # dense print statements
    print("\n=== Test Matrices ===")
    print("Matrix B:")
    print(B)
    print("\nInitial Matrix C:")
    print(C)

    print("\n=== Running Simulator ===")
    sim_mod = df.build(top, target="simulator")
    sim_mod(A, Ain, B, C)
    
    print("\nFinal Result Matrix C:")
    print(C)
    print("\nExpected Result (numpy.dot):")
    expected = np.dot(A_dense, B)
    print(expected)

    np.testing.assert_allclose(C, np.dot(A_dense, B), atol=1e-5)
    print("Dataflow Simulator Passed!")

    mod = df.build(top)
    if hls.is_available("vitis_hls"):
        s = df.customize(top)

        s.partition("top:A", dim=1, factor=2)
        s.partition("top:A_in", dim=1, factor=2)
        s.partition("top:B", dim=0, factor=2)
        s.partition("top:C", dim=0, factor=2)

        # s.pipeline("semm", initiation_interval=1)
        # mod = s.build(target="vitis_hls", mode="csyn", project="ssemmscyn.prj")
        # mod()
        
        mod = s.build(target="vitis_hls", mode="hw_emu", project="ssemmhw.prj")
        C = np.zeros((M, N), dtype=np.int32)
        mod(A, Ain, B, C)
        np.testing.assert_allclose(C, np.dot(A_dense, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_sparse_systolic()
