# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, int32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np
import random

M, N, K = 2, 2, 2
# P0, P1 = M + 2, N + 2
P0, P1 = 1, 1 # no fifos so you can straight up access all matrices

# Create a sparse pattern mask for A (2:4 sparsity)
# For each block of 4 elements, randomly select 2 positions to be zero
MAXNZ = K//2
Knz = int(MAXNZ)


@df.region()
def top():
    # Define FIFO streams for A values and their column-indices.
    # Streams have depth (e.g. 4) and dimensions (Mt+2, Nt+2) to include boundaries.
    fifo_A = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(P0, P1))
    fifo_idx = df.array(df.pipe(dtype=int32, shape=(), depth=4), shape=(P0, P1))

    # Systolic kernel on one Mt-by-Nt tile of C.
    @df.kernel(mapping=[P0, P1])
    def semm(A: float32[M, Knz], A_in: int32[M, Knz], B: float32[K, N], C: float32[M, N]):
        i, j = df.get_pid()
        # skip corners
        with allo.meta_if(i in {0, M + 1} and j in {0, N + 1}):
            pass

        # left end we create fifo for A values and A nonzero column indices.
        with allo.meta_elif(j == 0):
            # for each nonzero in row (i-1) of A_tile:
            for k in range(Knz):
                fifo_A[i, j+1].put(A[i-1, k])
                fifo_idx[i, j+1].put(A_in[i-1, k])

        # drain
        with allo.meta_elif(i == 0 or i == M + 1 and j > 0):
            # don't need to collect at bottom or load at top because 
            # B direct access and is not draining at bottom
            pass 

        with allo.meta_elif(j == N + 1 and i > 0):
            # drain matrix A
            for k in range(K):
                a: float32 = fifo_A[i, j].get()

        # cherry pick values of B to multiply with A, forward A and
        with allo.meta_else():
            c_val: float32 = 0
            for k in range(Knz):
                a_val:  float32   = fifo_A[i, j].get()
                idx:    int32  = fifo_idx[i, j].get()
                b_val:  float32   = B[idx, j-1]
                c_val += a_val * b_val
                # Propagate A and index to the next PE to the right.
                fifo_A[i, j+1].put(a_val)
                fifo_idx[i, j+1].put(idx)
            # put accumulated result in PE
            C[i-1, j-1] = c_val


def create_sparse_matrix(m, k, sparsity_ratio):
    """Create a sparse matrix with the given sparsity ratio."""
    A_dense = np.zeros((m, k), dtype=np.float32)
    A = [] # nonzero values in A
    A_in = [] # indices for each row of nonzero values in A
    
    for i in range(m):
        # randomly select indices for non-zeros
        nnz_indices = random.sample(range(k), int(k * sparsity_ratio))
        row_vals = []
        row_inds = []
        for j in nnz_indices:
            val = random.uniform(0.1, 1.0)
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

    A = np.array(A, dtype=np.float32)
    Ain = np.array(Ain, dtype=np.int32)

    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

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

    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
    print("Dataflow Simulator Passed!")

    mod = df.build(top)
    if hls.is_available("vitis_hls"):
        C = np.zeros((M, N), dtype=np.float32)
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_sparse_systolic()
