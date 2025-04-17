# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, int32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np
import random

M, N, K = 2, 2, 2
P0, P1 = M + 2, N + 2
# Create a sparse pattern mask for A (2:4 sparsity)
# For each block of 4 elements, randomly select 2 positions to be zero
MAXNZ = K/2
max_nzz = int(MAXNZ)


@df.region()
def top():
    fifo_A = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(P0, P1))
    fifo_Ain = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(P0, P1))
    fifo_B = df.array(df.pipe(dtype=float32, shape=(), depth=4), shape=(P0, P1))

    @df.kernel(mapping=[P0, P1])
    def semm(A: (float32)[M], Ain: (int32)[max_nzz], B: float32[K, N], C: float32[M, N]):
        i, j = df.get_pid()
        # periperals kernels
        with allo.meta_if(i in {0, M + 1} and j in {0, N + 1}):
            pass
        with allo.meta_elif(j == 0):
            # i > 0 and i <= M
            for k in range(len(A[i - 1])):
                idx, val = A[i - 1][k]
                fifo_Ain[i, j + 1].put(idx)
                fifo_A[i, j + 1].put(val)
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
            
            a_idx = fifo_Ain[i, j].get()
            a_val = fifo_A[i, j].get()
            
            # Get corresponding B values
            b_vals = [0.0] * K
            for k in range(K):
                b_vals[k] = fifo_B[i, j].get()
            
            # Compute product for matching index
            c += a_val * b_vals[a_idx]
            
            # Forward values
            fifo_Ain[i, j+1].put(a_idx)
            fifo_A[i, j+1].put(a_val)
            for k in range(K):
                fifo_B[i+1, j].put(b_vals[k])
                
            C[i - 1, j - 1] = c


def create_sparse_matrix(m, k, sparsity_ratio):
    """Create a sparse matrix with the given sparsity ratio."""
    A_dense = np.zeros((m, k), dtype=np.float32)
    A_sparse = []
    
    for i in range(m):
        row_nnz = []
        # Randomly select indices for non-zeros
        nnz_indices = random.sample(range(k), int(k * sparsity_ratio))
        for j in nnz_indices:
            val = random.uniform(0.1, 1.0)  # Random non-zero value
            A_dense[i, j] = val
            row_nnz.append((j, val))  # Store (index, value) pair
        A_sparse.append(row_nnz)
    
    return A_dense, A_sparse

def test_sparse_systolic():
    A = create_sparse_matrix(M, K, 0.5)
    # Create sparse pattern mask for A (2:4 sparsity)

    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B, C)
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
