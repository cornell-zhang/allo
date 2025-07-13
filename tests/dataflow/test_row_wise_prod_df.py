# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, int32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np
import random

M, N, K = 4, 4, 4

MAXNZ = K//2
Knz = int(MAXNZ)

@df.region()
def top():
    @df.kernel(mapping=[1])
    def row_wise_prod(A: float32[M, Knz], A_in: int32[M, Knz], B: float32[K, N], C: float32[M, N]):
        for rA in range(M):
            for cB in range(N):
                for idx in range(Knz):
                    i: int32 = A_in[rA, idx]
                    C[rA, cB] += A[rA, idx] * B[i, cB]


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
    print("A values:\n", A)
    print("A indices:\n", A_in)

    return A_dense, A, A_in

def test_sparse_row_wise():
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

    mod = df.build(top, target="vitis_hls", mode="hw_emu", project="rwp_df.prj")
    c_mod = df.build(top, target="vitis_hls", mode="cysn", project="rwp.prj")
    c_mod()
    
    print("\nFinal Result Matrix C:")
    print(C)
    print("\nExpected Result (numpy.dot):")
    expected = np.dot(A_dense, B)
    print(expected)

    np.testing.assert_allclose(C, np.dot(A_dense, B), atol=1e-5)
    print("Dataflow Simulator Passed!")

    mod = df.build(top)
    if hls.is_available("vitis_hls"):
        C = np.zeros((M, N), dtype=np.float32)
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_sparse_row_wise()
