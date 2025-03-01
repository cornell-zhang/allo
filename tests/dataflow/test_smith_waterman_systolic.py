# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int8, int32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


W: int32 = 2  # gap penalty score per length, assuming a linear increament

M, N = 4, 4
P0, P1 = M + 2, N + 2

Sim: int32 = 3
Mis: int32 = -3


def smith_waterman_score_matrix(seqA, seqB):
    score_matrix = np.zeros((len(seqA) + 1, len(seqB) + 1), dtype=int)
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            if i == 0 or j == 0:
                score_matrix[i][j] = 0
            else:
                similarity = Sim if seqA[i - 1] == seqB[j - 1] else Mis
                score_matrix[i][j] = max(
                    0,
                    score_matrix[i - 1][j - 1] + similarity,
                    max([score_matrix[a, j] - 2 * (i - a) for a in range(i)]),
                    max([score_matrix[i, b] - 2 * (j - b) for b in range(j)]),
                )

    return score_matrix


@df.region()
def top():
    # FIFOs
    # FIFO A goes in increasing j direction
    # FIFO B goes in increasing i direction
    # FIFO C goes in increasing i and j direction
    fifo_A = df.array(df.pipe(dtype=int32, shape=(), depth=4), shape=(P0, P1))
    fifo_B = df.array(df.pipe(dtype=int32, shape=(), depth=4), shape=(P0, P1))
    fifo_C = df.array(df.pipe(dtype=int32, shape=(), depth=4), shape=(P0, P1))

    @df.kernel(mapping=[P0, P1])
    def Smith_Waterman(A: int8[M], B: int8[N], S: int32[P0 - 1, P1 - 1]):
        i, j = df.get_pid()
        # Does nothing for the two boundary PEs
        with allo.meta_if((i == 0 and j == P1 - 1) or (i == P0 - 1 and j == 0)):
            pass
        # Input
        with allo.meta_elif(i == 0 and j == 0):
            fifo_C[i + 1, j + 1].put(0)
        with allo.meta_elif(i == 0):
            fifo_B[i + 1, j].put(0)
            fifo_C[i + 1, j + 1].put(0)
        with allo.meta_elif(j == 0):
            fifo_A[i, j + 1].put(0)
            fifo_C[i + 1, j + 1].put(0)
        # Drain
        with allo.meta_elif(i == P0 - 1 and j == P1 - 1):
            fifo_C[i, j].get()
        with allo.meta_elif(i == P0 - 1):
            fifo_B[i, j].get()
            fifo_C[i, j].get()
        with allo.meta_elif(j == P1 - 1):
            fifo_A[i, j].get()
            fifo_C[i, j].get()
        # Compute
        with allo.meta_else():
            a = fifo_A[i, j].get()
            b = fifo_B[i, j].get()
            c = fifo_C[i, j].get()
            # Calculate the score for the current position
            aligning: int32 = c + (Sim if A[i - 1] == B[j - 1] else Mis)
            gap_A: int32 = a - W
            gap_B: int32 = b - W
            # Choose the maximum score
            score: int32 = max(max(0, aligning), max(gap_A, gap_B))
            S[i, j] = score
            fifo_A[i, j + 1].put(max(gap_A, score))
            fifo_B[i + 1, j].put(max(gap_B, score))
            fifo_C[i + 1, j + 1].put(score)


def test_systolic():
    possible_chars = np.array(["A", "C", "G", "T"], dtype="c")
    A = np.random.choice(possible_chars, size=M).view(np.int8)
    B = np.random.choice(possible_chars, size=N).view(np.int8)
    S = np.zeros((P0 - 1, P1 - 1), dtype=np.int32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B, S)
    S_1 = smith_waterman_score_matrix(A, B)
    np.testing.assert_equal(S[1:, 1:], S_1[1:, 1:])
    print("Dataflow Simulator Passed!")

    mod = df.build(top, target="vitis_hls", mode="hw", project="smith_waterman.prj")
    if hls.is_available("vitis_hls"):
        S = np.zeros((P0 - 1, P1 - 1), dtype=np.int32)
        mod(A, B, S)


if __name__ == "__main__":
    test_systolic()
