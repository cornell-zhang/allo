import allo
from allo.ir.types import int8, int32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


W: int32 = 2  # gap penalty score per length, assuming a linear increament

max_size = 100
M, N = np.random.randint(30, max_size), np.random.randint(30, max_size)
P0, P1 = M + 1, N + 1

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


@df.kernel(mapping=[P0, P1])
def Smith_Waterman(A: int8[M], B: int8[N], S: int32[P0, P1]):
    i, j = df.get_pid()
    # using float32 for the score matrix to avoid overflow
    # or account for different scoring functions
    in_A: Stream[int32] = df.pipe(src=(i, j - 1), dst=(i, j))
    in_B: Stream[int32] = df.pipe(src=(i - 1, j), dst=(i, j))
    in_C: Stream[int32] = df.pipe(src=(i - 1, j - 1), dst=(i, j))
    out_B: Stream[int32] = df.pipe(src=(i, j), dst=(i + 1, j))
    out_A: Stream[int32] = df.pipe(src=(i, j), dst=(i, j + 1))
    out_C: Stream[int32] = df.pipe(src=(i, j), dst=(i + 1, j + 1))

    with allo.meta_if(i == 0 and j == 0):
        # Fill the first row and column with zeros
        out_A.put(0)
        out_B.put(0)
        out_C.put(0)
    with allo.meta_elif(i == 0):
        out_A.put(0)
        out_C.put(0)
    with allo.meta_elif(j == 0):
        out_B.put(0)
        out_C.put(0)
    with allo.meta_else():
        a = in_A.get()
        b = in_B.get()
        c = in_C.get()
        # Calculate the score for the current position
        aligning: int32 = c + (Sim if A[i - 1] == B[j - 1] else Mis)
        gap_A: int32 = a - W
        gap_B: int32 = b - W
        # Choose the maximum score
        score: int32 = max(max(0, aligning), max(gap_A, gap_B))
        S[i, j] = score
        out_A.put(max(gap_A, score))
        out_B.put(max(gap_B, score))
        out_C.put(score)


def test_systolic():
    possible_chars = np.array(["A", "C", "G", "T"], dtype="c")
    A = np.random.choice(possible_chars, size=M).view(np.int8)
    B = np.random.choice(possible_chars, size=N).view(np.int8)
    S = np.zeros((P0, P1), dtype=np.int32)
    print("awaiting hls...")
    if hls.is_available("vitis_hls"):
        Smith_Waterman(A, B, S)
        oracle = smith_waterman_score_matrix(A, B)
        np.testing.assert_array_equal(S, oracle)
        print("Passed!")


if __name__ == "__main__":
    for _ in range(10):
        test_systolic()
