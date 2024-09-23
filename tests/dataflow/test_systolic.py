# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 4, 4, 4
P0, P1 = M + 1, N + 1


@df.kernel(mapping=[P0, P1])
def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    i, j = df.get_pid()
    in_A: Stream[float32] = df.pipe(src=(i, j - 1), dst=(i, j))
    in_B: Stream[float32] = df.pipe(src=(i - 1, j), dst=(i, j))
    out_A: Stream[float32] = df.pipe(src=(i, j), dst=(i, j + 1))
    out_B: Stream[float32] = df.pipe(src=(i, j), dst=(i + 1, j))
    # periperals kernels
    with allo.meta_if(i == 0 and j == 0):
        pass
    with allo.meta_elif(j == 0):
        # i > 0
        for k in range(K):
            out_A.put(A[i - 1, k])
    with allo.meta_elif(i == 0):
        # j > 0
        for k in range(K):
            out_B.put(B[k, j - 1])
    # main body
    with allo.meta_else():
        c: float32 = 0
        for k in range(K):
            a: float32 = in_A.get()
            b: float32 = in_B.get()
            c += a * b
            out_A.put(a)
            out_B.put(b)
        C[i - 1, j - 1] = c


def test_systolic():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    if hls.is_available("vitis_hls"):
        gemm(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_systolic()
