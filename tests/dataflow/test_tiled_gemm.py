# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.ir.types import float32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 32, 32, 32
P0, P1 = 2, 2
Mt, Nt = M // P0, N // P1


@df.kernel(mapping=[P0, P1])
def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    pi, pj = df.get_pid()
    for i in range(pi * Mt, (pi + 1) * Mt):
        for j in range(pj * Nt, (pj + 1) * Nt):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]


def test_tiled_gemm():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    if hls.is_available("vitis_hls"):
        gemm(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), rtol=1e-5, atol=1e-5)
        print("Success!")


if __name__ == "__main__":
    test_tiled_gemm()
