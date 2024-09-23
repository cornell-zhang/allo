# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

Ty = float32
M, N, K = 16, 16, 16
P0, P1 = 2, 2
Mt, Nt = M // P0, N // P1


@df.kernel(mapping=[P0, P1])
def gemm0(A: Ty[M, K], B: Ty[K, N]):
    pi, pj = df.get_pid()
    pipe: Stream[Ty[Mt, Nt]] = df.pipe(src="gemm0", dst="gemm1")
    C_out: Ty[Mt, Nt] = 0
    for i in range(pi * Mt, (pi + 1) * Mt):
        for j in range(pj * Nt, (pj + 1) * Nt):
            c: Ty = 0
            for k in range(K // 2):
                c += A[i, k] * B[k, j]
            C_out[i - pi * Mt, j - pj * Nt] = c
    pipe.put(C_out)


@df.kernel(mapping=[P0, P1])
def gemm1(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
    pi, pj = df.get_pid()
    pipe: Stream[Ty[Mt, Nt]] = df.pipe(src="gemm0", dst="gemm1")
    C_out: Ty[Mt, Nt] = pipe.get()
    for i in range(pi * Mt, (pi + 1) * Mt):
        for j in range(pj * Nt, (pj + 1) * Nt):
            c: Ty = 0
            for k in range(K // 2, K):
                c += A[i, k] * B[k, j]
            C[i, j] = C_out[i - pi * Mt, j - pj * Nt] + c


def test_cooperative_gemm():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    top = df.build([gemm0, gemm1])
    if hls.is_available("vitis_hls"):
        top(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_cooperative_gemm()
