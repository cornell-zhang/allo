# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.ir.types import float32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 32, 32, 32
P0, P1 = 2, 2
Mt, Nt = M // P0, N // P1


@df.region()
def inner(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
        pi, pj = df.get_pid()
        for i in range(pi * Mt, (pi + 1) * Mt):
            for j in range(pj * Nt, (pj + 1) * Nt):
                for k in range(K):
                    local_C[i, j] += local_A[i, k] * local_B[k, j]


@df.region()
def top(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    @df.kernel(mapping=[2], args=[A, B, C])
    def wrapper(local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
        inner(local_A, local_B, local_C)


def test_hierachical_function():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    # sim_mod = df.build(top, target="simulator")
    # sim_mod(A, B, C)
    # np.testing.assert_allclose(C, np.dot(A, B), rtol=1e-5)
    # print("Dataflow Simulator Passed!")

    mod = df.build(top)
    print(mod.module)
    if hls.is_available("vitis_hls"):
        C = np.zeros((M, N), dtype=np.float32)
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), rtol=1e-5, atol=1e-5)
        print("Success!")


if __name__ == "__main__":
    test_hierachical_function()
