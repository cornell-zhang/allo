# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 32, 32, 32
P0, P1 = 2, 2
Mt, Nt = M // P0, N // P1


@df.region()
def top():
    @df.kernel(mapping=[P0, P1])
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        pi, pj = df.get_pid()
        for i in range(pi * Mt, (pi + 1) * Mt):
            for j in range(pj * Nt, (pj + 1) * Nt):
                for k in range(K):
                    C[i, j] += A[i, k] * B[k, j]


def test_wrap_void():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    mod = df.build(top)

    if hls.is_available("vitis_hls"):
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), rtol=1e-5, atol=1e-5)
        print("Passed!")


def test_wrap_nonvoid():
    M, N = 4, 4

    def matrix_add(A: float32[M, N]) -> float32[M, N]:
        B: float32[M, N]
        for i, j in allo.grid(M, N, name="PE"):
            B[i, j] = A[i, j] + 1
        return B

    s = allo.customize(matrix_add)

    if hls.is_available("vitis_hls"):
        mod = s.build(target="vitis_hls", mode="csim")
        assert f"func.func @matrix_add(%arg0: memref<16xf32>) -> memref<16xf32>" in str(
            mod.module
        )
        print("Passed!")


if __name__ == "__main__":
    test_wrap_void()
    test_wrap_nonvoid()
