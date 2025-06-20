# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import allo.dataflow as df
from allo.ir.types import int32
from allo.memory import Layout
import numpy as np


def _test_gemm_2D():
    LyA = Layout("S0R")
    LyB = Layout("RS1")
    LyC = Layout("S0S1")

    TyI, TyO = int32, int32
    M, N, K = 64, 64, 64
    P0, P1 = 2, 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            ("chain", ["gemm_0_0", "gemm_0_1"]),
            ("chain", ["gemm_1_0", "gemm_1_1"]),
            ("bundle", ["gemm_0_0-gemm_0_1","gemm_1_0-gemm_1_1"]),
        ],
        profile=True,
        warmup=200,
        num_iters=1000,
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_gemm_2D()
