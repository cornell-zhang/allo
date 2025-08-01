# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import allo.dataflow as df
from allo.ir.types import int16
from allo.memory import Layout
import numpy as np

# RRxRS->RS
# RSxSR->RR
LyW1 = Layout("RS0")
LyW2 = Layout("S0R")


def _test_gemm_1D():
    LyA = Layout("S0R")

    Ty = int16
    M, N, K = 16, 16, 16
    P0 = 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N], C: Ty[M, N] @ LyA):
            C[:, :] = allo.matmul(A, B)

        @df.kernel(mapping=[P0])
        def mmeg(D: Ty[M, K] @ LyA, E: Ty[K, N], F: Ty[M, N] @ LyA):
            F[:, :] = allo.matmul(D, E)

    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    F = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C, A, B, F)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_gemm_1D()
