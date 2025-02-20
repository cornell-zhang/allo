# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo.dataflow as df
from allo.ir.types import int32
import numpy as np
import allo


def _test_gemm():
    Ty = int32
    M, N, K = 16, 16, 16
    P0 = 2
    Mt = M // P0

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
            pi = df.get_pid()
            C[pi * Mt : (pi + 1) * Mt, :] = allo.matmul(
                A[pi * Mt : (pi + 1) * Mt, :], B
            )

    mod = df.build(top, target="aie")
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_gemm()
