# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32


def test_gemm_grid_for():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm, verbose=True)
    print(s.module)


if __name__ == "__main__":
    test_gemm_grid_for()
