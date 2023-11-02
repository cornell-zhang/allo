# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import float32, int32


def top_jacobi_2d(size="mini"):
    if size == "mini" or size is None:
        TSTEPS = 20
        N = 30
    elif size == "small":
        TSTEPS = 40
        N = 90
    elif size == "medium":
        TSTEPS = 100
        N = 250

    def kernel_jacobi_2d(A: float32[N, N], B: float32[N, N]):
        for m in range(TSTEPS):
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    B[i, j] = 0.2 * (
                        A[i, j] + A[i, j - 1] + A[i, j + 1] + A[i + 1, j] + A[i - 1, j]
                    )

            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    A[i, j] = 0.2 * (
                        B[i, j] + B[i, j - 1] + B[i, j + 1] + B[i + 1, j] + B[i - 1, j]
                    )

    s0 = allo.customize(kernel_jacobi_2d)
    orig = s0.build("vhls")

    s = allo.customize(kernel_jacobi_2d)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_jacobi_2d()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "jacobi_2d", liveout_vars="v0,v1")
