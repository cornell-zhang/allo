# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import float32, int32


def top_ludcmp(size="tiny"):
    if size == "mini":
        N = 40
    elif size == "small":
        N = 120
    elif size == "medium":
        N = 400
    elif size is None or size == "tiny":
        N = 4

    def kernel_ludcmp(A: float32[N, N], b: float32[N], x: float32[N]):
        cst_neg_1: int32 = -1
        # LU decomposition of A
        for i in range(N):
            for j in range(i):
                w: float32 = A[i, j]
                for k in range(j):
                    w -= A[i, k] * A[k, j]
                A[i, j] = w / A[j, j]

            for j in range(i, N):
                w: float32 = A[i, j]
                for k in range(i):
                    w -= A[i, k] * A[k, j]
                A[i, j] = w

        y: float32[N]
        for m in range(N):
            y[m] = 0

        # Finding solution for LY = b
        for i in range(N):
            alpha: float32 = b[i]
            for j in range(i):
                alpha -= A[i, j] * y[j]
            y[i] = alpha

        # Finding solution for Ux = y
        for i in range(N - 1, cst_neg_1, cst_neg_1):
            alpha: float32 = y[i]
            for j in range(i + 1, N):
                alpha -= A[i, j] * x[j]
            x[i] = alpha / A[i, i]

    s0 = allo.customize(kernel_ludcmp)
    orig = s0.build("vhls")

    s = allo.customize(kernel_ludcmp)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_ludcmp("tiny")
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "ludcmp", liveout_vars="v2")
