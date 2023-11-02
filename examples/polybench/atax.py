# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import float32


def top_atax(size="mini"):
    if size == "mini" or size is None:
        M = 38
        N = 42
    elif size == "small":
        M = 116
        N = 124
    elif size == "medium":
        M = 390
        N = 410

    def kernel_atax(A: float32[M, N], x: float32[N], y: float32[N]):
        out_Ax: float32[M] = 0
        for m in allo.grid(M):
            for r in allo.reduction(N):
                out_Ax[m] += A[m, r] * x[r]

        for n in allo.grid(N):
            for k in allo.reduction(M):
                y[n] += A[k, n] * out_Ax[k]

    s0 = allo.customize(kernel_atax)
    orig = s0.build("vhls")

    s = allo.customize(kernel_atax)
    s.split("m", factor=2)
    s.reorder("n", "m.inner", "m.outer")
    s.pipeline("m.inner")
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_atax()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "atax", liveout_vars="v2")
