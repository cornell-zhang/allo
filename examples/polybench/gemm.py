# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import int32


def top_gemm(size="mini", beta=0.1):
    if size == "mini" or size is None:
        P = 20
        R = 25
        Q = 30
    elif size == "small":
        P = 60
        R = 70
        Q = 80
    elif size == "medium":
        P = 200
        R = 220
        Q = 240

    def kernel_gemm(A: int32[P, Q], B: int32[Q, R], C: int32[P, R]):
        out_AB: int32[P, R] = 0
        for i0, j0, k0 in allo.grid(P, R, Q):
            out_AB[i0, j0] += A[i0, k0] * B[k0, j0]
        for i1, j1 in allo.grid(P, R):
            C[i1, j1] = beta * C[i1, j1] + out_AB[i1, j1]

    s0 = allo.customize(kernel_gemm)
    orig = s0.build("vhls")

    s = allo.customize(kernel_gemm)
    s.unroll("k0")
    s.pipeline("j1")
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_gemm()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "gemm", liveout_vars="v2")
