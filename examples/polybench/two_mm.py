# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import int32


def top_2mm(size="mini"):
    alpha = 0.1
    beta = 0.5

    if size == "mini" or size is None:
        P = 16  ## P = NI
        R = 18  ## R = NJ
        Q = 22  ## Q = NK
        S = 24  ## S = NL
    elif size == "small":
        P = 40
        R = 50
        Q = 70
        S = 80
    elif size == "medium":
        P = 180
        R = 190
        Q = 210
        S = 220

    def kernel_2mm(
        A: int32[P, Q], B: int32[Q, R], C: int32[R, S], D: int32[P, S]
    ) -> int32[P, S]:
        out_AB: int32[P, R] = 0
        for i0, j0, k0 in allo.grid(P, R, Q):
            out_AB[i0, j0] += A[i0, k0] * B[k0, j0]
        out_ABC: int32[P, S] = 0
        for i1, j1, k1 in allo.grid(P, S, R):
            out_ABC[i1, j1] += out_AB[i1, k1] * C[k1, j1]
        output: int32[P, S] = 0
        for i2, j2 in allo.grid(P, S):
            output[i2, j2] = out_ABC[i2, j2] * beta + D[i2, j2] * alpha
        return output

    s0 = allo.customize(kernel_2mm)
    orig = s0.build("vhls")

    s = allo.customize(kernel_2mm)
    s.compute_at("i0", "i1")
    s.compute_at("i1", "i2")
    s.unroll("k1")
    s.pipeline("j2")
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_2mm(size="mini")
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "2mm", liveout_vars="v4")
