# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import float32, int32


def top_syr2k(size="mini", alpha=1.5, beta=1.2):
    if size == "mini" or size is None:
        M = 20
        N = 30
    elif size == "small":
        M = 60
        N = 80
    elif size == "medium":
        M = 200
        N = 240

    def kernel_syr2k(A: float32[N, M], B: float32[N, M], C: float32[N, N]):
        for i in range(N):
            for j in range(i + 1):
                C[i, j] *= beta
            for k in range(M):
                for j in range(i + 1):
                    C[i, j] += A[j, k] * alpha * B[i, k] + B[j, k] * alpha * A[i, k]

    s0 = allo.customize(kernel_syr2k)
    orig = s0.build("vhls")

    s = allo.customize(kernel_syr2k)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_syr2k()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "syr2k", liveout_vars="v2")
