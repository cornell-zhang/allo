# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import float32


def top_covariance(size="mini"):
    if size == "mini" or size is None:
        M = 28
        N = 32
    elif size == "small":
        M = 80
        N = 100
    elif size == "medium":
        M = 240
        N = 260

    def kernel_covariance(data: float32[N, M], mean: float32[M], cov: float32[M, M]):
        # Compute mean
        for x in allo.grid(M):
            total: float32 = 0.0
            for k in allo.grid(N):
                total += data[k, x]
            mean[x] = total / N

        # Compute covariance
        for i, j in allo.grid(M, M):
            covariance: float32 = 0.0
            for p in allo.grid(N):
                covariance += (data[p, i] - mean[i]) * (data[p, j] - mean[j])
            cov[i, j] = covariance / (N - 1)

    s0 = allo.customize(kernel_covariance)
    orig = s0.build("vhls")

    s = allo.customize(kernel_covariance)
    s.split("i", factor=2)
    s.reorder("j", "i.inner", "i.outer")
    s.pipeline("i.inner")
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_covariance()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "covariance", liveout_vars="v2")
