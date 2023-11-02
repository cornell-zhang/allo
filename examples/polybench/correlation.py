# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import float32


def top_correlation(size="mini", epsilon=1e-5):
    if size == "mini" or size is None:
        M = 28
        N = 32
    elif size == "small":
        M = 80
        N = 100
    elif size == "medium":
        M = 240
        N = 260

    def kernel_correlation(
        data: float32[N, M], mean: float32[M], stddev: float32[M], corr: float32[M, M]
    ):
        # Compute mean
        for x in allo.grid(M):
            total: float32 = 0.0
            for k in allo.grid(N):
                total += data[k, x]
            mean[x] = total / N

        # Compute stddev
        for x in allo.grid(M):
            variance: float32 = 0.0
            for m in allo.grid(N):
                variance += (data[m, x] - mean[x]) * (data[m, x] - mean[x])
            stddev[x] = allo.sqrt(variance / N)
            # This is to avoid a division by zero situation
            # if stddev[x] <= epsilon:
            # stddev[x] = 1.0

        # Compute covariance
        cov: float32[M, M]
        for i in allo.grid(M):
            for j in allo.grid(M):
                covariance: float32 = 0.0
                for p in allo.grid(N):
                    covariance += (data[p, i] - mean[i]) * (data[p, j] - mean[j])
                cov[i, j] = covariance / N

        # Compute correlation
        for q, r in allo.grid(M, M):
            corr[q, r] = cov[q, r] / (stddev[q] * stddev[r])

    s0 = allo.customize(kernel_correlation)
    orig = s0.build("vhls")

    s = allo.customize(kernel_correlation)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_correlation()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "correlation", liveout_vars="v3")
