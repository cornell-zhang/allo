# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import float32, int32


def top_durbin(size="tiny"):
    if size == "mini":
        N = 40
    elif size == "small":
        N = 120
    elif size == "medium":
        N = 400
    elif size is None or size == "tiny":
        N = 4

    def kernel_durbin(r: float32[N], y: float32[N]):
        y[0] = -r[0]
        beta: float32 = 1.0
        alpha: float32 = -r[0]

        for k in range(1, N):
            beta = (1 - alpha * alpha) * beta
            sum_: float32 = 0.0

            z: float32[N] = 0.0
            for i in range(k):
                sum_ = sum_ + r[k - i - 1] * y[i]

            alpha = -1.0 * (r[k] + sum_) / beta

            for i in range(k):
                z[i] = y[i] + alpha * y[k - i - 1]

            for i in range(k):
                y[i] = z[i]

            y[k] = alpha

    s0 = allo.customize(kernel_durbin)
    orig = s0.build("vhls")

    s = allo.customize(kernel_durbin)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_durbin()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "durbin", liveout_vars="v1")
