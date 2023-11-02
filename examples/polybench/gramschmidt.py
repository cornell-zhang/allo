# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import float32, int32


def top_gramschmidt(size="tiny"):
    if size == "mini":
        M = 20
        N = 30
    elif size == "small":
        M = 60
        N = 80
    elif size == "medium":
        M = 200
        N = 240
    elif size is None or size == "tiny":
        M = 2
        N = 3

    def kernel_gramschmidt(A: float32[M, N], Q: float32[M, N], R: float32[N, N]):
        for k in range(N):
            nrm: float32 = 0.0
            for i in range(M):
                nrm += A[i, k] * A[i, k]
            R[k, k] = allo.sqrt(nrm)

            for i in range(M):
                Q[i, k] = A[i, k] / R[k, k]

            for j in range(k + 1, N):
                R[k, j] = 0.0
                for i in range(M):
                    R[k, j] += Q[i, k] * A[i, j]

                for i in range(M):
                    A[i, j] -= Q[i, k] * R[k, j]

    s0 = allo.customize(kernel_gramschmidt)
    orig = s0.build("vhls")

    s = allo.customize(kernel_gramschmidt)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_gramschmidt(size="tiny")
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "gramschmidt", liveout_vars="v0,v1,v2")
