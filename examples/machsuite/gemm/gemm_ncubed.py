# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32
import numpy as np


def gemm(A: float32[64, 64], B: float32[64, 64]) -> float32[64, 64]:
    C: float32[64, 64] = 0.0
    for i, j in allo.grid(64, 64):
        for k in allo.reduction(64):
            C[i, j] += A[i, k] * B[k, j]
    return C


if __name__ == "__main__":
    s = allo.customize(gemm)
    mod = s.build()
    print(s.module)
