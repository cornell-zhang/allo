# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, float64

N = 494  # Number of rows
L = 10  # Number of non zero entries in row


def ellpack(NZ: float64[N * L], cols: int32[N * L], vec: float64[N]) -> float64[N]:
    out: float64[N] = 0

    for i, j in allo.grid(N, L):
        if cols[j + i * L] != -1:  # For rows with fewer than L non zero entries
            out[i] += NZ[j + i * L] * vec[cols[j + i * L]]

    return out


if __name__ == "__main__":
    s = allo.customize(ellpack)
    mod = s.build()
    print(s.module)
