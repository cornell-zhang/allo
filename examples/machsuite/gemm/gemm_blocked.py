# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32
import numpy as np

M, N, K = 1024, 1024, 1024
S = 8


def bbgemm(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
    C: int32[M, N] = 0

    i_max: int32 = 0
    j_max: int32 = 0
    k_max: int32 = 0
    sum_value: int32 = 0

    for i in range(0, M, S):
        i_max = i + S if i + S < M else M
        for j in range(0, N, S):
            j_max = j + S if j + S < N else N
            for k in range(0, K, S):
                k_max = k + S if k + S < K else K
                for ii in range(i, i_max):
                    for jj in range(j, j_max):
                        sum_value = 0
                        for kk in range(k, k_max):
                            sum_value += A[ii, kk] * B[kk, jj]
                        C[ii, jj] += sum_value
    return C


if __name__ == "__main__":
    s = allo.customize(bbgemm)
    mod = s.build()
    print(s.module)
