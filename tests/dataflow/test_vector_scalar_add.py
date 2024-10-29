# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32
import allo.dataflow as df
import numpy as np

Ty = int32
M = 64

@df.kernel(mapping=[1])
def core(A: Ty[M], B: Ty[M]):
    for i in range(M):
        # load data
        # out: Ty = A[i]
        # # compute
        # result: Ty = out + 1
        # # store result
        # B[i] = result
        B[i] = A[i] + 1

# A = np.random.rand(M).astype(np.float32)
# B = np.zeros(M).astype(np.float32)
# core(A, B)
# np.testing.assert_allclose(B, A + 1)
# print("PASSED!")

top = df.build(core, target="aie")
