# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32
import allo.dataflow as df
import numpy as np

Ty = int32
M = 1024

def test_vector_scalar_add():
    @df.kernel(mapping=[1])
    def core(A: Ty[M], B: Ty[M]):
        for i in range(M):
            B[i] = A[i] + 1

    top = df.build(core, target="aie")
    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)
    top(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")

def test_vector_scalar_mul():
    @df.kernel(mapping=[1])
    def core(A: Ty[M], B: Ty[M]):
        for i in range(M):
            B[i] = A[i] * 2

    top = df.build(core, target="aie")
    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)
    top(A, B)
    np.testing.assert_allclose(B, A * 2)
    print("PASSED!")

if __name__ == "__main__":
    test_vector_scalar_add()
    test_vector_scalar_mul()
