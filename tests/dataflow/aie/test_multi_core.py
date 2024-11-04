# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, float32
import allo.dataflow as df
import numpy as np


def _test_vector_scalar_add():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_guide/section-2/section-2d
    #                |--------------------------------------------|
    #                v   v-------------------------v              v
    # shim tile <-> mem tile <-> comp tile0    comp tile1    comp tile2
    Ty = int32
    M = 48
    P0 = 3

    @df.kernel(mapping=[P0])
    def core(A: Ty[M], B: Ty[M]):
        for i in range(M // P0):
            B[i] = A[i] + 1

    top = df.build(core, target="aie")
    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)
    top(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")


def _test_vector_vector_add():
    #                  |--------------------------------------------|
    #                  v   v--------------------------v             v
    # shim tile <-> A mem tile 0 <-> comp tile0    comp tile1    comp tile2
    #       ^-----> B mem tile 1 <-------^------------^-------------^
    Ty = int32
    M = 1024
    P0 = 4

    @df.kernel(mapping=[P0])
    def core(A: Ty[M], B: Ty[M], C: Ty[M]):
        for i in range(M // P0):
            C[i] = A[i] + B[i]

    top = df.build(core, target="aie")
    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.random.randint(0, 100, M).astype(np.int32)
    C = np.zeros(M).astype(np.int32)
    top(A, B, C)
    np.testing.assert_allclose(C, A + B)
    print("PASSED!")


if __name__ == "__main__":
    _test_vector_scalar_add()
    _test_vector_vector_add()
