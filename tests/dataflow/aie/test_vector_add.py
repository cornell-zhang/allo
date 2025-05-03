# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16, int32, float32, bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

Ly = Layout("S0")


def _test_vector_scalar_add():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M]):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        B = np.zeros(M).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

def _test_vector_vector_add():
    # # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_vector_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M], C: Ty[M]):
            C[:] = allo.add(A, B)

    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.random.randint(0, 100, M).astype(np.int32)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        C = np.zeros(M).astype(np.int32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A + B)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
    
if __name__ == "__main__":
    _test_vector_scalar_add()
    _test_vector_vector_add()