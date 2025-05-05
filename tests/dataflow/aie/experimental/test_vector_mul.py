# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

Ly = Layout("S0")


def _test_vector_scalar_mul():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_mul
    Ty = float32
    M = 512

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M]):
            B[:] = allo.mul(A, 2)

    A = np.random.random(M).astype(np.float32)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        B = np.zeros(M).astype(np.float32)
        mod(A, B)
        np.testing.assert_allclose(B, A * 2, rtol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def _test_vector_vector_mul():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_vector_mul
    Ty = float32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M], C: Ty[M]):
            C[:] = allo.mul(A, B)

    A = np.random.random(M).astype(np.float32)
    B = np.random.random(M).astype(np.float32)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        C = np.zeros(M).astype(np.float32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A * B, rtol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    _test_vector_scalar_mul()
    _test_vector_vector_mul()
