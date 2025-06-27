# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

def _test_matrix_scalar_add():
    LyA = Layout("S0R")
    Ty = int32
    M, N = 64, 64
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M, N] @ LyA, B: Ty[M, N] @ LyA):
            B[:, :] = allo.add(A, 1)

    A = np.random.randint(0, 100, (M, N)).astype(np.int32)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        B = np.zeros((M, N)).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

if __name__ == "__main__":
    _test_matrix_scalar_add()
