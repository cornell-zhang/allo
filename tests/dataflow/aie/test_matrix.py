# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int32, float32
import allo.dataflow as df
import numpy as np


def _test_matrix_scalar_add():
    Ty = int32
    M, N = 64, 64
    P0 = 4
    Mt = M // P0

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M, N], B: Ty[M, N]):
            pi = df.get_pid()
            B[pi * Mt : (pi + 1) * Mt, :] = allo.add(A[pi * Mt : (pi + 1) * Mt, :], 1)

    A = np.random.randint(0, 100, (M, N)).astype(np.int32)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        B = np.zeros((M, N)).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    sim_mod = df.build(top, target="simulator")
    B = np.zeros((M, N)).astype(np.int32)
    sim_mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("Dataflow Simulator Passed!")


def _test_matrix_matrix_add():
    Ty = int32
    M, N = 64, 64
    P0 = 4
    Mt = M // P0

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M, N], B: Ty[M, N], C: Ty[M, N]):
            pi = df.get_pid()
            C[pi * Mt : (pi + 1) * Mt, :] = allo.add(
                A[pi * Mt : (pi + 1) * Mt, :], B[pi * Mt : (pi + 1) * Mt, :]
            )

    A = np.random.randint(0, 100, (M, N)).astype(np.int32)
    B = np.random.randint(0, 100, (M, N)).astype(np.int32)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        C = np.zeros((M, N)).astype(np.int32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A + B)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
    sim_mod = df.build(top, target="simulator")
    C = np.zeros((M, N)).astype(np.int32)
    sim_mod(A, B, C)
    np.testing.assert_allclose(C, A + B)
    print("Dataflow Simulator Passed!")


if __name__ == "__main__":
    _test_matrix_scalar_add()
    _test_matrix_matrix_add()
