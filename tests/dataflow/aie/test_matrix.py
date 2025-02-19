# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
            B[pi * Mt: (pi + 1) * Mt, :] = allo.add(A[pi * Mt: (pi + 1) * Mt, :], 1)

    mod = df.build(top, target="aie", enable_tensor=True)
    A = np.random.randint(0, 100, (M, N)).astype(np.int32)
    B = np.zeros((M, N)).astype(np.int32)
    mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")


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
            C[pi * Mt: (pi + 1) * Mt, :] = allo.add(A[pi * Mt: (pi + 1) * Mt, :], B[pi * Mt: (pi + 1) * Mt, :])

    mod = df.build(top, target="aie", enable_tensor=True)
    A = np.random.randint(0, 100, (M, N)).astype(np.int32)
    B = np.random.randint(0, 100, (M, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A + B)
    print("PASSED!")


if __name__ == "__main__":
    _test_matrix_scalar_add()
    _test_matrix_matrix_add()
