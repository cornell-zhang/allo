# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import allo.dataflow as df
from allo.ir.types import int16
from allo.memory import Layout
import numpy as np
from allo.backend.aie import is_available


def test_increase_decrease():
    LyA = Layout("S0R")

    Ty = int16
    M, N = 64, 64
    P0 = 4

    @df.region()
    def top(A: Ty[M, N], C: Ty[M, N], D: Ty[M, N], F: Ty[M, N]):
        @df.kernel(mapping=[P0], args=[A, C])
        def inc(local_A: Ty[M, N] @ LyA, local_C: Ty[M, N] @ LyA):
            local_C[:, :] = allo.add(local_A, 1)

        @df.kernel(mapping=[P0], args=[D, F])
        def dec(local_D: Ty[M, N] @ LyA, local_F: Ty[M, N] @ LyA):
            local_F[:, :] = allo.add(local_D, -1)

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(0, 64, (M, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
        F = np.zeros((M, N)).astype(np.int16)
        mod(A, C, A, F)
        np.testing.assert_allclose(C, A + 1, atol=1e-5)
        np.testing.assert_allclose(F, A - 1, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


# [NOTE] export ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH=0
def test_increase_decrease_more_arg():
    LyA = Layout("S0R")

    Ty = int16
    M, N = 64, 64
    P0 = 4

    @df.region()
    def top(
        A: Ty[M, N],
        B: Ty[M, N],
        C: Ty[M, N],
        D: Ty[M, N],
        E: Ty[M, N],
        F: Ty[M, N],
        G: Ty[M, N],
        H: Ty[M, N],
    ):
        @df.kernel(mapping=[P0], args=[A, B])
        def inc(local_A: Ty[M, N] @ LyA, local_B: Ty[M, N] @ LyA):
            local_B[:, :] = allo.add(local_A, 1)

        @df.kernel(mapping=[P0], args=[C, D])
        def incinc(local_C: Ty[M, N] @ LyA, local_D: Ty[M, N] @ LyA):
            local_D[:, :] = allo.add(local_C, 2)

        @df.kernel(mapping=[P0], args=[E, F])
        def dec(local_E: Ty[M, N] @ LyA, local_F: Ty[M, N] @ LyA):
            local_F[:, :] = allo.add(local_E, -1)

        @df.kernel(mapping=[P0], args=[G, H])
        def decdec(local_G: Ty[M, N] @ LyA, local_H: Ty[M, N] @ LyA):
            local_H[:, :] = allo.add(local_G, -2)

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(0, 64, (M, N)).astype(np.int16)
        B = np.zeros((M, N)).astype(np.int16)
        D = np.zeros((M, N)).astype(np.int16)
        F = np.zeros((M, N)).astype(np.int16)
        G = np.zeros((M, N)).astype(np.int16)
        mod(A, B, A, D, A, F, A, G)
        np.testing.assert_allclose(B, A + 1, atol=1e-5)
        np.testing.assert_allclose(D, A + 2, atol=1e-5)
        np.testing.assert_allclose(F, A - 1, atol=1e-5)
        np.testing.assert_allclose(G, A - 2, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_increase_decrease()
    test_increase_decrease_more_arg()
