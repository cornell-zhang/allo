# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import allo.dataflow as df
from allo.ir.types import int16
from allo.memory import Layout
import numpy as np


def _test_increase_decrease():
    LyA = Layout("S0R")

    Ty = int16
    M, N = 64, 64
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def inc(A: Ty[M, N] @ LyA, C: Ty[M, N] @ LyA):
            C[:, :] = allo.add(A, 1)

        @df.kernel(mapping=[P0])
        def dec(D: Ty[M, N] @ LyA, F: Ty[M, N] @ LyA):
            F[:, :] = allo.add(D, -1)

    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(0, 64, (M, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    F = np.zeros((M, N)).astype(np.int16)
    mod(A, C, A, F)
    np.testing.assert_allclose(C, A + 1, atol=1e-5)
    np.testing.assert_allclose(F, A - 1, atol=1e-5)
    print("PASSED!")

def _test_increase_decrease_more_arg():
    LyA = Layout("S0R")

    Ty = int16
    M, N = 64, 64
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def inc(A: Ty[M, N] @ LyA, B: Ty[M, N] @ LyA):
            B[:, :] = allo.add(A, 1)

        @df.kernel(mapping=[P0])
        def incinc(C: Ty[M, N] @ LyA, D: Ty[M, N] @ LyA):
            D[:, :] = allo.add(C, 2)

        @df.kernel(mapping=[P0])
        def dec(E: Ty[M, N] @ LyA, F: Ty[M, N] @ LyA):
            F[:, :] = allo.add(E, -1)

        @df.kernel(mapping=[P0])
        def decdec(G: Ty[M, N] @ LyA, H: Ty[M, N] @ LyA):
            H[:, :] = allo.add(G, -2)

    mod = df.build(top, target="aie-mlir")
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


if __name__ == "__main__":
    _test_increase_decrease()
    _test_increase_decrease_more_arg()

