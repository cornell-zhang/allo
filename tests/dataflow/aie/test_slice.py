# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import int16
import allo.dataflow as df
from allo.backend.aie import is_available


def test_store_slice():

    Ty = int16
    N = 32
    Pk = 4

    @df.region()
    def top_v1(A: Ty[N], B: Ty[Pk, N]):

        @df.kernel(mapping=[1], args=[A, B])
        def core(local_A: Ty[N], local_B: Ty[Pk, N]):
            for i in range(Pk):
                local_B[i, :] = local_A

    A = np.random.randint(0, 64, (N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    if is_available():
        mod = df.build(top_v1, target="aie")
        mod(A, B)
        np.testing.assert_allclose(A, B[0, :], atol=1e-5)
        np.testing.assert_allclose(A, B[1, :], atol=1e-5)
        np.testing.assert_allclose(A, B[2, :], atol=1e-5)
        np.testing.assert_allclose(A, B[3, :], atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    @df.region()
    def top_v2(A: Ty[N], B: Ty[Pk, N]):

        @df.kernel(mapping=[1], args=[A, B])
        def core(local_A: Ty[N], local_B: Ty[Pk, N]):
            for i in range(0, Pk, 2):
                local_B[i, :] = allo.add(local_A, 1)
            for i in range(1, Pk, 2):
                local_B[i, :] = allo.add(local_A, -1)

    if is_available():
        mod = df.build(top_v2, target="aie")
        mod(A, B)
        np.testing.assert_allclose(A + 1, B[0, :], atol=1e-5)
        np.testing.assert_allclose(A - 1, B[1, :], atol=1e-5)
        np.testing.assert_allclose(A + 1, B[2, :], atol=1e-5)
        np.testing.assert_allclose(A - 1, B[3, :], atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_load_slice():

    Ty = int16
    N = 32
    Pk = 4

    @df.region()
    def top(A: Ty[Pk, N], B: Ty[Pk, N]):

        @df.kernel(mapping=[1], args=[A, B])
        def core(local_A: Ty[Pk, N], local_B: Ty[Pk, N]):
            for i in range(Pk):
                local_B[i, :] = local_A[i, :]

    A = np.random.randint(0, 64, (Pk, N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)
    if is_available():
        mod = df.build(top, target="aie")
        mod(A, B)
        np.testing.assert_allclose(A, B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_store_slice()
    test_load_slice()
