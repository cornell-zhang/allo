# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import int16
import allo.dataflow as df


def _test_store_slice():

    Ty = int16
    N = 32
    Pk = 4

    @df.region()
    def top_v1():

        @df.kernel(mapping=[1])
        def core(A: Ty[N], B: Ty[Pk, N]):
            for i in range(Pk):
                B[i, :] = A

    A = np.random.randint(0, 64, (N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    mod = df.build(top_v1, target="aie")
    mod(A, B)
    np.testing.assert_allclose(A, B[0, :], atol=1e-5)
    np.testing.assert_allclose(A, B[1, :], atol=1e-5)
    np.testing.assert_allclose(A, B[2, :], atol=1e-5)
    np.testing.assert_allclose(A, B[3, :], atol=1e-5)
    print("PASSED!")

    @df.region()
    def top_v2():

        @df.kernel(mapping=[1])
        def core(A: Ty[N], B: Ty[Pk, N]):
            for i in range(0, Pk, 2):
                B[i, :] = allo.add(A, 1)
            for i in range(1, Pk, 2):
                B[i, :] = allo.add(A, -1)

    mod = df.build(top_v2, target="aie")
    mod(A, B)
    np.testing.assert_allclose(A + 1, B[0, :], atol=1e-5)
    np.testing.assert_allclose(A - 1, B[1, :], atol=1e-5)
    np.testing.assert_allclose(A + 1, B[2, :], atol=1e-5)
    np.testing.assert_allclose(A - 1, B[3, :], atol=1e-5)
    print("PASSED!")


def _test_load_slice():

    Ty = int16
    N = 32
    Pk = 4

    @df.region()
    def top():

        @df.kernel(mapping=[1])
        def core(A: Ty[Pk, N], B: Ty[Pk, N]):
            for i in range(Pk):
                B[i, :] = A[i, :]

    A = np.random.randint(0, 64, (Pk, N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    mod = df.build(top, target="aie")
    mod(A, B)
    np.testing.assert_allclose(A, B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_store_slice()
    _test_load_slice()
