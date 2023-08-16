# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import allo
from allo.ir.types import int32
import numpy as np


def test_same():
    def same(A: int32[32, 32]) -> int32[32, 32]:
        return A

    s = allo.customize(same, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_A = np.zeros((32, 32)).astype(np.int32)
    np_A_allo = mod(np_A)
    np.testing.assert_allclose(np_A, np_A_allo, rtol=1e-5)


#  scalars are not transformed into tensors even if enable_tensor=True
def test_same_scalar():
    def same_scalar(A: int32) -> int32:
        return A

    s = allo.customize(same_scalar, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_A = 0
    np_A_allo = mod(np_A)
    np.testing.assert_allclose(np_A, np_A_allo, rtol=1e-5)


def test_outzero():
    def outzero() -> int32[32, 32]:
        C: int32[32, 32] = 0
        return C

    s = allo.customize(outzero, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_C = np.zeros((32, 32)).astype(np.int32)
    np_C_allo = mod()
    np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)


#  scalars are not transformed into tensors even if enable_tensor=True
def test_outzero_scalar():
    def outzero_scalar() -> int32:
        C: int32 = 0
        return C

    s = allo.customize(outzero_scalar, enable_tensor=True)
    print(s.module)

    mod = s.build()
    np_C = 0
    np_C_allo = mod()
    np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)


if __name__ == "__main__":
    test_same()
    test_same_scalar()
    test_outzero()
    test_outzero_scalar()
