# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import allo
from allo.ir.types import int32, float32


def test_pybind11():
    mod = allo.IPModule(
        top="gemm",
        headers=["gemm.h"],
        impls=["gemm.cpp"],
        signature=["float32[16, 16]", "float32[16, 16]", "float32[16, 16]"],
        link_hls=False,
    )
    a = np.random.random((16, 16)).astype(np.float32)
    b = np.random.random((16, 16)).astype(np.float32)
    c = np.zeros((16, 16)).astype(np.float32)
    mod(a, b, c)
    np.testing.assert_allclose(np.matmul(a, b), c, atol=1e-6)
    print("Passed!")


def test_shared_lib():
    vadd = allo.IPModule(
        top="vadd",
        headers=["vadd.h"],
        impls=["vadd.cpp"],
        signature=["int32[32]", "int32[32]", "int32[32]"],
        link_hls=False,
    )

    def kernel(A: int32[32], B: int32[32]) -> int32[32]:
        C: int32[32] = 0
        vadd(A, B, C)
        return C

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(0, 100, (32,)).astype(np.int32)
    np_B = np.random.randint(0, 100, (32,)).astype(np.int32)
    allo_C = mod(np_A, np_B)
    np.testing.assert_allclose(np_A + np_B, allo_C, atol=1e-6)
    print("Passed!")


def test_lib_gemm():
    gemm = allo.IPModule(
        top="gemm",
        headers=["gemm.h"],
        impls=["gemm.cpp"],
        signature=["float32[16, 16]", "float32[16, 16]", "float32[16, 16]"],
        link_hls=False,
    )

    def kernel(A: float32[16, 16], B: float32[16, 16]) -> float32[16, 16]:
        C: float32[16, 16] = 0
        gemm(A, B, C)
        C = C + 1
        return C

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    a = np.random.random((16, 16)).astype(np.float32)
    b = np.random.random((16, 16)).astype(np.float32)
    c = mod(a, b)
    np.testing.assert_allclose(np.matmul(a, b) + 1, c, atol=1e-6)
    print("Passed!")


if __name__ == "__main__":
    pytest.main([__file__])
