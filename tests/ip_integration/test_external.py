# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
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

    def top(A: float32[16, 16], B: float32[16, 16]) -> float32[16, 16]:
        C: float32[16, 16] = 0
        gemm(A, B, C)
        D = C + 1
        return D

    s = allo.customize(top)
    print(s.module)
    mod = s.build()
    a = np.random.random((16, 16)).astype(np.float32)
    b = np.random.random((16, 16)).astype(np.float32)
    c = mod(a, b)
    np.testing.assert_allclose(np.matmul(a, b) + 1, c, atol=1e-6)
    print("Passed!")

    if os.system(f"which vivado_hls >> /dev/null") == 0:
        hls_mod = s.build(target="vivado_hls", mode="debug", project="gemm_ext.prj")
        print(hls_mod)
        hls_mod()
    else:
        print("Vivado HLS not found, skipping...")

    if os.system(f"which vitis_hls >> /dev/null") == 0:
        hls_mod = s.build(
            target="vitis_hls", mode="sw_emu", project="gemm_ext_vitis.prj"
        )
        print(hls_mod)
        hls_mod()
    else:
        print("Vitis HLS not found, skipping...")


if __name__ == "__main__":
    pytest.main([__file__])
