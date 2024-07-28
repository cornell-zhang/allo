# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np
import allo
from allo.ir.types import int32, float32
import allo.backend.hls as hls


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


def test_extern_c():
    vadd = allo.IPModule(
        top="vadd",
        headers=["vadd_extern.h"],
        impls=["vadd_extern.cpp"],
        signature=["int32[32]", "int32[32]", "int32[32]"],
        link_hls=False,
    )
    np_A = np.random.randint(0, 100, (32,)).astype(np.int32)
    np_B = np.random.randint(0, 100, (32,)).astype(np.int32)
    np_C = np.zeros((32,), dtype=np.int32)
    vadd(np_A, np_B, np_C)
    np.testing.assert_allclose(np_A + np_B, np_C, atol=1e-6)
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


def test_scalar():
    vadd_int = allo.IPModule(
        top="vadd_int",
        headers=["vadd_int.h"],
        impls=["vadd_int.cpp"],
        signature=["int32[32]", "int32[32]", "int32"],
        link_hls=False,
    )

    def kernel(A: int32[32]) -> int32[32]:
        B: int32[32] = 0
        vadd_int(A, B, 5)
        return B

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(0, 100, (32,)).astype(np.int32)
    allo_C = mod(np_A)
    np.testing.assert_allclose(np_A + 5, allo_C, atol=1e-6)
    print("Passed!")


def test_scalar_pybind():
    vadd_int = allo.IPModule(
        top="vadd_int",
        headers=["vadd_int.h"],
        impls=["vadd_int.cpp"],
        signature=["int32[32]", "int32[32]", "int32"],
        link_hls=False,
    )

    np_A = np.random.randint(0, 100, (32,)).astype(np.int32)
    np_B = np.zeros((32,), dtype=np.int32)
    vadd_int(np_A, np_B, 5)
    np.testing.assert_allclose(np_A + 5, np_B, atol=1e-6)
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

    if hls.is_available():
        hls_mod = s.build(target="vivado_hls", mode="debug", project="gemm_ext.prj")
        print(hls_mod)
        hls_mod()
    else:
        print("Vivado HLS not found, skipping...")

    if hls.is_available("vitis_hls"):
        hls_mod = s.build(
            target="vitis_hls", mode="sw_emu", project="gemm_ext_vitis.prj"
        )
        print(hls_mod)
        hls_mod()
    else:
        print("Vitis HLS not found, skipping...")


@pytest.mark.skipif(not hls.is_available(), reason="Vivado HLS not found")
def test_systolic_stream():
    M, N, K = 2, 2, 2
    sa = allo.IPModule(
        "systolic_array",
        headers=["sa.h"],
        impls=["sa.cpp"],
        signature=[f"int8[{M}, {K}]", f"int8[{K}, {N}]", f"int16[{M}, {N}]"],
        link_hls=True,
    )

    A = np.random.randint(-8, 8, (M, K)).astype(np.int8)
    B = np.random.randint(-8, 8, (K, N)).astype(np.int8)
    allo_D = np.zeros((M, N), dtype=np.int16)
    sa(A, B, allo_D)
    np_D = A @ B
    np.testing.assert_allclose(allo_D, np_D, atol=1e-3)
    print("Pass!")


if __name__ == "__main__":
    pytest.main([__file__])
