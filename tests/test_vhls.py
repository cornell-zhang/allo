# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import allo
from allo.ir.types import int32, float32
import numpy as np
import allo.backend.hls as hls


@pytest.mark.parametrize("flatten", [True, False])
def test_io_buffer_gemm(flatten):
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    print(s.module)
    allo.passes.generate_input_output_buffers(s.top_func, flatten=flatten)
    print(s.module)
    mod = s.build()
    if not flatten:
        np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_C = np.matmul(np_A, np_B)
    else:
        np_A = np.random.randint(0, 10, size=(32 * 32)).astype(np.int32)
        np_B = np.random.randint(0, 10, size=(32 * 32)).astype(np.int32)
        np_C = np.matmul(np_A.reshape(32, 32), np_B.reshape(32, 32)).reshape(32 * 32)
    np_C_allo = mod(np_A, np_B)
    np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)
    print("Passed!")


def test_vitis_gemm():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    print(s.module)
    mod = s.build(target="vitis_hls", mode="sw_emu", project="gemm_vitis.prj")
    print(mod.hls_code)
    if hls.is_available("vitis_hls"):
        mod = s.build(target="vitis_hls", mode="csim", project="gemm_vitis_csim.prj")
        np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_C = np.matmul(np_A, np_B)
        np_C_allo = np.zeros((32, 32), dtype=np.int32)
        mod(np_A, np_B, np_C_allo)
        np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)
        print("Passed!")


def test_vitis_gemm_template():
    def gemm[T, M, N, K](A: "T[M, K]", B: "T[K, N]") -> "T[M, N]":
        C: T[M, N] = 0
        for i, j, k in allo.grid(M, N, K, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm, instantiate=[int32, 32, 32, 32])
    if hls.is_available("vitis_hls"):
        mod = s.build(target="vitis_hls", mode="csim", project="gemm_vitis_csim.prj")
        np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_C = np.matmul(np_A, np_B)
        np_C_allo = np.zeros((32, 32), dtype=np.int32)
        mod(np_A, np_B, np_C_allo)
        np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-4)
        print("Passed!")

    s = allo.customize(gemm, instantiate=[float32, 64, 64, 64])
    if hls.is_available("vitis_hls"):
        mod = s.build(target="vitis_hls", mode="csim", project="gemm_vitis_csim.prj")
        np_A = np.random.random(size=(64, 64)).astype(np.float32)
        np_B = np.random.random(size=(64, 64)).astype(np.float32)
        np_C = np.matmul(np_A, np_B)
        np_C_allo = np.zeros((64, 64), dtype=np.float32)
        mod(np_A, np_B, np_C_allo)
        np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-4)
        print("Passed!")


def test_vitis_io_stream():
    def foo(A: int32[32, 32], B: int32[32, 32]):
        pass

    def top(A: int32[32, 32]) -> int32[32, 32]:
        B: int32[32, 32]
        foo(A, B)
        return B

    s = allo.customize(top)
    s.dataflow("top")
    if hls.is_available("vitis_hls"):
        hls_mod = s.build(target="vitis_hls", mode="sw_emu", project="test_io.prj")
        print(s.module)
        hls_mod()


if __name__ == "__main__":
    pytest.main([__file__])
