# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import pytest
import allo
from allo.ir.types import bool, int32, float32
import numpy as np
import allo.backend.hls as hls
from allo.passes import generate_input_output_buffers


@pytest.mark.parametrize("flatten", [True, False])
def test_io_buffer_gemm(flatten):
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    print(s.module)
    allo.passes.generate_input_output_buffers(
        s.module, s.top_func_name, flatten=flatten
    )
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
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
        print(mod.hls_code)
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
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
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
            np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
            np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
            np_C = np.matmul(np_A, np_B)
            np_C_allo = np.zeros((32, 32), dtype=np.int32)
            mod(np_A, np_B, np_C_allo)
            np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-4)
            print("Passed!")

    s = allo.customize(gemm, instantiate=[float32, 64, 64, 64])
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
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
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
            print(s.module)
            np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
            np_B = np.zeros((32, 32), dtype=np.int32)
            hls_mod(np_A, np_B)


def test_csim_write_back():
    N = 256

    def compute(x: int32[N], y: int32[N]):
        for i in range(N):
            y[i] = x[i]

    s = allo.customize(compute)
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
            A = np.random.randint(0, 10, size=(N)).astype(np.int32)
            B = np.zeros((N), dtype=np.int32)
            mod(A, B)
            np.testing.assert_allclose(A, B, rtol=1e-5)
            print("Passed!")


def test_pointer_generation():
    def top(inst: bool, C: int32[3]):
        if inst:
            C[0] = C[0] + 1

    s = allo.customize(top)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
        assert "bool v" in mod.hls_code and ",," not in mod.hls_code
        if hls.is_available("vitis_hls"):
            inst = np.array([1], dtype=np.bool_)
            C = np.array([1, 2, 3], dtype=np.int32)
            mod(inst, C)
            np.testing.assert_allclose(C, [2, 2, 3], rtol=1e-5)
            print("Passed!")


def test_scalar_not_array():
    def top(inst: bool, C: int32[3]):
        flag: bool = inst[0]
        if flag:
            C[0] = C[0] + 1

    s = allo.customize(top)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
        assert "bool v" in mod.hls_code and ",," not in mod.hls_code
        if hls.is_available("vitis_hls"):
            C = np.array([1, 2, 3], dtype=np.int32)
            mod(1, C)
            np.testing.assert_allclose(C, [2, 2, 3], rtol=1e-5)
            print("Passed!")


def test_scalar():
    def case1(C: int32) -> int32:
        return C + 1

    s = allo.customize(case1)
    mod = s.build()
    assert mod(1) == 2
    print("Passed CPU simulation!")
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
        assert "int32_t *v1" in mod.hls_code
        # Note: Should not expect it to run using csim! Need to generate correct binding for mutable scalars in PyBind.


def test_size1_array():
    def top(A: int32[1]) -> int32[1]:
        A[0] = A[0] + 1
        return A

    s = allo.customize(top)
    mod = s.build()
    np_A = np.array([1], dtype=np.int32)
    np.testing.assert_allclose(mod(np_A), [2], rtol=1e-5)
    print("Passed CPU simulation!")
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
        print(mod.hls_code)
        assert "[1]" in mod.hls_code
        if hls.is_available("vitis_hls"):
            np_B = np.array([0], dtype=np.int32)
            mod(np_A, np_B)
            np.testing.assert_allclose(np_A, [2], rtol=1e-5)
            print("Passed!")


@pytest.mark.parametrize("flatten", [True, False])
def test_wrap_nonvoid(flatten):
    M, N = 4, 4

    def matrix_add(A: float32[M, N]) -> float32[M, N]:
        B: float32[M, N]
        for i, j in allo.grid(M, N, name="PE"):
            B[i, j] = A[i, j] + 1
        return B

    s = allo.customize(matrix_add)
    generate_input_output_buffers(s.module, "matrix_add", flatten=flatten)
    module = str(s.module)

    if flatten:
        # Top Function Argument
        assert (
            f"func.func @matrix_add(%arg0: memref<16xf32>) -> memref<16xf32>" in module
        )
        # Movement Function Generation
        assert (
            f"func.func @load_buf0(%arg0: memref<16xf32>, %arg1: memref<4x4xf32>)"
            in module
        )
        assert (
            f"func.func @store_res1(%arg0: memref<4x4xf32>, %arg1: memref<16xf32>)"
            in module
        )
        # Buffer Allocation
        assert f'%alloc = memref.alloc() {{name = "buf0"}} : memref<4x4xf32>' in module
        # Function Call
        assert (
            f"call @load_buf0(%arg0, %alloc) : (memref<16xf32>, memref<4x4xf32>) -> ()"
            in module
        )
        assert (
            f"call @store_res1(%alloc_1, %alloc_0) : (memref<4x4xf32>, memref<16xf32>) -> ()"
            in module
        )
        # Return Value Allocation
        assert f'%alloc_0 = memref.alloc() {{name = "res1"}} : memref<16xf32>' in module
        # ReturnOP Update
        assert f"return %alloc_0 : memref<16xf32>" in module
    else:
        # Top Function Argument
        assert (
            f"func.func @matrix_add(%arg0: memref<4x4xf32>) -> memref<4x4xf32>"
            in module
        )
        # Movement Function Generation
        assert (
            f"func.func @load_buf0(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>)"
            in module
        )
        assert (
            f"func.func @store_res1(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>)"
            in module
        )
        # Buffer Allocation
        assert f'%alloc = memref.alloc() {{name = "buf0"}} : memref<4x4xf32>' in module
        # Function Call
        assert (
            f"call @load_buf0(%arg0, %alloc) : (memref<4x4xf32>, memref<4x4xf32>) -> ()"
            in module
        )
        assert (
            f"call @store_res1(%alloc_1, %alloc_0) : (memref<4x4xf32>, memref<4x4xf32>) -> ()"
            in module
        )
        # Return Value Allocation
        assert (
            f'%alloc_0 = memref.alloc() {{name = "res1"}} : memref<4x4xf32>' in module
        )
        # ReturnOP Update
        assert f"return %alloc_0 : memref<4x4xf32>" in module

    print("Passed!")


def test_ihls():
    def top(A: int32[1]) -> int32[1]:
        A[0] = A[0] + 1
        return A

    s = allo.customize(top)
    mod = s.build(target="ihls")
    assert "h.single_task<Top>([=]() [[intel::kernel_args_restrict]]" in mod.hls_code


if __name__ == "__main__":
    pytest.main([__file__])
