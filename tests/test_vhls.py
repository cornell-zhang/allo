# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import pytest
import allo
from allo.ir.types import bool, int32, float32
from allo.memory import Memory
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


def test_wrap_io_linearized_index():
    M, N = 4, 4

    def matrix_copy(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N]
        for i, j in allo.grid(M, N, name="copy"):
            B[i, j] = A[i, j]
        return B

    s = allo.customize(matrix_copy)

    # Test wrap_io=True
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir, wrap_io=True)
        hls_code = mod.hls_code

        # Check that function arguments are declared as pointers
        assert (
            "int32_t *v" in hls_code
        ), "Expected pointer declaration for function arguments with wrap_io=True"

        # Check that helper functions (load_buf/store_res) exist for data movement
        assert (
            "load_buf0" in hls_code
        ), "Expected load_buf0 helper function with wrap_io=True"
        assert (
            "store_res1" in hls_code
        ), "Expected store_res1 helper function with wrap_io=True"

        # Check that linearized index pattern exists in helper functions
        # Pattern: ((var * 4) + var) for 4x4 array
        assert (
            "* 4) +" in hls_code
        ), "Expected linearized index pattern '* 4) +' in helper functions"
        print("wrap_io=True: All specific assertions passed!")

    # Test wrap_io=False
    with tempfile.TemporaryDirectory() as tmpdir:
        mod_no_wrap = s.build(
            target="vitis_hls", mode="sw_emu", project=tmpdir, wrap_io=False
        )
        hls_code_no_wrap = mod_no_wrap.hls_code

        # Check that function arguments are pointers with direct linearized access
        assert (
            "int32_t *v" in hls_code_no_wrap
        ), "Expected pointer declaration with wrap_io=False"

        # Check for direct linearized indexing pattern in main kernel
        # Pattern: v0[(i) * 4 + (j)] for direct pointer access
        assert (
            ") * 4 + (" in hls_code_no_wrap
        ), "Expected direct linearized index pattern ') * 4 + (' with wrap_io=False"

        # Verify no helper functions are generated
        assert (
            "load_buf0" not in hls_code_no_wrap
        ), "Should not have load_buf0 helper with wrap_io=False"
        print("wrap_io=False: All specific assertions passed!")

    print("Passed!")


# Module-level kernel functions for Memory HLS tests
# (Functions need to be at module level for proper AST parsing)
_MemUram = Memory(resource="URAM")
_MemBram2P = Memory(resource="BRAM", storage_type="RAM_2P")
_MemBram = Memory(resource="BRAM")
_MemLutram = Memory(resource="LUTRAM")


def _kernel_uram(a: int32[32] @ _MemUram) -> int32[32]:
    """Kernel with URAM memory annotation."""
    b: int32[32]
    for i in range(32):
        b[i] = a[i] + 1
    return b


def _kernel_bram_2p(a: float32[16, 16] @ _MemBram2P) -> float32[16, 16]:
    """Kernel with BRAM RAM_2P memory annotation."""
    b: float32[16, 16]
    for i, j in allo.grid(16, 16):
        b[i, j] = a[i, j] * 2.0
    return b


def _kernel_multi_mem(
    a: int32[32] @ _MemBram, b: int32[32] @ _MemUram, c: int32[32] @ _MemLutram
):
    """Kernel with multiple memory annotations."""
    for i in range(32):
        c[i] = a[i] + b[i]


def _kernel_local_mem(a: int32[32]) -> int32[32]:
    """Kernel with local variable using Memory annotation."""
    # Local buffer with URAM annotation
    buf: int32[32] @ _MemUram
    for i in range(32):
        buf[i] = a[i] * 2
    b: int32[32]
    for i in range(32):
        b[i] = buf[i] + 1
    return b


def _kernel_local_bram(a: float32[16, 16]) -> float32[16, 16]:
    """Kernel with local variable using BRAM RAM_2P annotation."""
    # Local buffer with BRAM RAM_2P annotation
    temp: float32[16, 16] @ _MemBram2P
    for i, j in allo.grid(16, 16):
        temp[i, j] = a[i, j] + 1.0
    b: float32[16, 16]
    for i, j in allo.grid(16, 16):
        b[i, j] = temp[i, j] * 2.0
    return b


def test_memory_uram_hls():
    """Test kernel with URAM Memory annotation generates bind_storage pragma."""
    s = allo.customize(_kernel_uram)
    print("=== MLIR Module (URAM) ===")
    print(s.module)

    # Check the memory space is in the memref type
    mlir_str = str(s.module)
    # URAM = impl_code 2, no storage = 0 -> memory_space = 32
    assert "32 : i32" in mlir_str, "Memory space 32 (URAM) should be in MLIR"

    # Build HLS code
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check for bind_storage pragma with URAM
    assert "#pragma HLS bind_storage variable=" in mod.hls_code
    assert "impl=uram" in mod.hls_code


def test_memory_bram_2p_hls():
    """Test kernel with BRAM RAM_2P Memory annotation generates bind_storage pragma."""
    s = allo.customize(_kernel_bram_2p)
    print("=== MLIR Module (BRAM RAM_2P) ===")
    print(s.module)

    # Check the memory space is in the memref type
    mlir_str = str(s.module)
    # BRAM = 1, RAM_2P = 2 -> memory_space = 1*16 + 2 = 18
    assert "18 : i32" in mlir_str, "Memory space 18 (BRAM+RAM_2P) should be in MLIR"

    # Build HLS code
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check for bind_storage pragma with BRAM and RAM_2P
    assert "#pragma HLS bind_storage variable=" in mod.hls_code
    assert "impl=bram" in mod.hls_code
    assert "type=ram_2p" in mod.hls_code


def test_multiple_memory_hls():
    """Test kernel with multiple Memory annotations generates multiple pragmas."""
    s = allo.customize(_kernel_multi_mem)
    print("=== MLIR Module (Multiple Memory) ===")
    print(s.module)

    # Build HLS code
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Count bind_storage pragmas - should have 3 (BRAM, URAM, LUTRAM)
    pragma_count = mod.hls_code.count("#pragma HLS bind_storage")
    assert pragma_count == 3, f"Expected 3 bind_storage pragmas, got {pragma_count}"

    # Check all implementation types are present
    assert "impl=bram" in mod.hls_code
    assert "impl=uram" in mod.hls_code
    assert "impl=lutram" in mod.hls_code


def test_memory_local_variable_uram():
    """Test local variable with URAM Memory annotation generates bind_storage pragma."""
    s = allo.customize(_kernel_local_mem)
    print("=== MLIR Module (Local URAM) ===")
    print(s.module)

    # Check the memory space is in the memref type for the local buffer
    mlir_str = str(s.module)
    # URAM = 2, no storage = 0 -> memory_space = 32
    assert (
        "32 : i32" in mlir_str
    ), "Memory space 32 (URAM) should be in MLIR for local buffer"

    # Build HLS code
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check for bind_storage pragma with URAM for the local buffer
    assert "#pragma HLS bind_storage variable=" in mod.hls_code
    assert "impl=uram" in mod.hls_code


def test_memory_local_variable_bram():
    """Test local variable with BRAM RAM_2P Memory annotation generates bind_storage pragma."""
    s = allo.customize(_kernel_local_bram)
    print("=== MLIR Module (Local BRAM RAM_2P) ===")
    print(s.module)

    # Check the memory space is in the memref type for the local buffer
    mlir_str = str(s.module)
    # BRAM = 1, RAM_2P = 2 -> memory_space = 18
    assert (
        "18 : i32" in mlir_str
    ), "Memory space 18 (BRAM+RAM_2P) should be in MLIR for local buffer"

    # Build HLS code
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check for bind_storage pragma with BRAM and RAM_2P for the local buffer
    assert "#pragma HLS bind_storage variable=" in mod.hls_code
    assert "impl=bram" in mod.hls_code
    assert "type=ram_2p" in mod.hls_code


def test_ihls():
    def top(A: int32[1]) -> int32[1]:
        A[0] = A[0] + 1
        return A

    s = allo.customize(top)
    mod = s.build(target="ihls")
    assert "h.single_task<Top>([=]() [[intel::kernel_args_restrict]]" in mod.hls_code


def test_while_basic():
    """Test basic while loop support in HLS backend."""

    def kernel(A: int32[10]):
        i: int32 = 0
        while i < 10:
            A[i] = i
            i += 1

    s = allo.customize(kernel)
    print("=== MLIR Module ===")
    print(s.module)

    # Build for vhls target
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check that while loop is generated
    assert "while (true)" in mod.hls_code or "while (" in mod.hls_code
    assert "break" in mod.hls_code
    print("test_while_basic passed!")


def test_while_with_array():
    """Test while loop with array operations."""

    def kernel(A: int32[10], B: int32[10]):
        i: int32 = 0
        while i < 10:
            B[i] = A[i] * 2
            i += 1

    s = allo.customize(kernel)
    print("=== MLIR Module ===")
    print(s.module)

    # Build for vhls target
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check that while loop is generated
    assert "while (true)" in mod.hls_code or "while (" in mod.hls_code
    assert "break" in mod.hls_code
    print("test_while_with_array passed!")


def test_wrap_io_false_nested_function():
    """Test that wrap_io=False does not flatten array indexing in nested functions."""
    M, N = 4, 8

    def top(A: "float32[M * N]", B: "float32[M * N]"):
        C: float32[M, N]
        inner(A, B, C)

    def inner(A: "float32[M * N]", B: "float32[M * N]", C: "float32[M, N]"):
        for m, n in allo.grid(M, N):
            C[m, n] = A[m * N + n]
        for m, n in allo.grid(M, N):
            B[m * N + n] = C[m, n]

    s = allo.customize(top)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir, wrap_io=False)
        hls_code = mod.hls_code
        print("\n=== Generated HLS Code ===")
        print(hls_code)
        # Check that inner function uses 2D indexing for C
        assert (
            "v2[(m1) * 8 + (n1)]" not in hls_code
        ), "C should use 2D indexing in inner function"
        assert "v2[m1][n1]" in hls_code, "C should use 2D indexing in inner function"
        print("test_wrap_io_false_nested_function passed!")


def test_wrap_io_false_nested_function_2D():
    M, N = 4, 8

    def inner(A: float32[M, N], B: float32[M, N]):
        for m, n in allo.grid(M, N):
            B[m, n] = A[m, n]

    def top(A: float32[M, N], B: float32[M, N]):
        inner(A, B)

    s = allo.customize(top)
    with pytest.raises(RuntimeError):
        s.build(target="vitis_hls", mode="sw_emu", project="", wrap_io=False)


def test_floordiv():
    def floordiv(A: int32[10], B: int32[10]) -> int32[10]:
        C: int32[10] = 0
        for i in range(10):
            C[i] = A[i] // B[i]
        return C

    s = allo.customize(floordiv)
    print(s.module)
    # CPU simulation
    mod = s.build()
    np_A = np.random.randint(1, 10, size=(10,)).astype(np.int32)
    np_B = np.random.randint(1, 10, size=(10,)).astype(np.int32)
    np_C = np_A // np_B
    np_C_allo = mod(np_A, np_B)
    np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)

    # Vitis HLS code generation
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
        hls_code = mod.hls_code
        print(hls_code)
        # Check if the division operator is used for FloorDivSIOp
        assert " / " in hls_code


if __name__ == "__main__":
    pytest.main([__file__])
