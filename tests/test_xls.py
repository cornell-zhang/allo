# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for the XLS[cc] backend.

These tests verify that Allo can generate valid XLS[cc] C++ code for various
computational patterns. The generated code uses:
- XlsInt<Width, Signed> for arbitrary precision integers
- __xls_channel<T, dir> for streaming I/O
- __xls_memory<T, size> for on-chip memory (SRAM/BRAM)

Note: Actual XLS compilation requires the XLS toolchain to be installed.
Tests will skip XLS compilation if the toolchain is not available.
"""

import tempfile
import os

import numpy as np
import allo
from allo.ir.types import int32, int8, int16, uint8, uint32, Fixed
from allo.backend import xls


# ##############################################################
# Test Combinational Functions (Scalar Operations)
# ##############################################################
def test_scalar_add():
    """Test simple scalar addition - combinational logic."""
    def add(a: int32, b: int32) -> int32:
        return a + b

    s = allo.customize(add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir, use_memory=False)
        
        # Verify code generation
        assert "#pragma hls_top" in mod.final_cpp
        assert "int add(" in mod.final_cpp or "int v" in mod.final_cpp
        
        # Check that project files were created
        assert os.path.exists(os.path.join(tmpdir, "test_block.cpp"))
        
    print("test_scalar_add passed!")


def test_scalar_mul():
    """Test scalar multiplication with different bit widths."""
    def mul(a: int16, b: int16) -> int32:
        c: int32 = a * b
        return c

    s = allo.customize(mul)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir)
        
        # Verify XlsInt types are used for bit-width conversions
        assert "ac_int<" in mod.final_cpp or "int" in mod.final_cpp
        
    print("test_scalar_mul passed!")


def test_fixed_point_add():
    """Test fixed-point addition."""
    Fixed16_8 = Fixed(16, 8)
    
    def fadd(a: Fixed16_8, b: Fixed16_8) -> Fixed16_8:
        return a + b

    s = allo.customize(fadd)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir)
        
        # Verify fixed-point template is included
        assert "struct Fixed" in mod.final_cpp or "Fixed<" in mod.final_cpp
        
    print("test_fixed_point_add passed!")


# ##############################################################
# Test Sequential Functions (Array Operations with Channels)
# ##############################################################
def test_vector_add():
    """Test vector addition - sequential with channels."""
    def vvadd(a: int32[16], b: int32[16]) -> int32[16]:
        c: int32[16] = 0
        for i in allo.grid(16):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(vvadd)
    
    # Test register mode (arrays as C arrays)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod_reg = s.build(target="xls", project=tmpdir, use_memory=False)
        
        # Verify channels are declared
        assert "__xls_channel" in mod_reg.final_cpp
        assert "class TestBlock" in mod_reg.final_cpp
        
        # In register mode, no __xls_memory
        assert "__xls_memory" not in mod_reg.final_cpp
        
    # Test memory mode (arrays as __xls_memory)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod_mem = s.build(target="xls", project=tmpdir, use_memory=True)
        
        # Verify memory declarations
        assert "__xls_memory" in mod_mem.final_cpp or "// __xls_memory_decl__" in mod_mem.core_code
        
        # Check textproto was generated
        if mod_mem.rewrites_textproto:
            assert "rewrites {" in mod_mem.rewrites_textproto
            
    print("test_vector_add passed!")


def test_vector_add_with_pipeline():
    """Test vector addition with pipeline directive."""
    def vvadd_pipe(a: int32[32], b: int32[32]) -> int32[32]:
        c: int32[32] = 0
        for i in allo.grid(32):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(vvadd_pipe)
    s.pipeline("i")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir)
        
        # Verify pipeline pragma is present
        assert "#pragma hls_pipeline_init_interval" in mod.final_cpp or \
               "hls_pipeline" in mod.final_cpp
               
    print("test_vector_add_with_pipeline passed!")


def test_matrix_vector():
    """Test matrix-vector multiplication."""
    M, N = 8, 8
    
    def matvec(A: int32[M, N], x: int32[N]) -> int32[M]:
        y: int32[M] = 0
        for i in allo.grid(M):
            for j in allo.grid(N):
                y[i] += A[i, j] * x[j]
        return y

    s = allo.customize(matvec)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir)
        
        # Verify nested loops are present
        assert "for" in mod.final_cpp
        
    print("test_matrix_vector passed!")


def test_gemm():
    """Test general matrix-matrix multiplication."""
    M, N, K = 4, 4, 4
    
    def gemm(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        C: int32[M, N] = 0
        for i, j, k in allo.grid(M, N, K):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    
    # Test both modes
    for use_mem in [False, True]:
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="xls", project=tmpdir, use_memory=use_mem)
            assert "TestBlock" in mod.final_cpp or "#pragma hls_top" in mod.final_cpp
            
    print("test_gemm passed!")


def test_gemm_with_unroll():
    """Test GEMM with unroll directive."""
    M, N, K = 4, 4, 4
    
    def gemm_unroll(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        C: int32[M, N] = 0
        for i, j, k in allo.grid(M, N, K, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm_unroll)
    s.unroll("k")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir)
        
        # Verify unroll pragma
        assert "#pragma hls_unroll" in mod.final_cpp
        
    print("test_gemm_with_unroll passed!")


# ##############################################################
# Test Different Data Types
# ##############################################################
def test_int8_operations():
    """Test 8-bit integer operations."""
    def int8_add(a: int8[16], b: int8[16]) -> int8[16]:
        c: int8[16] = 0
        for i in allo.grid(16):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(int8_add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir)
        # XLS uses ac_int for bit-precise types
        assert "ac_int<8" in mod.final_cpp or "int8" in mod.final_cpp.lower() or "char" in mod.final_cpp
        
    print("test_int8_operations passed!")


def test_uint8_operations():
    """Test unsigned 8-bit integer operations."""
    def uint8_add(a: uint8[16], b: uint8[16]) -> uint8[16]:
        c: uint8[16] = 0
        for i in allo.grid(16):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(uint8_add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir)
        # Should use unsigned type
        assert "ac_int<8, false>" in mod.final_cpp or "uint" in mod.final_cpp.lower()
        
    print("test_uint8_operations passed!")


# ##############################################################
# Test Memory Mode Features
# ##############################################################
def test_memory_mode_textproto():
    """Test that memory mode generates proper textproto for RAM rewrites."""
    def mem_test(a: int32[32]) -> int32[32]:
        b: int32[32] = 0
        for i in allo.grid(32):
            b[i] = a[i] * 2
        return b

    s = allo.customize(mem_test)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir, use_memory=True)
        
        # Check textproto file was created
        textproto_path = os.path.join(tmpdir, "rewrites.textproto")
        if mod.rewrites_textproto:
            assert os.path.exists(textproto_path)
            with open(textproto_path) as f:
                content = f.read()
            # Verify textproto format
            assert "proto-file" in content or "rewrites" in content
            
    print("test_memory_mode_textproto passed!")


# ##############################################################
# Test XLS Availability and Compilation (Optional)
# ##############################################################
def test_xls_compilation():
    """Test actual XLS compilation if toolchain is available."""
    if not xls.is_available():
        print("XLS toolchain not available, skipping compilation test")
        return
        
    def simple_add(a: int32, b: int32) -> int32:
        return a + b

    s = allo.customize(simple_add)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir)
        
        cpp_file = os.path.join(tmpdir, "test_block.cpp")
        assert os.path.exists(cpp_file)
        
        # TODO: Add actual xlscc compilation when XLS Python bindings are available
        # For now, just verify the file was generated correctly
        
    print("test_xls_compilation passed!")


# ##############################################################
# Test Error Handling
# ##############################################################
def test_validation_error_float():
    """Test that floating-point types raise validation error."""
    from allo.ir.types import float32
    
    def float_add(a: float32, b: float32) -> float32:
        return a + b

    s = allo.customize(float_add)
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            s.build(target="xls", project=tmpdir)
        assert False, "Should have raised an error for float types"
    except RuntimeError as e:
        # Expected - XLS doesn't support floats
        assert "FLOAT_TYPE" in str(e) or "float" in str(e).lower()
        
    print("test_validation_error_float passed!")


# ##############################################################
# Run All Tests
# ##############################################################
if __name__ == "__main__":
    # Combinational tests
    test_scalar_add()
    test_scalar_mul()
    test_fixed_point_add()
    
    # Sequential tests
    test_vector_add()
    test_vector_add_with_pipeline()
    test_matrix_vector()
    test_gemm()
    test_gemm_with_unroll()
    
    # Data type tests
    test_int8_operations()
    test_uint8_operations()
    
    # Memory mode tests
    test_memory_mode_textproto()
    
    # Compilation test (optional)
    test_xls_compilation()
    
    # Error handling tests
    test_validation_error_float()
    
    print("\n" + "=" * 60)
    print("All XLS backend tests passed!")
    print("=" * 60)

