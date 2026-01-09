# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=invalid-name,import-outside-toplevel,unsupported-assignment-operation
# pylint: disable=broad-exception-caught

import tempfile
import os

import allo
from allo.ir.types import int32, Fixed
from allo.memory import Memory


# ##############################################################
# Test XLS backend code generation
# ##############################################################
def test_codegen_scalar_add():
    def add(a: int32, b: int32) -> int32:
        return a + b

    s = allo.customize(add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir, use_memory=False)
        assert "#pragma hls_top" in mod.final_cpp
        assert os.path.exists(os.path.join(tmpdir, "test_block.cpp"))
    print("test_codegen_scalar_add .")


def test_codegen_vector_add():
    def vvadd(a: int32[16], b: int32[16]) -> int32[16]:
        c: int32[16] = 0
        for i in allo.grid(16):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(vvadd)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod_reg = s.build(target="xls", project=tmpdir, use_memory=False)
        assert "__xls_channel" in mod_reg.final_cpp
        assert "class TestBlock" in mod_reg.final_cpp

    with tempfile.TemporaryDirectory() as tmpdir:
        mod_mem = s.build(target="xls", project=tmpdir, use_memory=True)
        assert (
            "__xls_memory" in mod_mem.final_cpp
            or "// __xls_memory" in mod_mem.core_code
        )

    print("test_codegen_vector_add .")


def test_codegen_pipeline():
    def pipe_test(a: int32[32], b: int32[32]) -> int32[32]:
        c: int32[32] = 0
        for i in allo.grid(32):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(pipe_test)
    s.pipeline("i")

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir)
        assert "#pragma hls_pipeline_init_interval" in mod.final_cpp

    print("test_codegen_pipeline .")


def test_codegen_unroll():
    def unroll_test(a: int32[8], b: int32[8]) -> int32[8]:
        c: int32[8] = 0
        for i in allo.grid(8, name="loop"):
            c[i] = a[i] * b[i]
        return c

    s = allo.customize(unroll_test)
    s.unroll("i")

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir)
        assert "#pragma hls_unroll" in mod.final_cpp

    print("test_codegen_unroll .")


def test_codegen_fixed_point():
    Fixed16_8 = Fixed(16, 8)

    def fadd(a: Fixed16_8, b: Fixed16_8) -> Fixed16_8:
        return a + b

    s = allo.customize(fadd)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir)
        assert "struct Fixed" in mod.final_cpp or "Fixed<" in mod.final_cpp

    print("test_codegen_fixed_point .")


def test_validation_error_float():
    from allo.ir.types import float32

    def float_add(a: float32, b: float32) -> float32:
        return a + b

    s = allo.customize(float_add)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            s.build(target="xls", project=tmpdir)
        assert False, "Should have raised error for float types"
    except RuntimeError as e:
        assert "FLOAT_TYPE" in str(e) or "float" in str(e).lower()

    print("test_validation_error_float .")


# ##############################################################
# Test Memory annotation support for XLS backend
# ##############################################################
def test_memory_annotation_ram_1p():
    """Test that RAM_1P storage type generates correct textproto with RAM_1RW."""

    def kernel(A: int32[8] @ Memory(resource="BRAM", storage_type="RAM_1P")):
        for i in allo.grid(8):
            A[i] = A[i] + 1

    s = allo.customize(kernel)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir, use_memory=True)
        # Check that textproto is generated
        assert mod.rewrites_textproto, "rewrites_textproto should be generated"
        # Check for RAM_1RW (single port)
        assert "kind: RAM_1RW" in mod.rewrites_textproto
        # Check for shared channels (single port uses _req and _resp)
        assert "_req" in mod.rewrites_textproto
        assert "_resp" in mod.rewrites_textproto
        # Should NOT have separate read/write channels
        assert "_read_request" not in mod.rewrites_textproto
        assert "_write_request" not in mod.rewrites_textproto

    print("test_memory_annotation_ram_1p .")


def test_memory_annotation_ram_2p():
    """Test that RAM_2P storage type generates correct textproto with RAM_1R1W."""

    def kernel(A: int32[8] @ Memory(resource="BRAM", storage_type="RAM_2P")):
        for i in allo.grid(8):
            A[i] = A[i] + 1

    s = allo.customize(kernel)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir, use_memory=True)
        # Check that textproto is generated
        assert mod.rewrites_textproto, "rewrites_textproto should be generated"
        # Check for RAM_1R1W (dual port)
        assert "kind: RAM_1R1W" in mod.rewrites_textproto
        # Check for separate read/write channels
        assert "_read_request" in mod.rewrites_textproto
        assert "_read_response" in mod.rewrites_textproto
        assert "_write_request" in mod.rewrites_textproto
        assert "_write_response" in mod.rewrites_textproto

    print("test_memory_annotation_ram_2p .")


def test_memory_annotation_rom_1p():
    """Test that ROM_1P storage type generates correct textproto with RAM_1RW."""

    def kernel(A: int32[8] @ Memory(resource="BRAM", storage_type="ROM_1P")):
        for i in allo.grid(8):
            A[i] = A[i] + 1

    s = allo.customize(kernel)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir, use_memory=True)
        # Check that textproto is generated
        assert mod.rewrites_textproto, "rewrites_textproto should be generated"
        # Check for RAM_1RW (ROM_1P maps to RAM_1RW)
        assert "kind: RAM_1RW" in mod.rewrites_textproto

    print("test_memory_annotation_rom_1p .")


def test_memory_annotation_default():
    """Test that default (no storage_type) uses RAM_2P (RAM_1R1W)."""

    def kernel(A: int32[8] @ Memory(resource="BRAM")):
        for i in allo.grid(8):
            A[i] = A[i] + 1

    s = allo.customize(kernel)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir, use_memory=True)
        # Check that textproto is generated
        assert mod.rewrites_textproto, "rewrites_textproto should be generated"
        # Default should be RAM_1R1W (RAM_2P)
        assert "kind: RAM_1R1W" in mod.rewrites_textproto

    print("test_memory_annotation_default .")


def test_memory_annotation_unsupported_storage_type():
    """Test that unsupported storage types raise appropriate errors."""

    def kernel(A: int32[8] @ Memory(resource="BRAM", storage_type="RAM_T2P")):
        for i in allo.grid(8):
            A[i] = A[i] + 1

    s = allo.customize(kernel)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            s.build(target="xls", project=tmpdir, use_memory=True)
        assert False, "Should have raised error for unsupported storage type"
    except RuntimeError as e:
        error_msg = str(e)
        # Check that error mentions the unsupported storage type
        assert "RAM_T2P" in error_msg or "storage_type" in error_msg.lower()
        # Check that error mentions supported types
        assert "RAM_1P" in error_msg or "RAM_2P" in error_msg or "ROM_1P" in error_msg
        # Check that it mentions XLS backend
        assert "XLS" in error_msg or "xls" in error_msg.lower()

    print("test_memory_annotation_unsupported_storage_type .")


def test_memory_annotation_multiple_arrays():
    """Test that multiple arrays with different storage types generate correct textproto."""

    def kernel(
        A: int32[4] @ Memory(resource="BRAM", storage_type="RAM_1P"),
        B: int32[4] @ Memory(resource="BRAM", storage_type="RAM_2P"),
        C: int32[4] @ Memory(resource="BRAM", storage_type="ROM_1P"),
    ):
        for i in allo.grid(4):
            C[i] = A[i] + B[i]

    s = allo.customize(kernel)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir, use_memory=True)
        # Check that textproto is generated
        assert mod.rewrites_textproto, "rewrites_textproto should be generated"
        # Check for all three storage types
        assert "kind: RAM_1RW" in mod.rewrites_textproto  # RAM_1P and ROM_1P
        assert "kind: RAM_1R1W" in mod.rewrites_textproto  # RAM_2P
        # Count occurrences to verify all three arrays are present
        assert mod.rewrites_textproto.count("rewrites {") == 3

    print("test_memory_annotation_multiple_arrays .")


def test_memory_annotation_textproto_structure():
    """Test that generated textproto has correct structure."""

    def kernel(A: int32[16] @ Memory(resource="BRAM", storage_type="RAM_2P")):
        for i in allo.grid(16):
            A[i] = A[i] + 1

    s = allo.customize(kernel)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", project=tmpdir, use_memory=True)
        textproto = mod.rewrites_textproto
        # Check header comments
        assert "Automatically Generated by Allo" in textproto
        assert "XLS [CC] Backend" in textproto
        # Check required fields
        assert "from_config" in textproto
        assert "to_config" in textproto
        assert "kind: RAM_ABSTRACT" in textproto
        assert "kind: RAM_1R1W" in textproto
        assert "depth: 16" in textproto  # Array size
        # Check channel mappings
        assert "abstract_read_req" in textproto
        assert "abstract_read_resp" in textproto
        assert "abstract_write_req" in textproto
        assert "write_completion" in textproto
        assert "to_name_prefix" in textproto

    print("test_memory_annotation_textproto_structure .")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("                    XLS Backend Test Suite")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("  Code Generation Tests")
    print("-" * 70)

    test_codegen_scalar_add()
    test_codegen_vector_add()
    test_codegen_pipeline()
    test_codegen_unroll()
    test_codegen_fixed_point()
    test_validation_error_float()

    print("\n" + "-" * 70)
    print("  Memory Annotation Tests")
    print("-" * 70)

    test_memory_annotation_ram_1p()
    test_memory_annotation_ram_2p()
    test_memory_annotation_rom_1p()
    test_memory_annotation_default()
    test_memory_annotation_unsupported_storage_type()
    test_memory_annotation_multiple_arrays()
    test_memory_annotation_textproto_structure()

    print("\n" + "=" * 70)
    print("                    ✓ ALL TESTS PASSED")
    print("=" * 70 + "\n")
