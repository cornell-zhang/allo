# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=invalid-name,import-outside-toplevel,unsupported-assignment-operation
# pylint: disable=broad-exception-caught

import tempfile
import os

import allo
from allo.ir.types import int32, Fixed
from allo.memory import Memory
from allo.backend.xls import is_available as xls_sw_emu_available


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
# Test XLS sw_emu functional execution
# These tests compile and run the generated C++ code using g++
# ##############################################################
def test_sw_emu_scalar_add():
    def add(a: int32, b: int32) -> int32:
        return a + b

    s = allo.customize(add)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", mode="sw_emu", project=tmpdir)

        if xls_sw_emu_available():
            test_cases = [(10, 20, 30), (-5, 15, 10), (0, 0, 0), (100, -100, 0)]
            failures = []
            print("test_sw_emu_scalar_add ", end="")
            for a_val, b_val, expected in test_cases:
                result = mod(a_val, b_val)
                if result == expected:
                    print(".", end="", flush=True)
                else:
                    print("F", end="", flush=True)
                    failures.append(
                        f"add({a_val}, {b_val}) = {result}, expected {expected}"
                    )
            print()
            for msg in failures:
                print(f"  FAIL: {msg}")
            assert not failures
        else:
            assert "#pragma hls_top" in mod.final_cpp
            print("test_sw_emu_scalar_add . (codegen only)")


def test_sw_emu_multiply():
    def mul(a: int32, b: int32) -> int32:
        return a * b

    s = allo.customize(mul)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", mode="sw_emu", project=tmpdir)

        if xls_sw_emu_available():
            test_cases = [(3, 4, 12), (-2, 5, -10), (0, 100, 0), (7, 8, 56)]
            failures = []
            print("test_sw_emu_multiply ", end="")
            for a_val, b_val, expected in test_cases:
                result = mod(a_val, b_val)
                if result == expected:
                    print(".", end="", flush=True)
                else:
                    print("F", end="", flush=True)
                    failures.append(
                        f"mul({a_val}, {b_val}) = {result}, expected {expected}"
                    )
            print()
            for msg in failures:
                print(f"  FAIL: {msg}")
            assert not failures
        else:
            assert "class TestBlock" in mod.final_cpp
            print("test_sw_emu_multiply . (codegen only)")


def test_sw_emu_subtract():
    def sub(a: int32, b: int32) -> int32:
        return a - b

    s = allo.customize(sub)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", mode="sw_emu", project=tmpdir)

        if xls_sw_emu_available():
            test_cases = [(20, 10, 10), (-5, -5, 0), (100, 200, -100)]
            failures = []
            print("test_sw_emu_subtract ", end="")
            for a_val, b_val, expected in test_cases:
                result = mod(a_val, b_val)
                if result == expected:
                    print(".", end="", flush=True)
                else:
                    print("F", end="", flush=True)
                    failures.append(
                        f"sub({a_val}, {b_val}) = {result}, expected {expected}"
                    )
            print()
            for msg in failures:
                print(f"  FAIL: {msg}")
            assert not failures
        else:
            print("test_sw_emu_subtract . (codegen only)")


def test_sw_emu_mac():
    # Multiply-accumulate: a*b + c (common in GEMM)
    def mac(a: int32, b: int32, c: int32) -> int32:
        return a * b + c

    s = allo.customize(mac)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", mode="sw_emu", project=tmpdir)

        if xls_sw_emu_available():
            # Test cases: (a, b, c, expected = a*b+c)
            test_cases = [
                (2, 3, 4, 10),  # 2*3+4=10
                (5, 5, 0, 25),  # 5*5+0=25
                (-2, 3, 10, 4),  # -2*3+10=4
                (10, 10, 100, 200),  # 10*10+100=200
            ]
            failures = []
            print("test_sw_emu_mac ", end="")
            for a_val, b_val, c_val, expected in test_cases:
                result = mod(a_val, b_val, c_val)
                if result == expected:
                    print(".", end="", flush=True)
                else:
                    print("F", end="", flush=True)
                    failures.append(
                        f"mac({a_val}, {b_val}, {c_val}) = {result}, expected {expected}"
                    )
            print()
            for msg in failures:
                print(f"  FAIL: {msg}")
            assert not failures
        else:
            print("test_sw_emu_mac . (codegen only)")


def test_sw_emu_dot2():
    # 2-element dot product: a0*b0 + a1*b1
    def dot2(a0: int32, a1: int32, b0: int32, b1: int32) -> int32:
        return a0 * b0 + a1 * b1

    s = allo.customize(dot2)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", mode="sw_emu", project=tmpdir)

        if xls_sw_emu_available():
            # Test cases: (a0, a1, b0, b1, expected)
            test_cases = [
                (1, 2, 3, 4, 11),  # 1*3 + 2*4 = 11
                (2, 2, 2, 2, 8),  # 2*2 + 2*2 = 8
                (1, 0, 5, 10, 5),  # 1*5 + 0*10 = 5
                (-1, 1, 1, 1, 0),  # -1*1 + 1*1 = 0
            ]
            failures = []
            print("test_sw_emu_dot2 ", end="")
            for a0, a1, b0, b1, expected in test_cases:
                result = mod(a0, a1, b0, b1)
                if result == expected:
                    print(".", end="", flush=True)
                else:
                    print("F", end="", flush=True)
                    failures.append(
                        f"dot2({a0},{a1},{b0},{b1}) = {result}, expected {expected}"
                    )
            print()
            for msg in failures:
                print(f"  FAIL: {msg}")
            assert not failures
        else:
            print("test_sw_emu_dot2 . (codegen only)")


def test_sw_emu_vvadd():
    import numpy as np

    def vvadd(A: int32[8], B: int32[8]) -> int32[8]:
        C: int32[8] = 0
        for i in allo.grid(8):
            C[i] = A[i] + B[i]
        return C

    s = allo.customize(vvadd)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", mode="sw_emu", project=tmpdir)

        if xls_sw_emu_available():
            A = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
            B = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
            expected = A + B

            print("test_sw_emu_vvadd ", end="")
            try:
                result = mod(A, B)
                if result is not None and np.array_equal(result, expected):
                    print(".")
                else:
                    print("F")
                    print(f"  FAIL: got {result}, expected {expected}")
                    assert False
            except Exception as e:
                print("F")
                print(f"  FAIL: {e}")
                assert False
        else:
            print("test_sw_emu_vvadd . (codegen only)")


def test_sw_emu_gemm():
    import numpy as np

    def gemm(A: int32[2, 2], B: int32[2, 2]) -> int32[2, 2]:
        C: int32[2, 2] = 0
        for i in allo.grid(2):
            for j in allo.grid(2):
                for k in allo.grid(2):
                    C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="xls", mode="sw_emu", project=tmpdir)

        if xls_sw_emu_available():
            A = np.array([[1, 2], [3, 4]], dtype=np.int32)
            B = np.array([[5, 6], [7, 8]], dtype=np.int32)
            expected = A @ B  # [[19, 22], [43, 50]]

            print("test_sw_emu_gemm ", end="")
            try:
                result = mod(A, B)
                if result is not None and np.array_equal(result, expected):
                    print(".")
                else:
                    print("F")
                    print(f"  FAIL: got {result}, expected {expected}")
                    assert False
            except Exception as e:
                print("F")
                print(f"  FAIL: {e}")
                assert False
        else:
            print("test_sw_emu_gemm . (codegen only)")


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
    print("  PART 1: sw_emu Functional Tests (C++ Compilation & Execution)")
    print("-" * 70)

    test_sw_emu_scalar_add()
    test_sw_emu_multiply()
    test_sw_emu_subtract()
    test_sw_emu_mac()
    test_sw_emu_dot2()
    test_sw_emu_vvadd()
    test_sw_emu_gemm()

    print("\n" + "-" * 70)
    print("  PART 2: Code Generation Tests")
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
