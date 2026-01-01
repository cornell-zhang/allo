# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=invalid-name,import-outside-toplevel,unsupported-assignment-operation

import tempfile
import os

import allo
from allo.ir.types import int32, Fixed
from allo.backend.xls import is_available as xls_sw_emu_available


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
        assert "#pragma hls_pipeline" in mod.final_cpp

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

    print("\n" + "-" * 70)
    print("  PART 2: Code Generation Tests")
    print("-" * 70)

    test_codegen_scalar_add()
    test_codegen_vector_add()
    test_codegen_pipeline()
    test_codegen_unroll()
    test_codegen_fixed_point()
    test_validation_error_float()

    print("\n" + "=" * 70)
    print("                    ✓ ALL TESTS PASSED")
    print("=" * 70 + "\n")
