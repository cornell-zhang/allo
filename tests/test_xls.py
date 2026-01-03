# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=invalid-name,import-outside-toplevel,unsupported-assignment-operation
# pylint: disable=broad-exception-caught

import tempfile
import os

import allo
from allo.ir.types import int32, Fixed


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
    print("  Code Generation Tests")
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
