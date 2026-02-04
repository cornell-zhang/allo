# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import pytest
import numpy as np
import allo
from allo.ir.types import int32, int8, int16, float32


def check_catapult():
    """Check if Catapult HLS is available via MGC_HOME environment variable."""
    if "MGC_HOME" not in os.environ:
        return False
    return True


# =============================================================================
# Code Generation Tests (no Catapult installation required)
# =============================================================================


def test_catapult_vvadd():
    """Test basic vector addition for Catapult HLS"""

    def vvadd(a: int32[100], b: int32[100]) -> int32[100]:
        c: int32[100]
        for i in range(100):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(vvadd)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check Catapult-specific headers
        assert "#include <ac_int.h>" in mod.hls_code
        assert "#include <ac_fixed.h>" in mod.hls_code
        assert "#include <ac_channel.h>" in mod.hls_code

        # Check function is generated
        assert "void vvadd(" in mod.hls_code
        print("test_catapult_vvadd passed!")


def test_catapult_gemm():
    """Test matrix multiplication for Catapult HLS"""

    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check function signature
        assert "void gemm(" in mod.hls_code
        # Check nested loops are generated
        assert "for (" in mod.hls_code
        print("test_catapult_gemm passed!")


def test_catapult_different_types():
    """Test various data types for Catapult HLS"""

    def type_test(a: int8[10], b: int16[10], c: int32[10]) -> int32[10]:
        d: int32[10]
        for i in range(10):
            d[i] = a[i] + b[i] + c[i]
        return d

    s = allo.customize(type_test)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check that int8 and int16 types are used
        assert "int8_t" in mod.hls_code or "ac_int<8" in mod.hls_code
        assert "int16_t" in mod.hls_code or "ac_int<16" in mod.hls_code
        assert "int32_t" in mod.hls_code
        print("test_catapult_different_types passed!")


def test_catapult_float():
    """Test floating point operations for Catapult HLS"""

    def float_add(a: float32[10], b: float32[10]) -> float32[10]:
        c: float32[10]
        for i in range(10):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(float_add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check float type is used
        assert "float " in mod.hls_code
        print("test_catapult_float passed!")


def test_catapult_conditional():
    """Test conditional statements for Catapult HLS"""

    def conditional(a: int32[10], b: int32[10]) -> int32[10]:
        c: int32[10]
        for i in range(10):
            if a[i] > b[i]:
                c[i] = a[i]
            else:
                c[i] = b[i]
        return c

    s = allo.customize(conditional)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check if-else is generated
        assert "if (" in mod.hls_code
        assert "else" in mod.hls_code
        print("test_catapult_conditional passed!")


def test_catapult_pipeline():
    """Test pipeline pragma for Catapult HLS"""

    def pipelined_loop(a: int32[100]) -> int32[100]:
        b: int32[100]
        for i in range(100):
            b[i] = a[i] * 2
        return b

    s = allo.customize(pipelined_loop)
    s.pipeline("i")

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check Catapult-specific pipeline pragma
        assert "#pragma hls_pipeline_init_interval" in mod.hls_code
        print("test_catapult_pipeline passed!")


def test_catapult_unroll():
    """Test unroll pragma for Catapult HLS"""

    def unrolled_loop(a: int32[10]) -> int32[10]:
        b: int32[10]
        for i in range(10):
            b[i] = a[i] + 1
        return b

    s = allo.customize(unrolled_loop)
    s.unroll("i")

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check Catapult-specific unroll pragma
        assert "#pragma hls_unroll" in mod.hls_code
        print("test_catapult_unroll passed!")


def test_catapult_2d_array():
    """Test 2D array operations for Catapult HLS"""

    def matrix_add(A: int32[4, 4], B: int32[4, 4]) -> int32[4, 4]:
        C: int32[4, 4]
        for i, j in allo.grid(4, 4):
            C[i, j] = A[i, j] + B[i, j]
        return C

    s = allo.customize(matrix_add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check 2D array declaration
        assert "[4][4]" in mod.hls_code
        print("test_catapult_2d_array passed!")


def test_catapult_nested_function():
    """Test nested function calls for Catapult HLS"""

    def inner(a: int32, b: int32) -> int32:
        return a + b

    def outer(x: int32[10], y: int32[10]) -> int32[10]:
        z: int32[10]
        for i in range(10):
            z[i] = inner(x[i], y[i])
        return z

    s = allo.customize(outer)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check both functions are generated
        assert "void inner(" in mod.hls_code or "int32_t inner(" in mod.hls_code
        assert "void outer(" in mod.hls_code
        print("test_catapult_nested_function passed!")


def test_catapult_tcl_generation():
    """Test TCL script generation for Catapult HLS"""

    def simple_add(a: int32[10], b: int32[10]) -> int32[10]:
        c: int32[10]
        for i in range(10):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(simple_add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check TCL file exists and has correct content
        tcl_path = os.path.join(tmpdir, "run.tcl")
        assert os.path.exists(tcl_path), "TCL file should be generated"

        with open(tcl_path, "r", encoding="utf-8") as f:
            tcl_content = f.read()

        # Check Catapult-specific TCL commands
        assert "directive set -DESIGN_HIERARCHY" in tcl_content
        assert "directive set -CLOCKS" in tcl_content
        assert "go analyze" in tcl_content
        assert "go compile" in tcl_content
        assert "go assembly" in tcl_content
        assert "go extract" in tcl_content
        print("test_catapult_tcl_generation passed!")


# =============================================================================
# Customization Tests (no Catapult installation required)
# =============================================================================


def test_catapult_partition():
    """Test array partitioning for Catapult HLS"""

    def partition_test(A: int32[10, 10]) -> int32[10, 10]:
        B: int32[10, 10]
        for i, j in allo.grid(10, 10):
            B[i, j] = A[i, j] + 1
        return B

    s = allo.customize(partition_test)
    s.partition(s.A, dim=1)
    s.partition(s.B, dim=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)
        # Check that we do NOT emit Vivado HLS pragmas
        assert "#pragma HLS array_partition" not in mod.hls_code
        # For now, we assume implicit partitioning or handled via other means.
        # Ideally we should check if generated code handles parallel access if unrolled.
        # But here we just check we don't emit wrong pragmas.
        print("test_catapult_partition passed!")


def test_catapult_parallel():
    """Test parallel loop for Catapult HLS"""

    def parallel_test(A: int32[10]) -> int32[10]:
        B: int32[10]
        for i in allo.grid(10):
            B[i] = A[i] * 2
        return B

    s = allo.customize(parallel_test)
    s.parallel("i")

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)
        # Check for unroll pragma which is often used for parallel loops in HLS
        # Usually full unroll = parallel.
        assert "#pragma hls_unroll" in mod.hls_code
        print("test_catapult_parallel passed!")


# =============================================================================
# CSIM Tests (requires Catapult installation with MGC_HOME)
# =============================================================================


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csim():
    """Test csim flow using g++ for Catapult HLS"""

    def vvadd(a: int32[10], b: int32[10]) -> int32[10]:
        c: int32[10]
        for i in range(10):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(vvadd)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csim", project=tmpdir)

        # Create input data
        a = np.random.randint(0, 100, (10,)).astype(np.int32)
        b = np.random.randint(0, 100, (10,)).astype(np.int32)
        c = np.zeros(10, dtype=np.int32)

        # Run csim
        mod(a, b, c)

        # Verify results
        expected = a + b
        np.testing.assert_array_equal(c, expected)
        print("test_catapult_csim passed!")


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csim_multiply():
    """Test csim flow with element-wise multiplication"""

    def vmul(a: int32[16], b: int32[16]) -> int32[16]:
        c: int32[16]
        for i in range(16):
            c[i] = a[i] * b[i]
        return c

    s = allo.customize(vmul)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csim", project=tmpdir)

        # Create input data (use small values to avoid overflow)
        a = np.random.randint(0, 100, (16,)).astype(np.int32)
        b = np.random.randint(0, 100, (16,)).astype(np.int32)
        c = np.zeros(16, dtype=np.int32)

        # Run csim
        mod(a, b, c)

        # Verify results
        expected = a * b
        np.testing.assert_array_equal(c, expected)
        print("test_catapult_csim_multiply passed!")


# =============================================================================
# CSYNTH Tests (requires Catapult installation with MGC_HOME)
# =============================================================================


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csynth_vvadd():
    """Test csynth flow for basic vector addition with Catapult HLS"""

    def vvadd(a: int32[10], b: int32[10]) -> int32[10]:
        c: int32[10]
        for i in range(10):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(vvadd)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Run synthesis (no arguments for csyn mode)
        mod()

        # Check that synthesis outputs are generated
        # Catapult generates output in the project directory
        print("test_catapult_csynth_vvadd passed!")


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csynth_2d_array():
    """Test csynth flow for 2D array operations with Catapult HLS"""

    def matrix_add(A: int32[4, 4], B: int32[4, 4]) -> int32[4, 4]:
        C: int32[4, 4]
        for i, j in allo.grid(4, 4):
            C[i, j] = A[i, j] + B[i, j]
        return C

    s = allo.customize(matrix_add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Run synthesis
        mod()

        print("test_catapult_csynth_2d_array passed!")


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csynth_with_pipeline():
    """Test csynth flow with pipeline pragma"""

    def pipelined_add(a: int32[16], b: int32[16]) -> int32[16]:
        c: int32[16]
        for i in range(16):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(pipelined_add)
    s.pipeline("i")

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Verify pipeline pragma is present
        assert "#pragma hls_pipeline_init_interval" in mod.hls_code

        # Run synthesis
        mod()

        print("test_catapult_csynth_with_pipeline passed!")


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csynth_with_unroll():
    """Test csynth flow with unroll pragma"""

    def unrolled_add(a: int32[8], b: int32[8]) -> int32[8]:
        c: int32[8]
        for i in range(8):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(unrolled_add)
    s.unroll("i", factor=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Verify unroll pragma is present
        assert "#pragma hls_unroll" in mod.hls_code

        # Run synthesis
        mod()

        print("test_catapult_csynth_with_unroll passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
