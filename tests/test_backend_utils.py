# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for backend utility functions that do not require Vitis/Catapult HLS.

Covers:
  - postprocess_hls_code: stripping MLIR SSA %identifiers from generated C++
  - resolve_nb_type: mapping HLS ap_int<N>/ap_uint<N> to nanobind-compatible stdint types
  - parse_cpp_function: parsing C++ function signatures including ap_int parameter types
"""

import os
import tempfile
import pytest
from allo.backend.ip import resolve_nb_type, parse_cpp_function, IPModule
from allo.backend.vitis import postprocess_hls_code


# ---------------------------------------------------------------------------
# postprocess_hls_code: %alloc stripping
# ---------------------------------------------------------------------------


def test_postprocess_strips_percent_alloc():
    """MLIR SSA names like %alloc must be stripped to plain identifiers."""
    hls_code = "int %alloc;\nfloat %alloc1;\n"
    result = postprocess_hls_code(hls_code, top=None, pragma=False)
    assert "%alloc" not in result
    assert "alloc" in result
    assert "%alloc1" not in result
    assert "alloc1" in result


def test_postprocess_strips_generic_percent_ident():
    """Any %word pattern should be stripped."""
    hls_code = "return %result;\n"
    result = postprocess_hls_code(hls_code, top=None, pragma=False)
    assert "%" not in result
    assert "result" in result


def test_postprocess_preserves_modulo_operator():
    """The C++ modulo operator (% followed by non-word char) must not be touched."""
    # 'x % 4' — space after %, not a word char, should be left alone
    hls_code = "int y = x % 4;\n"
    result = postprocess_hls_code(hls_code, top=None, pragma=False)
    assert "x % 4" in result


def test_postprocess_realistic_mlir_snippet():
    """Realistic MLIR-emitted snippet: %alloc stripped, C++ modulo preserved."""
    # Reproduces the exact pattern that caused g++ to fail in test_three_level_systolic_csim
    hls_code = (
        "void top(int16_t %alloc[4][4]) {\n"
        "  int16_t %alloc1 = 0;\n"
        "  int idx = i % 4;\n"  # C++ modulo — must be preserved
        "}\n"
    )
    result = postprocess_hls_code(hls_code, top=None, pragma=False)
    assert "%alloc" not in result
    assert "%alloc1" not in result
    assert "alloc[4][4]" in result
    assert "alloc1" in result
    assert "i % 4" in result  # C++ modulo untouched


# ---------------------------------------------------------------------------
# resolve_nb_type: ap_int / ap_uint → stdint mapping
# ---------------------------------------------------------------------------


def test_resolve_nb_type_ap_int():
    assert resolve_nb_type("ap_int<8>") == "int8_t"
    assert resolve_nb_type("ap_int<16>") == "int16_t"
    assert resolve_nb_type("ap_int<32>") == "int32_t"
    assert resolve_nb_type("ap_int<64>") == "int64_t"


def test_resolve_nb_type_ap_uint():
    assert resolve_nb_type("ap_uint<8>") == "uint8_t"
    assert resolve_nb_type("ap_uint<16>") == "uint16_t"
    assert resolve_nb_type("ap_uint<32>") == "uint32_t"


def test_resolve_nb_type_passthrough():
    """Non-ap_int types should be returned unchanged."""
    assert resolve_nb_type("float") == "float"
    assert resolve_nb_type("int") == "int"
    assert resolve_nb_type("double") == "double"


# ---------------------------------------------------------------------------
# parse_cpp_function: ap_int parameter types
# ---------------------------------------------------------------------------


def test_parse_cpp_function_plain_types():
    """Regression: plain types (int8_t, float, int) still parse correctly after regex change."""
    code = "void top(int8_t A[4], float B[2][3], int c) {}"
    result = parse_cpp_function(code, "top")
    assert result is not None
    assert result[0] == ("int8_t", (4,))
    assert result[1] == ("float", (2, 3))
    assert result[2] == ("int", ())


def test_parse_cpp_function_ap_int_scalar():
    """ap_int<8> scalar parameter should be parsed with type 'ap_int<8>' and shape ()."""
    code = "void my_kernel(ap_int<8> x, ap_uint<16> y) {}"
    result = parse_cpp_function(code, "my_kernel")
    assert result is not None
    assert len(result) == 2
    assert result[0] == ("ap_int<8>", ())
    assert result[1] == ("ap_uint<16>", ())


def test_parse_cpp_function_ap_int_array():
    """ap_int<8> array parameter should be parsed with correct shape."""
    code = "void my_kernel(ap_int<8> buf[4][8]) {}"
    result = parse_cpp_function(code, "my_kernel")
    assert result is not None
    assert len(result) == 1
    assert result[0] == ("ap_int<8>", (4, 8))


def test_parse_cpp_function_mixed():
    """Mix of plain and ap_int types should all parse correctly."""
    code = "void top(float A[4], ap_int<16> B[2], int c) {}"
    result = parse_cpp_function(code, "top")
    assert result is not None
    assert result[0] == ("float", (4,))
    assert result[1] == ("ap_int<16>", (2,))
    assert result[2] == ("int", ())


# ---------------------------------------------------------------------------
# generate_nanobind_wrapper: ap_int types use stdint in the wrapper interface
# ---------------------------------------------------------------------------


def test_generate_nanobind_wrapper_uses_stdint_for_ap_int():
    """Wrapper signature must use int8_t/int16_t, not ap_int<N>, for nanobind compatibility."""
    impl_code = "void my_top(ap_int<8> A[4], ap_uint<16> B[4]) {}\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        impl_path = os.path.join(tmpdir, "my_top.cpp")
        with open(impl_path, "w") as f:
            f.write(impl_code)
        ip = IPModule("my_top", impl_path, link_hls=False)
        wrapper_path = ip.generate_nanobind_wrapper()
        with open(wrapper_path) as f:
            wrapper = f.read()
    # ap_int<8> → int8_t in the nanobind interface
    assert "int8_t" in wrapper
    assert "uint16_t" in wrapper
    # Original HLS types used in the cast inside the body
    assert "ap_int<8>" in wrapper
    assert "ap_uint<16>" in wrapper
    # The raw ap_int type must NOT appear in the function signature line
    sig_lines = [ln for ln in wrapper.splitlines() if "nb::ndarray" in ln]
    assert all("ap_int" not in ln for ln in sig_lines)
    assert all("ap_uint" not in ln for ln in sig_lines)
