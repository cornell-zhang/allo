# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for backend utility functions that do not require Vitis/Catapult HLS.

Covers:
  - postprocess_hls_code: stripping MLIR SSA %identifiers from generated C++
  - resolve_nb_type: mapping HLS ap_int<N>/ap_uint<N> to nanobind-compatible stdint types
  - parse_cpp_function: parsing C++ function signatures including ap_int parameter types
"""

import pytest
from allo.backend.ip import resolve_nb_type, parse_cpp_function
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
