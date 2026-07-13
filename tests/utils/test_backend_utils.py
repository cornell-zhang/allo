# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for backend utility functions that do not require Vitis/Catapult HLS.

Covers:
  - analyze_use_def name derivation: SSA sigils ("%") must not leak into the
    "name" attribute or the emitted HLS C++
  - postprocess_hls_code: it must never mangle the C++ modulo operator
  - resolve_nb_type: mapping HLS ap_int<N>/ap_uint<N> to nanobind-compatible stdint types
  - parse_cpp_function: parsing C++ function signatures including ap_int parameter types
"""

import os
import re
import tempfile
import pytest
import allo
from allo.ir.types import int8, int16
from allo.backend.ip import resolve_nb_type, parse_cpp_function, IPModule
from allo.backend.vitis import postprocess_hls_code


# ---------------------------------------------------------------------------
# analyze_use_def name derivation: no stray SSA sigils in emitted HLS
# ---------------------------------------------------------------------------


def test_no_ssa_sigil_in_emitted_hls_code():
    """Names derived from textual SSA ids must not leak "%" into HLS C++.

    Mirrors tests/test_systolic_array.py::test_three_level_systolic_csim up to
    HLS code emission (buffer_at + partition give an alloc a textual-SSA-derived
    name). This exercises the analyze_use_def naming path via the vhls backend
    and is Vitis-free and fast.
    """
    M, N, K = 4, 4, 4

    def gemm(A: int8[M, K], B: int8[K, N], C: int16[M, N]):
        for i, j in allo.grid(M, N, name="PE"):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]

    s = allo.customize(gemm)
    buf_A = s.buffer_at(s.A, "i")
    buf_B = s.buffer_at(s.B, "j")
    pe = s.unfold("PE", [0, 1])
    s.partition(s.C, dim=0)
    s.partition(s.A, dim=1)
    s.partition(s.B, dim=2)
    s.to(buf_A, pe, axis=1, depth=M + 1)
    s.to(buf_B, pe, axis=0, depth=N + 1)

    code = str(s.build(target="vhls"))
    # "%" followed by a word char is exclusively the MLIR SSA sigil; it is
    # illegal in C++ and must never survive into the emitted kernel.
    assert not re.findall(r"%\w+", code)


# ---------------------------------------------------------------------------
# postprocess_hls_code: must not mangle the C++ modulo operator
# ---------------------------------------------------------------------------


def test_postprocess_preserves_modulo_operator():
    """The C++ modulo operator must survive postprocess_hls_code untouched."""
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
