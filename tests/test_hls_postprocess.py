# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.backend.pynq import postprocess_hls_code_pynq
from allo.backend.vitis import extract_hls_arg_names, postprocess_hls_code

HLS_CODE = """#ifndef KERNEL_H
#define KERNEL_H
using namespace std;

ap_int<22> helper(ap_int<32> value) {
  return value * 2;
}

void top(
  int input[10],
  int output[10]
) {
  output[0] = helper(input[0]);
}
#endif
"""


def _check_top_only_has_c_linkage(code):
    assert code.count('extern "C"') == 1
    assert 'extern "C" void top(' in code
    assert 'extern "C" ap_int<22> helper' not in code
    assert "ap_int<22> helper" in code


def test_vitis_only_wraps_top_function_with_c_linkage():
    code = postprocess_hls_code(HLS_CODE, top="top", pragma=False)
    _check_top_only_has_c_linkage(code)


def test_vitis_extracts_arguments_from_c_linkage_top_function():
    code = postprocess_hls_code(HLS_CODE, top="top", pragma=True)
    assert extract_hls_arg_names(code, "top") == ["input", "output"]


def test_pynq_only_wraps_top_function_with_c_linkage():
    _check_top_only_has_c_linkage(
        postprocess_hls_code_pynq(HLS_CODE, top="top", pragma=False)
    )
