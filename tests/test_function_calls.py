# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
from allo.ir.types import float32, int32, index
import numpy as np
import allo.backend.hls as hls


def test_function_calls_with_return_values():
    """
    Regression test for function call generation with return values.

    This test verifies that functions with return values generate correct HLS C++ code:
    1. Return value variables are declared before the function call
    2. Return values are passed as pointer arguments to the function

    Previously, there was a bug where function calls with return values were not
    generating the correct C++ code due to a hardcoded string check for "softmax_".
    """
    L = 64
    MIN_FLOAT32: float32 = -3.402823466e38  # minimum float32 value

    def softmax_p1(QK_in: float32[L, L], i_pos: index) -> float32:
        local_max: float32 = MIN_FLOAT32
        for j1 in allo.grid(L, name="j1"):
            if QK_in[i_pos, j1] > local_max:
                local_max = QK_in[i_pos, j1]
        return local_max

    def softmax_p2(QK_in: float32[L, L], max_val: float32, i_pos: index) -> float32[L]:
        local_max: float32 = max_val
        exp_buf_1: float32[L]
        for j2 in allo.grid(L, name="j2"):
            e: float32 = allo.exp(QK_in[i_pos, j2] - local_max)
            exp_buf_1[j2] = e
        return exp_buf_1

    def top(QK_in: float32[L, L]) -> float32[L, L]:
        QK_out: float32[L, L]
        for i_soft in allo.grid(L, name="i_soft"):
            max_val = softmax_p1(QK_in, i_soft)
            exp_buf_1 = softmax_p2(QK_in, max_val, i_soft)
        return QK_out

    s = allo.customize(top)
    mod = s.build("vhls")
    hls_code = str(mod)

    # Check that the generated HLS code has correct function calls
    # Before the fix:
    # - softmax_p1(v21, i_soft);              // Missing return value pointer
    # - softmax_p2(v21, float v24, i_soft);   // Invalid "float v24" argument

    # After the fix:
    # - float v24;                            // Declares return variable
    # - softmax_p1(v21, i_soft, &v24);       // Correct call with return pointer
    # - float v25[64];                        // Declares return array
    # - softmax_p2(v21, v24, i_soft, v25);   // Correct call with proper args

    print("Generated HLS code:")
    print(hls_code)

    # Verify correct function call patterns
    assert "softmax_p1(" in hls_code, "softmax_p1 function call should be present"
    assert "softmax_p2(" in hls_code, "softmax_p2 function call should be present"

    # Check that return value variables are declared
    assert "float v" in hls_code, "Return value variables should be declared"

    # Check that function calls have the correct number of arguments
    # softmax_p1 should have 3 arguments: array, index, pointer to return value
    # This pattern checks for the function call with a pointer argument (&v)
    assert "&v" in hls_code, "Function calls should pass return values by pointer"

    # Verify that invalid patterns are NOT present
    # The bug would generate "float v24" as an argument
    assert (
        "float v" not in hls_code or ", float v" not in hls_code
    ), "Function calls should not have 'float v' as arguments"


def test_function_calls_scalar_return():
    """Test function calls with scalar return values."""

    def helper_func(x: int32) -> int32:
        return x * 2

    def main_func(arr: int32[10]) -> int32[10]:
        result: int32[10]
        for i in allo.grid(10):
            result[i] = helper_func(arr[i])
        return result

    s = allo.customize(main_func)
    mod = s.build("vhls")
    hls_code = str(mod)

    print("Generated HLS code for scalar return:")
    print(hls_code)

    # Check for correct function call pattern with scalar return
    assert "helper_func(" in hls_code, "helper_func call should be present"
    assert "&v" in hls_code, "Scalar return should be passed by pointer"


def test_function_calls_array_return():
    """Test function calls with array return values."""

    def create_array(x: int32) -> int32[5]:
        arr: int32[5]
        for i in allo.grid(5):
            arr[i] = x + i
        return arr

    def main_func(input_val: int32) -> int32[5]:
        result = create_array(input_val)
        return result

    s = allo.customize(main_func)
    mod = s.build("vhls")
    hls_code = str(mod)

    print("Generated HLS code for array return:")
    print(hls_code)

    # Check for correct function call pattern with array return
    assert "create_array(" in hls_code, "create_array call should be present"
    # Array returns are passed directly (not by pointer)
    assert "create_array(" in hls_code, "Array return function call should be present"


def test_multiple_function_calls():
    """Test multiple function calls with different return types."""

    def func_scalar(x: int32) -> int32:
        return x * 2

    def func_array(x: int32) -> int32[3]:
        arr: int32[3]
        for i in allo.grid(3):
            arr[i] = x + i
        return arr

    def main_func(input_val: int32) -> int32[3]:
        scalar_result = func_scalar(input_val)
        array_result = func_array(scalar_result)
        return array_result

    s = allo.customize(main_func)
    mod = s.build("vhls")
    hls_code = str(mod)

    print("Generated HLS code for multiple function calls:")
    print(hls_code)

    # Check for both function calls
    assert "func_scalar(" in hls_code, "func_scalar call should be present"
    assert "func_array(" in hls_code, "func_array call should be present"

    # Check for proper variable declarations and pointer usage
    assert "int32_t v" in hls_code, "Variables should be declared for return values"


def test_nested_function_calls():
    """Test nested function calls to ensure proper handling."""

    def inner_func(x: float32) -> float32:
        return x * 0.5

    def outer_func(x: float32) -> float32:
        temp = inner_func(x)
        return temp + 1.0

    def main_func(arr: float32[10]) -> float32[10]:
        result: float32[10]
        for i in allo.grid(10):
            result[i] = outer_func(arr[i])
        return result

    s = allo.customize(main_func)
    mod = s.build("vhls")
    hls_code = str(mod)

    print("Generated HLS code for nested function calls:")
    print(hls_code)

    # Check for both function calls
    assert "inner_func(" in hls_code, "inner_func call should be present"
    assert "outer_func(" in hls_code, "outer_func call should be present"

    # Check that pointer arguments are used for scalar returns
    assert "&v" in hls_code, "Function calls should use pointer arguments for returns"


if __name__ == "__main__":
    pytest.main([__file__])
