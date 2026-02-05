# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the traceback and error reporting facility.
"""

import pytest
import subprocess
import sys
import os
import tempfile


def run_allo_script_from_file(script_content: str) -> str:
    """Write script to a temp file and run it to capture traceback output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": os.path.dirname(os.path.dirname(__file__)),
            },
        )
        return result.stdout + result.stderr
    finally:
        os.unlink(temp_path)


def test_traceback_line_number_type_error():
    """Test that traceback reports the correct line number for type errors."""
    # Line numbers are based on the file content:
    # Line 1: from allo.ir.types import int32
    # Line 2: import allo
    # Line 3: (empty)
    # Line 4: def gemm(...)
    # Line 5: C: Undefined[32, 32] = 0  <-- Error line
    script = """from allo.ir.types import int32
import allo

def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    C: Undefined[32, 32] = 0
    for i, j, k in allo.grid(32, 32, 32, name="C"):
        C[i, j] += A[i, k] * B[k, j]
    return C

allo.customize(gemm)
"""
    output = run_allo_script_from_file(script)

    # The error is on line 5 (C: Undefined[32, 32] = 0)
    assert "Unsupported type `Undefined`" in output
    assert "Line: 5" in output
    # Verify the error line is shown
    assert "C: Undefined[32, 32] = 0" in output


def test_traceback_line_number_nested_function():
    """Test traceback line number for errors in nested functions."""
    # Line 1: from allo.ir.types import int32
    # Line 2: import allo
    # Line 3: (empty)
    # Line 4: def outer():
    # Line 5: def inner(...)
    # Line 6: B: BadType[32, 32] = 0  <-- Error line
    script = """from allo.ir.types import int32
import allo

def outer():
    def inner(A: int32[32, 32]) -> int32[32, 32]:
        B: BadType[32, 32] = 0
        return B
    allo.customize(inner)

outer()
"""
    output = run_allo_script_from_file(script)

    # The error is on line 6 (B: BadType[32, 32] = 0)
    assert "Unsupported type `BadType`" in output
    assert "Line: 6" in output
    assert "B: BadType[32, 32] = 0" in output


def test_traceback_line_number_return_type_error():
    """Test traceback line number for errors in function return types."""
    # Line 1: from allo.ir.types import int32
    # Line 2: import allo
    # Line 3: (empty)
    # Line 4: UnsupportedRetType = None  # Allo doesn't support None as a type
    # Line 5: (empty)
    # Line 6: def kernel(A: int32[32, 32]) -> UnsupportedRetType:  <-- Error line
    script = """from allo.ir.types import int32
import allo

UnsupportedRetType = None  # Allo doesn't support None as a type

def kernel(A: int32[32, 32]) -> UnsupportedRetType:
    return A

allo.customize(kernel)
"""
    output = run_allo_script_from_file(script)

    # The error is on line 6 (function definition with bad return type)
    # Check that the error correctly points to the function definition
    assert "Line: 6" in output
    assert "def kernel" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
