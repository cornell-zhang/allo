#!/usr/bin/env python3
"""
XLS[cc] Functionality Tests

This file tests the generated XLS[cc] C++ code for correctness.
Run with: python test_xls_functionality.py

Output:
  . = passed
  F = failed
"""

import sys
import os

# Add the allo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import allo
from allo.ir.types import int32
from xls_test_framework import XLSTestRunner


def test_add():
    """Test simple addition (combinational)."""
    def add(a: int32, b: int32) -> int32:
        return a + b
    
    s = allo.customize(add)
    
    runner = XLSTestRunner()
    runner.test_combinational(
        allo_func=add,
        schedule=s,
        test_cases=[
            ((2, 3), 5),
            ((10, -5), 5),
            ((0, 0), 0),
            ((-1, 1), 0),
            ((100, 200), 300),
            ((-50, -50), -100),
        ]
    )
    return runner


def test_multiply():
    """Test simple multiplication (combinational)."""
    def mul(a: int32, b: int32) -> int32:
        return a * b
    
    s = allo.customize(mul)
    
    runner = XLSTestRunner()
    runner.test_combinational(
        allo_func=mul,
        schedule=s,
        test_cases=[
            ((2, 3), 6),
            ((10, -5), -50),
            ((0, 100), 0),
            ((-2, -3), 6),
            ((7, 8), 56),
        ]
    )
    return runner


def test_subtract():
    """Test simple subtraction (combinational)."""
    def sub(a: int32, b: int32) -> int32:
        return a - b
    
    s = allo.customize(sub)
    
    runner = XLSTestRunner()
    runner.test_combinational(
        allo_func=sub,
        schedule=s,
        test_cases=[
            ((5, 3), 2),
            ((3, 5), -2),
            ((0, 0), 0),
            ((-1, -1), 0),
            ((100, 50), 50),
        ]
    )
    return runner


def test_complex_expr():
    """Test complex expression (combinational)."""
    def expr(a: int32, b: int32) -> int32:
        return (a + b) * 2 - a
    
    s = allo.customize(expr)
    
    runner = XLSTestRunner()
    runner.test_combinational(
        allo_func=expr,
        schedule=s,
        test_cases=[
            ((2, 3), (2 + 3) * 2 - 2),  # 10 - 2 = 8
            ((5, 5), (5 + 5) * 2 - 5),  # 20 - 5 = 15
            ((0, 0), 0),
            ((1, 0), 1),  # (1 + 0) * 2 - 1 = 1
        ]
    )
    return runner


def test_vvadd():
    """Test vector addition (sequential with arrays)."""
    def vvadd(v0: int32[4], v1: int32[4]) -> int32[4]:
        c: int32[4] = 0
        for i in allo.grid(4):
            c[i] = v0[i] + v1[i]
        return c
    
    s = allo.customize(vvadd)
    s.pipeline("i")
    
    runner = XLSTestRunner()
    runner.test_sequential(
        allo_func=vvadd,
        schedule=s,
        test_cases=[
            (([1, 2, 3, 4], [5, 6, 7, 8]), [6, 8, 10, 12]),
            (([0, 0, 0, 0], [1, 1, 1, 1]), [1, 1, 1, 1]),
            (([-1, -2, -3, -4], [1, 2, 3, 4]), [0, 0, 0, 0]),
        ],
        use_memory=False  # Register mode for simpler testing
    )
    return runner


def test_vvadd_16():
    """Test vector addition with size 16 (like test_xls_vvadd.py)."""
    def vvadd(a: int32[16], b: int32[16]) -> int32[16]:
        c: int32[16] = 0
        for i in allo.grid(16):
            c[i] = a[i] + b[i]
        return c
    
    s = allo.customize(vvadd)
    s.pipeline("i")
    
    import numpy as np
    a = np.arange(16, dtype=np.int32)
    b = np.arange(16, 32, dtype=np.int32)
    expected = (a + b).tolist()
    
    runner = XLSTestRunner()
    runner.test_sequential(
        allo_func=vvadd,
        schedule=s,
        test_cases=[
            ((a.tolist(), b.tolist()), expected),
        ],
        use_memory=False
    )
    return runner


def test_mv():
    """Test matrix-vector multiplication."""
    def mv(A: int32[4, 4], x: int32[4]) -> int32[4]:
        y: int32[4] = 0
        for i in allo.grid(4):
            y[i] = 0
            for j in allo.grid(4):
                y[i] += A[i, j] * x[j]
        return y
    
    s = allo.customize(mv)
    s.pipeline("i")
    
    import numpy as np
    
    # Identity matrix test
    A_identity = np.eye(4, dtype=np.int32)
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    expected_identity = (A_identity @ x).astype(np.int32).tolist()
    
    # Simple matrix test
    A_simple = np.array([[1, 0, 0, 0],
                          [0, 2, 0, 0],
                          [0, 0, 3, 0],
                          [0, 0, 0, 4]], dtype=np.int32)
    expected_simple = (A_simple @ x).astype(np.int32).tolist()
    
    runner = XLSTestRunner()
    runner.test_sequential(
        allo_func=mv,
        schedule=s,
        test_cases=[
            ((A_identity.tolist(), x.tolist()), expected_identity),
            ((A_simple.tolist(), x.tolist()), expected_simple),
        ],
        use_memory=False
    )
    return runner


def run_all_tests():
    """Run all XLS functionality tests."""
    print("XLS[cc] Functionality Tests")
    print("=" * 60)
    
    all_runners = []
    
    print("\nTest: add", end=" ")
    all_runners.append(test_add())
    
    print("\nTest: multiply", end=" ")
    all_runners.append(test_multiply())
    
    print("\nTest: subtract", end=" ")
    all_runners.append(test_subtract())
    
    print("\nTest: complex_expr", end=" ")
    all_runners.append(test_complex_expr())
    
    print("\nTest: vvadd (sequential)", end=" ")
    all_runners.append(test_vvadd())
    
    print("\nTest: vvadd_16 (size 16)", end=" ")
    all_runners.append(test_vvadd_16())
    
    print("\nTest: mv (mat-vec)", end=" ")
    all_runners.append(test_mv())
    
    # Summary
    total_passed = sum(sum(1 for r in runner.results if r.passed) for runner in all_runners)
    total_failed = sum(sum(1 for r in runner.results if not r.passed) for runner in all_runners)
    
    print(f"\n\n{'=' * 60}")
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    if total_failed > 0:
        print("\nFailed tests:")
        for runner in all_runners:
            for r in runner.results:
                if not r.passed:
                    print(f"  - {r.test_name}: expected {r.expected}, got {r.actual}")
                    if r.error_message:
                        # Print first line of error
                        first_line = r.error_message.split('\n')[0][:80]
                        print(f"    Error: {first_line}")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

