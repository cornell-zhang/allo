#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick test script for Intel HLS and Tapa HLS backends
to verify they still work after the ModuleEmitter refactoring.
"""

import allo
from allo.ir.types import int32
import numpy as np


def test_intel_hls():
    """Test Intel HLS backend code generation."""
    print("=" * 60)
    print("Testing Intel HLS Backend")
    print("=" * 60)
    
    def top(A: int32[4]) -> int32[4]:
        B: int32[4]
        for i in range(4):
            B[i] = A[i] + 1
        return B
    
    s = allo.customize(top)
    mod = s.build(target="ihls")
    
    # Check for Intel-specific syntax
    assert "intel::kernel_args_restrict" in mod.hls_code or "single_task" in mod.hls_code, \
        "Intel HLS code should contain Intel-specific syntax"
    
    print("✓ Intel HLS code generation successful!")
    print(f"Generated code length: {len(mod.hls_code)} characters")
    print(f"First 300 chars:\n{mod.hls_code[:300]}\n")


def test_tapa_hls():
    """Test Tapa HLS backend code generation."""
    print("=" * 60)
    print("Testing Tapa HLS Backend")
    print("=" * 60)
    
    def top(A: int32[4]) -> int32[4]:
        B: int32[4]
        for i in range(4):
            B[i] = A[i] + 1
        return B
    
    s = allo.customize(top)
    mod = s.build(target="tapa")
    
    # Check that code was generated (Tapa has specific patterns)
    assert len(mod.hls_code) > 0, "Tapa HLS should generate code"
    
    print("✓ Tapa HLS code generation successful!")
    print(f"Generated code length: {len(mod.hls_code)} characters")
    print(f"First 300 chars:\n{mod.hls_code[:300]}\n")


def test_vivado_hls():
    """Test Vivado HLS backend code generation (baseline)."""
    print("=" * 60)
    print("Testing Vivado HLS Backend (baseline)")
    print("=" * 60)
    
    def top(A: int32[4]) -> int32[4]:
        B: int32[4]
        for i in range(4):
            B[i] = A[i] + 1
        return B
    
    s = allo.customize(top)
    mod = s.build(target="vhls")
    
    assert len(mod.hls_code) > 0, "Vivado HLS should generate code"
    
    print("✓ Vivado HLS code generation successful!")
    print(f"Generated code length: {len(mod.hls_code)} characters")
    print(f"First 300 chars:\n{mod.hls_code[:300]}\n")


if __name__ == "__main__":
    try:
        test_vivado_hls()
        test_intel_hls()
        test_tapa_hls()
        print("=" * 60)
        print("All HLS backend tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

