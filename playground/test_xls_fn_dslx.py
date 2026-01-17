"""Test script to see DSLX code emitted by xls_fn backend.

This script demonstrates both function-based and proc-based DSLX lowering.
"""

import numpy as np
import allo
from allo.ir.types import int32
from allo.backend.xls_fn import DslxFunctionModule, DslxProcModule


def test_function_lowering():
    """Test DSLX function lowering (combinational)."""
    print("=" * 70)
    print("TEST 1: Function-Based DSLX Lowering")
    print("=" * 70)
    print()

    def simple_add(a: int32, b: int32) -> int32:
        return a + b

    s = allo.customize(simple_add)
    
    # Build for XLS backend
    mod = s.build(target="xls", project="test_add.prj")
    
    # Use xls_fn to generate DSLX function
    # mod.module contains the MLIR module
    dslx_mod = DslxFunctionModule(str(mod.module), "simple_add")
    dslx_code = dslx_mod.codegen()
    
    print("Generated DSLX Function Code:")
    print("-" * 70)
    print(dslx_code)
    print()


def test_function_with_loops():
    """Test DSLX function lowering with loops."""
    print("=" * 70)
    print("TEST 2: Function-Based DSLX with Loops")
    print("=" * 70)
    print()

    def dot_product(a: int32[4], b: int32[4]) -> int32:
        result = 0
        for i in range(4):
            result += a[i] * b[i]
        return result

    s = allo.customize(dot_product)
    mod = s.build(target="xls", project="test_dot.prj")
    
    dslx_mod = DslxFunctionModule(str(mod.module), "dot_product")
    dslx_code = dslx_mod.codegen()
    
    print("Generated DSLX Function Code:")
    print("-" * 70)
    print(dslx_code)
    print()


def test_proc_lowering():
    """Test DSLX proc lowering (stateful with channels)."""
    print("=" * 70)
    print("TEST 3: Proc-Based DSLX Lowering (Stateful)")
    print("=" * 70)
    print()

    def accumulator(a: int32) -> int32:
        state = 0
        for i in range(10):
            state = state + a
        return state

    s = allo.customize(accumulator)
    mod = s.build(target="xls", project="test_accum.prj")
    
    # Use proc lowerer
    dslx_mod = DslxProcModule(str(mod.module), "accumulator")
    dslx_code = dslx_mod.codegen()
    
    print("Generated DSLX Proc Code:")
    print("-" * 70)
    print(dslx_code)
    print()


def test_proc_with_memory():
    """Test DSLX proc lowering with memory arrays."""
    print("=" * 70)
    print("TEST 4: Proc-Based DSLX with Memory")
    print("=" * 70)
    print()

    def vector_add(a: int32[8], b: int32[8], c: int32[8]):
        for i in range(8):
            c[i] = a[i] + b[i]

    s = allo.customize(vector_add)
    mod = s.build(target="xls", project="test_vadd.prj")
    
    dslx_mod = DslxProcModule(str(mod.module), "vector_add")
    dslx_code = dslx_mod.codegen()
    
    print("Generated DSLX Proc Code (with memory channels):")
    print("-" * 70)
    print(dslx_code)
    print()


def test_simple_comb_proc():
    """Test simple combinational proc (no loops)."""
    print("=" * 70)
    print("TEST 5: Combinational Proc (No Loops)")
    print("=" * 70)
    print()

    def add(a: int32, b: int32) -> int32:
        return a + b

    s = allo.customize(add)
    mod = s.build(target="xls", project="test_add_proc.prj")
    
    dslx_mod = DslxProcModule(str(mod.module), "add")
    dslx_code = dslx_mod.codegen()
    
    print("Generated DSLX Proc Code (combinational):")
    print("-" * 70)
    print(dslx_code)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("XLS_FN DSLX Code Generation Test")
    print("=" * 70)
    print()

    try:
        test_function_lowering()
    except Exception as e:
        print(f"Error in test_function_lowering: {e}")
        import traceback
        traceback.print_exc()
        print()

    try:
        test_function_with_loops()
    except Exception as e:
        print(f"Error in test_function_with_loops: {e}")
        import traceback
        traceback.print_exc()
        print()

    try:
        test_proc_lowering()
    except Exception as e:
        print(f"Error in test_proc_lowering: {e}")
        import traceback
        traceback.print_exc()
        print()

    try:
        test_proc_with_memory()
    except Exception as e:
        print(f"Error in test_proc_with_memory: {e}")
        import traceback
        traceback.print_exc()
        print()

    try:
        test_simple_comb_proc()
    except Exception as e:
        print(f"Error in test_simple_comb_proc: {e}")
        import traceback
        traceback.print_exc()
        print()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)
