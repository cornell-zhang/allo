"""Compare DSLX output from dslx/ and xls_fn/ backends for test3 and test4."""

import allo
from allo.ir.types import int32
from allo.backend.xls_fn import DslxProcModule as XlsFnProcModule
from allo.backend.dslx import DSLXModule as DslxModule


def test3_accumulator():
    """Test 3: accumulator function."""
    def accumulator(a: int32) -> int32:
        state = 0
        for i in range(10):
            state = state + a
        return state

    s = allo.customize(accumulator)
    mod = s.build(target="xls", project="test_accum.prj")
    return str(mod.module), "accumulator"


def test4_vector_add():
    """Test 4: vector_add function."""
    def vector_add(a: int32[8], b: int32[8], c: int32[8]):
        for i in range(8):
            c[i] = a[i] + b[i]

    s = allo.customize(vector_add)
    mod = s.build(target="xls", project="test_vadd.prj")
    return str(mod.module), "vector_add"


def compare_backends():
    """Compare outputs from both backends."""
    print("=" * 70)
    print("COMPARING DSLX BACKENDS: dslx/ vs xls_fn/")
    print("=" * 70)
    print()

    # Test 3: Accumulator
    print("\n" + "=" * 70)
    print("TEST 3: Accumulator")
    print("=" * 70)
    print()
    
    mlir_str, func_name = test3_accumulator()
    
    print("Generating with xls_fn backend...")
    try:
        xls_fn_mod = XlsFnProcModule(mlir_str, func_name)
        xls_fn_code = xls_fn_mod.codegen()
        print("✓ xls_fn backend succeeded")
    except Exception as e:
        print(f"✗ xls_fn backend failed: {e}")
        import traceback
        traceback.print_exc()
        xls_fn_code = None
    
    print("\nGenerating with dslx backend...")
    try:
        from allo._mlir.ir import Context, Module as MlirModule
        with Context() as ctx:
            dslx_mlir = MlirModule.parse(mlir_str, ctx)
        dslx_mod = DslxModule(dslx_mlir, func_name)
        dslx_code = dslx_mod.codegen()
        print("✓ dslx backend succeeded")
    except Exception as e:
        print(f"✗ dslx backend failed: {e}")
        import traceback
        traceback.print_exc()
        dslx_code = None
    
    if xls_fn_code and dslx_code:
        print("\n" + "-" * 70)
        print("xls_fn OUTPUT:")
        print("-" * 70)
        print(xls_fn_code)
        print("\n" + "-" * 70)
        print("dslx OUTPUT:")
        print("-" * 70)
        print(dslx_code)
        print("\n" + "-" * 70)
        print("COMPARISON:")
        print("-" * 70)
        if xls_fn_code == dslx_code:
            print("✓ OUTPUTS ARE IDENTICAL")
        else:
            print("✗ OUTPUTS DIFFER")
            # Show differences
            xls_fn_lines = xls_fn_code.split('\n')
            dslx_lines = dslx_code.split('\n')
            max_len = max(len(xls_fn_lines), len(dslx_lines))
            for i in range(max_len):
                xls_line = xls_fn_lines[i] if i < len(xls_fn_lines) else ""
                dslx_line = dslx_lines[i] if i < len(dslx_lines) else ""
                if xls_line != dslx_line:
                    print(f"Line {i+1}:")
                    print(f"  xls_fn: {xls_line}")
                    print(f"  dslx:   {dslx_line}")
    
    # Test 4: Vector Add
    print("\n\n" + "=" * 70)
    print("TEST 4: Vector Add")
    print("=" * 70)
    print()
    
    mlir_str, func_name = test4_vector_add()
    
    print("Generating with xls_fn backend...")
    try:
        xls_fn_mod = XlsFnProcModule(mlir_str, func_name)
        xls_fn_code = xls_fn_mod.codegen()
        print("✓ xls_fn backend succeeded")
    except Exception as e:
        print(f"✗ xls_fn backend failed: {e}")
        import traceback
        traceback.print_exc()
        xls_fn_code = None
    
    print("\nGenerating with dslx backend...")
    try:
        from allo._mlir.ir import Context, Module as MlirModule
        with Context() as ctx:
            dslx_mlir = MlirModule.parse(mlir_str, ctx)
        dslx_mod = DslxModule(dslx_mlir, func_name)
        dslx_code = dslx_mod.codegen()
        print("✓ dslx backend succeeded")
    except Exception as e:
        print(f"✗ dslx backend failed: {e}")
        import traceback
        traceback.print_exc()
        dslx_code = None
    
    if xls_fn_code and dslx_code:
        print("\n" + "-" * 70)
        print("xls_fn OUTPUT:")
        print("-" * 70)
        print(xls_fn_code)
        print("\n" + "-" * 70)
        print("dslx OUTPUT:")
        print("-" * 70)
        print(dslx_code)
        print("\n" + "-" * 70)
        print("COMPARISON:")
        print("-" * 70)
        if xls_fn_code == dslx_code:
            print("✓ OUTPUTS ARE IDENTICAL")
        else:
            print("✗ OUTPUTS DIFFER")
            # Show differences
            xls_fn_lines = xls_fn_code.split('\n')
            dslx_lines = dslx_code.split('\n')
            max_len = max(len(xls_fn_lines), len(dslx_lines))
            diff_count = 0
            for i in range(max_len):
                xls_line = xls_fn_lines[i] if i < len(xls_fn_lines) else ""
                dslx_line = dslx_lines[i] if i < len(dslx_lines) else ""
                if xls_line != dslx_line:
                    diff_count += 1
                    if diff_count <= 10:  # Show first 10 differences
                        print(f"Line {i+1}:")
                        print(f"  xls_fn: {xls_line}")
                        print(f"  dslx:   {dslx_line}")
            if diff_count > 10:
                print(f"... and {diff_count - 10} more differences")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    compare_backends()
