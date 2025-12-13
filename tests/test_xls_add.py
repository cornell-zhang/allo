import allo
from allo.ir.types import int32


def add(a: int32, b: int32) -> int32:
    return a + b

s = allo.customize(add)
print("=" * 60)
print("MLIR Module:")
print("=" * 60)
print(s.module)

# Note: This is a combinational function (no arrays), so use_memory has no effect
# Both modes will produce the same output
print("\n" + "=" * 60)
print("COMBINATIONAL MODE (no arrays - use_memory has no effect):")
print("=" * 60)
mod_register = s.build(target="xlscc", project="add_register.prj", use_memory=False)
print(mod_register)

print("\n" + "=" * 60)
print("COMBINATIONAL MODE with use_memory=True (same output):")
print("=" * 60)
mod_memory = s.build(target="xlscc", project="add_memory.prj", use_memory=True)
print(mod_memory)

# Run functional verification tests
print("\n" + "=" * 60)
print("FUNCTIONAL VERIFICATION:")
print("=" * 60)

from xls_test_framework import XLSTestRunner

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
runner.print_summary()
