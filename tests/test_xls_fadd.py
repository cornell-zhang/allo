import allo
from allo.ir.types import Fixed


# Define a fixed-point type: 16-bit total width, 8 fractional bits
Fixed16_8 = Fixed(16, 8)


def fadd(a: Fixed16_8, b: Fixed16_8) -> Fixed16_8:
    return a + b


s = allo.customize(fadd)
print("=" * 60)
print("MLIR Module (Fixed-Point Addition):")
print("=" * 60)
print(s.module)

# Note: This is a combinational function (no arrays), so use_memory has no effect
# Both modes will produce the same output
print("\n" + "=" * 60)
print("COMBINATIONAL MODE (no arrays - use_memory has no effect):")
print("=" * 60)
mod_register = s.build(target="xls", project="fadd_register.prj", use_memory=False)
print(mod_register)

print("\n" + "=" * 60)
print("COMBINATIONAL MODE with use_memory=True (same output):")
print("=" * 60)
mod_memory = s.build(target="xls", project="fadd_memory.prj", use_memory=True)
print(mod_memory)

