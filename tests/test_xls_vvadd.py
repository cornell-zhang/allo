import allo
from allo.ir.types import int32

def vvadd(a: int32[16], b: int32[16]) -> int32[16]:
    c: int32[16] = 0
    for i in allo.grid(16):
        c[i] = a[i] + b[i]
    return c

s = allo.customize(vvadd)
s.pipeline("i")
print("=" * 60)
print("MLIR Module:")
print("=" * 60)
print(s.module)

# Register mode (default) - arrays as plain C arrays
print("\n" + "=" * 60)
print("REGISTER MODE (use_memory=False):")
print("Arrays emitted as: int arr[16]")
print("=" * 60)
mod_register = s.build(target="xlscc", project="vvadd_register.prj", use_memory=False)
print(mod_register)

# Memory mode - arrays as __xls_memory<T, size>
print("\n" + "=" * 60)
print("MEMORY MODE (use_memory=True):")
print("Arrays emitted as: __xls_memory<int, 16>")
print("=" * 60)
mod_memory = s.build(target="xlscc", project="vvadd_memory.prj", use_memory=True)
print(mod_memory)

# Print the textproto
mod_memory.print_textproto()
