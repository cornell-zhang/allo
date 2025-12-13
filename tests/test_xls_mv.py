import allo
from allo.ir.types import int32

def mv(A: int32[4, 4], x: int32[4]) -> int32[4]:
    y: int32[4] = 0
    for i in allo.grid(32):
        y[i] = 0
        for j in allo.grid(32):
            y[i] += A[i, j] * x[j]
    return y

s = allo.customize(mv)
s.pipeline("i")
print("=" * 60)
print("MLIR Module:")
print("=" * 60)
print(s.module)

# Register mode (default) - arrays as plain C arrays
print("\n" + "=" * 60)
print("REGISTER MODE (use_memory=False):")
print("Arrays emitted as: int A[32][32], int x[32]")
print("=" * 60)
mod_register = s.build(target="xlscc", project="mv_register.prj", use_memory=False)
print(mod_register)

# Memory mode - arrays as __xls_memory<T, size>
print("\n" + "=" * 60)
print("MEMORY MODE (use_memory=True):")
print("Arrays emitted as: __xls_memory<int, 1024>, __xls_memory<int, 32>")
print("=" * 60)
mod_memory = s.build(target="xlscc", project="mv_memory.prj", use_memory=True)
print(mod_memory)
