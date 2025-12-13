import allo
from allo.ir.types import int32

def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    C: int32[32, 32] = 0
    for i, j, k in allo.grid(32, 32, 32):
        C[i, j] += A[i, k] * B[k, j]
    return C

s = allo.customize(gemm)
s.pipeline("i")
print("=" * 60)
print("MLIR Module:")
print("=" * 60)
print(s.module)

# Register mode (default) - arrays as plain C arrays
print("\n" + "=" * 60)
print("REGISTER MODE (use_memory=False):")
print("Arrays emitted as: int arr[32][32]")
print("=" * 60)
mod_register = s.build(target="xls", project="gemm_register.prj", use_memory=False)
print(mod_register)

# Memory mode - arrays as __xls_memory<T, size>
print("\n" + "=" * 60)
print("MEMORY MODE (use_memory=True):")
print("Arrays emitted as: __xls_memory<int, 1024> (flattened 32x32)")
print("=" * 60)
mod_memory = s.build(target="xls", project="gemm_memory.prj", use_memory=True)
print(mod_memory)
