import allo
from allo.ir.types import int32
import numpy as np

def gemm(A: int32[4, 4], B: int32[4, 4]) -> int32[4, 4]:
    C: int32[4, 4] = 0
    for i, j, k in allo.grid(4, 4, 4):
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
mod_register = s.build(target="xlscc", project="gemm_register.prj", use_memory=False)
print(mod_register)

# Memory mode - arrays as __xls_memory<T, size>
print("\n" + "=" * 60) 
print("MEMORY MODE (use_memory=True):")
print("Arrays emitted as: __xls_memory<int, 1024> (flattened 32x32)")
print("=" * 60)
mod_memory = s.build(target="xlscc", project="gemm_memory.prj", use_memory=True)
print(mod_memory)

# Print the textproto
mod_memory.print_textproto()

# Run functional verification tests
print("\n" + "=" * 60)
print("FUNCTIONAL VERIFICATION (register mode):")
print("=" * 60)

from xls_test_framework import XLSTestRunner

# Create test matrices
A = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=np.int32)  # Identity matrix
B = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]], dtype=np.int32)
expected_C = (A @ B).astype(np.int32)  # Should equal B for identity A

runner = XLSTestRunner()
runner.test_sequential(
    allo_func=gemm,
    schedule=s,
    test_cases=[
        ((A.tolist(), B.tolist()), expected_C.tolist()),
    ],
    use_memory=False  # Register mode
)
runner.print_summary()