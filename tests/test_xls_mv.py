import allo
from allo.ir.types import int32
import numpy as np

def mv(A: int32[4, 4], x: int32[4]) -> int32[4]:
    y: int32[4] = 0
    for i in allo.grid(4):
        y[i] = 0
        for j in allo.grid(4):
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
print("Arrays emitted as: int A[4][4], int x[4]")
print("=" * 60)
mod_register = s.build(target="xlscc", project="mv_register.prj", use_memory=False)
print(mod_register)

# Memory mode - arrays as __xls_memory<T, size>
print("\n" + "=" * 60)
print("MEMORY MODE (use_memory=True):")
print("Arrays emitted as: __xls_memory<int, 16>, __xls_memory<int, 4>")
print("=" * 60)
mod_memory = s.build(target="xlscc", project="mv_memory.prj", use_memory=True)
print(mod_memory)

# Print the textproto
mod_memory.print_textproto()

# Run functional verification tests
print("\n" + "=" * 60)
print("FUNCTIONAL VERIFICATION (register mode):")
print("=" * 60)

from xls_test_framework import XLSTestRunner

# Identity matrix test: I @ x = x
A_identity = np.eye(4, dtype=np.int32)
x = np.array([1, 2, 3, 4], dtype=np.int32)
expected_identity = (A_identity @ x).astype(np.int32).tolist()

# Diagonal matrix test
A_diag = np.diag([1, 2, 3, 4]).astype(np.int32)
expected_diag = (A_diag @ x).astype(np.int32).tolist()  # [1, 4, 9, 16]

# Full matrix test
A_full = np.array([[1, 1, 1, 1],
                   [2, 2, 2, 2],
                   [3, 3, 3, 3],
                   [4, 4, 4, 4]], dtype=np.int32)
expected_full = (A_full @ x).astype(np.int32).tolist()  # [10, 20, 30, 40]

runner = XLSTestRunner()
runner.test_sequential(
    allo_func=mv,
    schedule=s,
    test_cases=[
        ((A_identity.tolist(), x.tolist()), expected_identity),
        ((A_diag.tolist(), x.tolist()), expected_diag),
        ((A_full.tolist(), x.tolist()), expected_full),
    ],
    use_memory=False
)
runner.print_summary()
