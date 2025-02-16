import numpy as np
import allo
from allo.ir.types import int8, int16, index


def gemm(
    A: int8[4, 4],
    B: int8[4, 4],
    C: int16[4, 4]
):
    for i, j in allo.grid(4, 4, name="PE"):
        for k in range(4):
            C[i, j] += A[i, k] * B[k, j]


s = allo.customize(gemm)
s.prepare_systolic("PE")
print(s.module)
# s.partition(s.C, dim=0)
# s.partition(s.A, dim=1)
# s.partition(s.B, dim=2)
# pe = s.unfold("PE", [0, 1])
mod = s.build()
A = np.random.randint(0, 10, size=(4, 4), dtype=np.int8)
B = np.random.randint(0, 10, size=(4, 4), dtype=np.int8)
C = np.zeros((4, 4), dtype=np.int16)
mod(A, B, C)
print(C)
np_C = A.astype(np.int32) @ B.astype(np.int32)
print(np_C)