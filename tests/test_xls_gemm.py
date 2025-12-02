import allo
from allo.ir.types import int32

def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    C: int32[32, 32] = 0
    for i, j, k in allo.grid(32, 32, 32):
        C[i, j] += A[i, k] * B[k, j]
    return C

s = allo.customize(gemm)
s.pipeline("i")
print(s.module)
mod = s.build(target="xls", project="my_project.prj")
print(mod)