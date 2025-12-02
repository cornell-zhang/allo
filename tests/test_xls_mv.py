import allo
from allo.ir.types import int32

def mv(A: int32[32, 32], x: int32[32]) -> int32[32]:
    y: int32[32] = 0
    for i in allo.grid(32):
        y[i] = 0
        for j in allo.grid(32):
            y[i] += A[i, j] * x[j]
    return y

s = allo.customize(mv)
s.pipeline("i")
print(s.module)
mod = s.build(target="xls", project="my_project.prj")
print(mod)