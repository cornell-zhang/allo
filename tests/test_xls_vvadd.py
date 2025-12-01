import allo
from allo.ir.types import int32

def vvadd(a: int32[16], b: int32[16]) -> int32[16]:
    c: int32[16] = 0
    for i in allo.grid(16):
        c[i] = a[i] + b[i]
    return c

s = allo.customize(vvadd)
s.pipeline("i")
print(s.module)
mod = s.build(target="xls", project="my_project.prj")
print(mod)