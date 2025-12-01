import allo
from allo.ir.types import int32

def add(a: int32, b: int32) -> int32:
    return a + b

s = allo.customize(add)
print(s.module)
mod = s.build(target="xls", project="my_project.prj")
print(mod)