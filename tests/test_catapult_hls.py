import allo
from allo.ir.types import int32

def vvadd(a: int32[100], b: int32[100]) -> int32[100]:
    c: int32[100]
    for i in range(100):
        c[i] = a[i] + b[i]
    return c

s = allo.customize(vvadd)
mod = s.build(target="catapult", mode="csyn", project="my_project.prj")

# Then run synthesis:
# cd my_project.prj
# module load catapult  
# catapult -shell -f run.tcl