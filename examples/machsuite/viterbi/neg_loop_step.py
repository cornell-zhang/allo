import allo
from allo.ir.types import int32

N = 256

def compute(
    x: int32[N],
    y: int32[N]
):
    for i in range(N-1, -1, -1):
        y[i+1] = x[i+1]


s = allo.customize(compute)
print(s.module)

