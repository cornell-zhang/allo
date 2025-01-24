import allo
from allo.ir.types import int32

N = 256

def compute(
    x: int32[N],
    y: int32[N]
):
    for i in range(N):
        y[i] = x[i]

s = allo.customize(compute)
s.build(target="vitis_hls", mode="csim", project="test.prj")

