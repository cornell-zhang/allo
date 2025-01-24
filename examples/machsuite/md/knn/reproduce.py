import allo
from allo.ir.types import int32

N = 256
M = 16

def compute_dist(x: int32[N], y: int32[N], z: int32[N], NL: int32[N, M], del_x: int32[N], del_y: int32[N], del_z: int32[N]):
    for i in allo.grid(N):
        for j in allo.grid(M):
            del_x[i] = x[i] - x[NL[i, j]]
            del_y[i] = y[i] - y[NL[i, j]]
            del_z[i] = z[i] - z[NL[i, j]]

s = allo.customize(compute_dist)
s.split("i",factor=16)
s.pipeline("i.inner")

