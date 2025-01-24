import allo
from allo.ir.types import int32

N = 256
M = 16


def compute_dist(
    position_x: int32[N],
    position_y: int32[N],
    position_z: int32[N],
    NL: int32[N, M],
    del_x: int32[N],
    del_y: int32[N],
    del_z: int32[N],
):
    for i0, j0 in allo.grid(N, M):
        del_x[i0] = position_x[i0] - position_x[NL[i0, j0]]
        del_y[i0] = position_y[i0] - position_y[NL[i0, j0]]
        del_z[i0] = position_z[i0] - position_z[NL[i0, j0]]


def kernel_md(
    position_x: int32[N],
    position_y: int32[N],
    position_z: int32[N],
    NL: int32[N, M],
    force_x: int32[N],
    force_y: int32[N],
    force_z: int32[N],
):
    del_x: int32[N]
    del_y: int32[N]
    del_z: int32[N]
    compute_dist(position_x, position_y, position_z, NL, del_x, del_y, del_z)


s0 = allo.customize(compute_dist)
s0.split("i0", factor=16)
s0.pipeline("i0.inner")
s = allo.customize(kernel_md)
s.compose(s0)
