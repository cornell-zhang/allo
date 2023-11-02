import allo
import numpy as np
from allo.ir.types import float32


def top_bicg(size="mini"):
    if size == "mini" or size is None:
        M = 38
        N = 42
    elif size == "small":
        M = 116
        N = 124
    elif size == "medium":
        M = 390
        N = 410

    def kernel_bicg(
        A: float32[N, M], p: float32[M], r: float32[N], q: float32[N], s: float32[M]
    ):
        for i0 in allo.grid(N):
            for j0 in allo.reduction(M):
                q[i0] = +A[i0, j0] * p[j0]

        for i1 in allo.grid(M):
            for j1 in allo.reduction(N):
                s[i1] = +A[j1, i1] * r[j1]

    s0 = allo.customize(kernel_bicg)
    orig = s0.build("vhls")

    s = allo.customize(kernel_bicg)
    s.unroll("j0")
    s.unroll("j1")
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_bicg()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "bicg", liveout_vars="v4")
