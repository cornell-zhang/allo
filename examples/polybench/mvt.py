import allo
import numpy as np
from allo.ir.types import int32


def top_mvt(size="mini"):
    if size == "mini" or size is None:
        N = 40
    elif size == "small":
        N = 120
    elif size == "medium":
        N = 400

    def kernel_mvt(
        A: int32[N, N], y1: int32[N], y2: int32[N], x1: int32[N], x2: int32[N]
    ):
        for i0, j0 in allo.grid(N, N):
            x1[i0] = x1[i0] + A[i0, j0] * y1[j0]

        for i1, j1 in allo.grid(N, N):
            x2[i1] = x2[i1] + A[j1, i1] * y2[j1]

    s0 = allo.customize(kernel_mvt)
    orig = s0.build("vhls")

    s = allo.customize(kernel_mvt)
    s.compute_at("i0", "i1")
    s.unroll("j0")
    s.unroll("j1")
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_mvt()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "mvt", liveout_vars="v3,v4")
