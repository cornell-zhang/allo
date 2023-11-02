import allo
import numpy as np
from allo.ir.types import float32, int32


def top_trisolv(size="tiny"):
    if size == "mini":
        N = 40
    elif size == "small":
        N = 120
    elif size == "medium":
        N = 400
    elif size is None or size == "tiny":
        N = 4

    def kernel_trisolv(L: float32[N, N], b: float32[N], x: float32[N]):
        for i in range(N):
            x[i] = b[i]
            for j in range(i):
                x[i] -= L[i, j] * x[j]
            x[i] /= L[i, i]

    s0 = allo.customize(kernel_trisolv)
    orig = s0.build("vhls")

    s = allo.customize(kernel_trisolv)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_trisolv("tiny")
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "trisolv", liveout_vars="v2")
