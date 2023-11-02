import allo
import numpy as np
from allo.ir.types import float32, int32


def top_gesummv(size="mini", alpha=0.1, beta=0.1):
    if size == "mini" or size is None:
        N = 30
    elif size == "small":
        N = 90
    elif size == "medium":
        N = 250

    def kernel_gesummv(
        A: float32[N, N], B: float32[N, N], x: float32[N], y: float32[N]
    ):
        for i, j in allo.grid(N, N):
            y[i] += alpha * A[i, j] * x[j]

        for i, j in allo.grid(N, N):
            y[i] += beta * B[i, j] * x[j]

    s0 = allo.customize(kernel_gesummv)
    orig = s0.build("vhls")

    s = allo.customize(kernel_gesummv)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_gesummv()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "gesummv", liveout_vars="v3")
