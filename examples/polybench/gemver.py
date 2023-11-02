import allo
import numpy as np
from allo.ir.types import float32, int32


def top_gemver(size="mini", alpha=0.1, beta=0.1):
    if size == "mini" or size is None:
        N = 40
    elif size == "small":
        N = 120
    elif size == "medium":
        N = 400

    def kernel_gemver(
        A: float32[N, N],
        u1: float32[N],
        u2: float32[N],
        v1: float32[N],
        v2: float32[N],
        x: float32[N],
        y: float32[N],
        w: float32[N],
        z: float32[N],
    ):
        for i, j in allo.grid(N, N):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

        for i, j in allo.grid(N, N):
            x[i] = x[i] + beta * A[j, i] * y[j]

        for i in allo.grid(N):
            x[i] = x[i] + z[i]

        for i, j in allo.grid(N, N):
            w[i] = w[i] + alpha * A[i, j] * x[j]

    s0 = allo.customize(kernel_gemver)
    orig = s0.build("vhls")

    s = allo.customize(kernel_gemver)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_gemver()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "gemver", liveout_vars="v0,v5,v7")
