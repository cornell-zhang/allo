import allo
import numpy as np
from allo.ir.types import float32, int32


def top_jacobi_1d(size="mini"):
    if size == "mini" or size is None:
        TSTEPS = 20
        N = 30
    elif size == "small":
        TSTEPS = 40
        N = 120
    elif size == "medium":
        TSTEPS = 100
        N = 400

    def kernel_jacobi_1d(A: float32[N], B: float32[N]):
        for m in range(TSTEPS):
            for i0 in range(1, N - 1):
                B[i0] = 0.33333 * (A[i0 - 1] + A[i0] + A[i0 + 1])

            for i1 in range(1, N - 1):
                A[i1] = 0.33333 * (B[i1 - 1] + B[i1] + B[i1 + 1])

    s0 = allo.customize(kernel_jacobi_1d)
    orig = s0.build("vhls")

    s = allo.customize(kernel_jacobi_1d)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_jacobi_1d()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "jacobi_1d", liveout_vars="v0,v1")
