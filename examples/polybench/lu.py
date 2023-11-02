import allo
import numpy as np
from allo.ir.types import float32, int32


def top_lu(size="tiny"):
    if size == "mini":
        N = 40
    elif size == "small":
        N = 120
    elif size == "medium":
        N = 400
    elif size is None or size == "tiny":
        N = 4

    def kernel_lu(A: float32[N, N]):
        for i in range(N):
            for j in range(i):
                for k in range(j):
                    A[i, j] -= A[i, k] * A[k, j]
                A[i, j] /= A[j, j]

            for j in range(i, N):
                for k in range(i):
                    A[i, j] -= A[i, k] * A[k, j]

    s0 = allo.customize(kernel_lu)
    orig = s0.build("vhls")

    s = allo.customize(kernel_lu)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_lu(size="tiny")
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "lu", liveout_vars="v0")
