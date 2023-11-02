import allo
import numpy as np
from allo.ir.types import float32, int32


def top_symm(size="mini", alpha=1.5, beta=1.2):
    if size == "mini" or size is None:
        M = 20
        N = 30
    elif size == "small":
        M = 60
        N = 80
    elif size == "medium":
        M = 200
        N = 240

    def kernel_symm(A: float32[M, M], B: float32[M, N], C: float32[M, N]):
        for i in range(M):
            for j in range(N):
                sum_: float32 = 0.0
                for k in range(i):
                    C[k, j] = C[k, j] + alpha * B[i, j] * A[i, k]
                    sum_ += B[k, j] * A[i, k]
                C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * sum_

    s0 = allo.customize(kernel_symm)
    orig = s0.build("vhls")

    s = allo.customize(kernel_symm)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_symm()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "symm", liveout_vars="v2")
