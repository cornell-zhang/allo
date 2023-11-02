import allo
import numpy as np
from allo.ir.types import float32, int32


def top_trmm(size="mini", alpha=1.5):
    if size == "mini" or size is None:
        M = 20
        N = 30
    elif size == "small":
        M = 60
        N = 80
    elif size == "medium":
        M = 200
        N = 240

    def kernel_trmm(A: float32[M, M], B: float32[M, N]):
        for i in range(M):
            for j in range(N):
                for k in range(i + 1, M):
                    B[i, j] += A[k, i] * B[k, j]
                B[i, j] *= alpha

    s0 = allo.customize(kernel_trmm)
    orig = s0.build("vhls")

    s = allo.customize(kernel_trmm)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_trmm()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "trmm", liveout_vars="v1")
