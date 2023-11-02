import allo
import numpy as np
from allo.ir.types import float32, int32


def top_syrk(size="mini", alpha=1.5, beta=1.2):
    if size == "mini" or size is None:
        M = 20
        N = 30
    elif size == "small":
        M = 60
        N = 80
    elif size == "medium":
        M = 200
        N = 240

    def kernel_syrk(A: float32[N, M], C: float32[N, N]):
        for i in range(N):
            for j in range(i + 1):
                C[i, j] *= beta
            for k in range(M):
                for j in range(i + 1):
                    C[i, j] += alpha * A[i, k] * A[j, k]

    s0 = allo.customize(kernel_syrk)
    orig = s0.build("vhls")

    s = allo.customize(kernel_syrk)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_syrk()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "syrk", liveout_vars="v1")
