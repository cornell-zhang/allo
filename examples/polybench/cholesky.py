import allo
import numpy as np
from allo.ir.types import int32


def top_cholesky(size="tiny"):
    if size == "mini":
        N = 40
    elif size == "small":
        N = 120
    elif size == "medium":
        N = 400
    elif size is None or size == "tiny":
        N = 4

    def kernel_cholesky(A: int32[N, N]):
        for i in range(N):
            # Case: j < i
            for j in range(i):
                for k in range(j):
                    A[i, j] = A[i, j] - A[i, k] * A[j, k]
                A[i, j] = A[i, j] / A[j, j]
            # Case: i == j
            for k in range(i):
                A[i, i] = A[i, i] - A[i, k] * A[i, k]
            A[i, i] = allo.sqrt(A[i, i] * 1.0)

    s0 = allo.customize(kernel_cholesky)
    orig = s0.build("vhls")

    s = allo.customize(kernel_cholesky)
    s.split("i", factor=2)
    # s.reorder("j", "i.inner", "i.outer")
    s.pipeline("i.inner")
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_cholesky()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "cholesky", liveout_vars="v0")
