import allo
import numpy as np
from allo.ir.types import float32, int32


def top_seidel_2d(size="tiny"):
    if size == "mini":
        TSTEPS = 20
        N = 40
    elif size == "small":
        TSTEPS = 40
        N = 120
    elif size == "medium":
        TSTEPS = 100
        N = 400
    elif size is None or size == "tiny":
        TSTEPS = 2
        N = 4

    def kernel_seidel_2d(A: float32[N, N]):
        for t in range(TSTEPS):
            for i in range(1, N - 2):
                for j in range(1, N - 2):
                    A[i, j] = (
                        A[i - 1, j - 1]
                        + A[i - 1, j]
                        + A[i - 1, j + 1]
                        + A[i, j - 1]
                        + A[i, j]
                        + A[i, j + 1]
                        + A[i + 1, j - 1]
                        + A[i + 1, j]
                        + A[i + 1, j + 1]
                    ) / 9

    s0 = allo.customize(kernel_seidel_2d)
    orig = s0.build("vhls")

    s = allo.customize(kernel_seidel_2d)

    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_seidel_2d("tiny")
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "seidel_2d", liveout_vars="v0")
