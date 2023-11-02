import allo
import numpy as np
from allo.ir.types import float32, int32


def top_heat_3d(size="tiny"):
    if size == "mini":
        TSTEPS = 20
        N = 10
    elif size == "small":
        TSTEPS = 40
        N = 20
    elif size == "medium":
        TSTEPS = 100
        N = 40
    elif size is None or size == "tiny":
        TSTEPS = 4
        N = 6

    def kernel_heat_3d(A: float32[N, N, N], B: float32[N, N, N]):
        const0: float32 = 0.125
        const1: float32 = 2.0

        for m in range(TSTEPS):
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    for k in range(1, N - 1):
                        B[i, j, k] = (
                            const0
                            * (A[i + 1, j, k] - const1 * A[i, j, k] + A[i - 1, j, k])
                            + const0
                            * (A[i, j + 1, k] - const1 * A[i, j, k] + A[i, j - 1, k])
                            + const0
                            * (A[i, j, k + 1] - const1 * A[i, j, k] + A[i, j, k - 1])
                            + A[i, j, k]
                        )

                        A[i, j, k] = (
                            const0
                            * (B[i + 1, j, k] - const1 * B[i, j, k] + B[i - 1, j, k])
                            + const0
                            * (B[i, j + 1, k] - const1 * B[i, j, k] + B[i, j - 1, k])
                            + const0
                            * (B[i, j, k + 1] - const1 * B[i, j, k] + B[i, j, k - 1])
                            + B[i, j, k]
                        )

    s0 = allo.customize(kernel_heat_3d)
    orig = s0.build("vhls")

    s = allo.customize(kernel_heat_3d)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_heat_3d(size="tiny")
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "heat_3d", liveout_vars="v0,v1")
