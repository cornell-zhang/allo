import allo
import numpy as np
from allo.ir.types import float32, int32


def top_floyd_warshall(size="tiny"):
    if size == "mini":
        N = 60
    elif size == "small":
        N = 180
    elif size == "medium":
        N = 500
    elif size is None or size == "tiny":
        N = 2

    def kernel_floyd_warshall(path: float32[N, N]):
        for k, i, j in allo.grid(N, N, N):
            path_: float32 = path[i, k] + path[k, j]
            if path[i, j] >= path_:
                path[i, j] = path_

    s0 = allo.customize(kernel_floyd_warshall)
    orig = s0.build("vhls")

    s = allo.customize(kernel_floyd_warshall)
    if N > 2:
        s.split("k", factor=2)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_floyd_warshall()
    # print(orig)
    # print(opt)
    from cedar.verify import verify_pair

    verify_pair(
        orig,
        opt,
        "floyd_warshall",
        options="--symbolic-conditionals",
        liveout_vars="v0",
    )
