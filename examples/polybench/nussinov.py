import allo
import numpy as np
from allo.ir.types import float32, int32


def top_nussinov(size="mini"):
    if size == "mini" or size is None:
        N = 60
    elif size == "small":
        N = 180
    elif size == "medium":
        N = 500

    def kernel_nussinov(seq: float32[N], table: float32[N, N]):
        neg_one: int32 = -1
        for i in range(N - 1, neg_one, neg_one):
            for j in range(i + 1, N):
                if j - 1 >= 0:
                    if table[i, j] < table[i, j - 1]:
                        table[i, j] = table[i, j - 1]

                if i + 1 < N:
                    if table[i, j] < table[i + 1, j]:
                        table[i, j] = table[i + 1, j]

                if j - 1 >= 0 and i + 1 < N:
                    if i < j - 1:
                        w: float32 = seq[i] + seq[j]

                        match: float32 = 0.0
                        if w == 3:
                            match = 1.0

                        s2: float32 = 0.0
                        s2 = table[i + 1, j - 1] + match

                        if table[i, j] < s2:
                            table[i, j] = s2
                    else:
                        if table[i, j] < table[i + 1, j - 1]:
                            table[i, j] = table[i + 1, j - 1]

                for k in range(i + 1, j):
                    s3: float32 = table[i, k] + table[k + 1, j]
                    if table[i, j] < s3:
                        table[i, j] = s3

    s0 = allo.customize(kernel_nussinov)
    orig = s0.build("vhls")

    s = allo.customize(kernel_nussinov)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_nussinov()
    from cedar.verify import verify_pair

    verify_pair(
        orig, opt, "nussinov", options="--symbolic-conditionals", liveout_vars="v1"
    )
