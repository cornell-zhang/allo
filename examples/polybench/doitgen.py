import allo
import numpy as np
from allo.ir.types import int32


def top_doitgen(size="mini"):
    if size == "mini" or size is None:
        Q = 8
        R = 10
        P = 12
        S = P
    elif size == "small":
        Q = 20
        R = 25
        P = 30
        S = P
    elif size == "medium":
        Q = 20
        R = 25
        P = 30
        S = P

    def kernel_doitgen(A: int32[R, Q, S], x: int32[P, S]):
        sum_: int32[P] = 0
        for r, q in allo.grid(R, Q):
            for p in allo.grid(P):
                sum_[p] = 0
                for s in allo.grid(P):
                    sum_[p] = sum_[p] + A[r, q, s] * x[s, p]
            for p1 in allo.grid(P):
                A[r, q, p1] = sum_[p1]

    s0 = allo.customize(kernel_doitgen)
    orig = s0.build("vhls")

    s = allo.customize(kernel_doitgen)
    s.split("p", factor=2)
    s.reorder("q", "p.inner", "p.outer")
    s.pipeline("p.inner")
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_doitgen()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "doitgen", liveout_vars="v1")
