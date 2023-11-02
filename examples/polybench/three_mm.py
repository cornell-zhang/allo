import allo
import numpy as np
from allo.ir.types import int32


def top_3mm(size="mini"):
    if size == "mini" or size is None:
        P = 16  ## P = NI
        R = 18  ## R = NJ
        Q = 20  ## Q = NK
        T = 22  ## T = NL
        S = 24  ## S = NM
    elif size == "small":
        P = 40
        R = 50
        Q = 60
        T = 70
        S = 80
    elif size == "medium":
        P = 180
        R = 190
        Q = 200
        T = 210
        S = 220

    def kernel_3mm(
        A: int32[P, Q], B: int32[Q, R], C: int32[R, S], D: int32[S, T]
    ) -> int32[P, T]:
        out_AB: int32[P, R] = 0
        for i0, j0, k0 in allo.grid(P, R, Q):
            out_AB[i0, j0] += A[i0, k0] * B[k0, j0]
        out_CD: int32[R, T] = 0
        for i1, j1, k1 in allo.grid(R, T, S):
            out_CD[i1, j1] += C[i1, k1] * D[k1, j1]
        output: int32[P, T] = 0
        for i2, j2, k2 in allo.grid(P, T, R):
            output[i2, j2] += out_AB[i2, k2] * out_CD[k2, j2]
        return output

    s0 = allo.customize(kernel_3mm)
    orig = s0.build("vhls")

    s = allo.customize(kernel_3mm)
    s.reorder("j1", "i1")
    opt = s.build("vhls")
    return orig, opt


if __name__ == "__main__":
    orig, opt = top_3mm()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "3mm", liveout_vars="v4")
