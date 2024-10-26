# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int8, int32, bool, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np
import pytest

M, N, K = 4, 4, 4
Rt, Ct = 2, 2
# Rt, Ct = 4, 4
# M, N, K = 16, 16, 16
# Rt, Ct = 4, 4
# Rt, Ct = 16, 16
P0, P1 = Rt + 2, Ct + 2


@df.kernel(mapping=[P0, P1])
def gemm(A: int32[M, K], B: int32[K, N], flowtag: bool, C: int32[M, N]):

    # --------------------------------------------------------
    Rtimes: int32 = M // Rt if flowtag else K // Rt
    Ctimes: int32 = N // Ct
    Tlength: int32 = K if flowtag else M

    local_S: int32

    # --------------------------------------------------------
    # Interconnect
    i, j = df.get_pid()
    in_R: Stream[int32] = df.pipe(src=(i, j - 1), dst=(i, j))
    in_C: Stream[int32] = df.pipe(src=(i - 1, j), dst=(i, j))
    out_R: Stream[int32] = df.pipe(src=(i, j), dst=(i, j + 1))
    out_C: Stream[int32] = df.pipe(src=(i, j), dst=(i + 1, j))

    # --------------------------------------------------------
    # Computation
    for ri in range(Rtimes, name="row_loop"):
        for ci in range(Ctimes, name="column_loop"):
            # corner
            with allo.meta_if(i in {0, Rt + 1} and j in {0, Ct + 1}):
                pass

            # peripheral Load
            with allo.meta_elif(i == 0):
                for t in range(Tlength):
                    if flowtag:
                        out_C.put(B[t, ci * Ct + (j - 1)])
                    else:
                        out_C.put(0)

            with allo.meta_elif(j == 0):
                for t in range(Tlength):
                    out_R.put(
                        A[ri * Rt + (i - 1), t] if flowtag else A[t, ri * Rt + (i - 1)]
                    )

            # peripheral Drain
            with allo.meta_elif(i == Rt + 1 and j > 0):
                for t in range(Tlength):
                    if flowtag:
                        c_drain: int32 = in_C.get()
                    else:
                        C[t, ci * Ct + (j - 1)] = C[t, ci * Ct + (j - 1)] + in_C.get()

            with allo.meta_elif(j == Ct + 1 and i > 0):
                for t in range(Tlength):
                    r_drain: int32 = in_R.get()

            # main Compute
            with allo.meta_else():
                local_S = 0 if flowtag else B[ri * Rt + (i - 1), ci * Ct + (j - 1)]

                for t in range(Tlength):
                    # Flow IN
                    s: int32 = local_S  # omit peripheral pe
                    r: int32 = in_R.get()
                    c: int32 = in_C.get()
                    # Core MAC
                    act: int32 = r
                    weight: int32 = c if flowtag else s
                    psum: int32 = s if flowtag else c
                    acc: int32 = act * weight + psum
                    # FLOW OUT
                    local_S = acc if flowtag else s  # *
                    out_R.put(r)
                    out_C.put(c if flowtag else acc)

                if flowtag:
                    C[ri * Rt + (i - 1), ci * Ct + (j - 1)] = local_S
                else:
                    pass


def test_unified_systolic():

    A = np.random.randint(0, 10, (M, K)).astype(np.int32)
    B = np.random.randint(0, 10, (K, N)).astype(np.int32)

    A_flat = A.flatten()
    B_flat = B.flatten()

    C = np.zeros((M, N), dtype=np.int32)
    C_flat = C.flatten()
    flowtag: bool = True

    if hls.is_available("vitis_hls"):
        gemm(A_flat, B_flat, flowtag, C_flat)
        print(C_flat)
        C_truth = np.dot(A, B).flatten()
        print(C_truth)
        np.testing.assert_allclose(C_flat, C_truth, atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    pytest.main([__file__])
