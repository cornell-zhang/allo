# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, bool
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 4, 4, 4
Rt, Ct = 2, 2
# Rt, Ct = 4, 4
# M, N, K = 16, 16, 16
# Rt, Ct = 4, 4
# Rt, Ct = 16, 16
P0, P1 = Rt + 2, Ct + 2


@df.region()
def top():
    fifo_R = df.array(df.pipe(dtype=int32, shape=(), depth=4), shape=(P0, P1))
    fifo_C = df.array(df.pipe(dtype=int32, shape=(), depth=4), shape=(P0, P1))

    @df.kernel(mapping=[P0, P1])
    def gemm(A: int32[M, K], B: int32[K, N], flowtag: bool, C: int32[M, N]):

        # --------------------------------------------------------
        Rtimes: int32 = M // Rt if flowtag else K // Rt
        Ctimes: int32 = N // Ct
        Tlength: int32 = K if flowtag else M
        local_S: int32
        i, j = df.get_pid()

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
                            fifo_C[i + 1, j].put(B[t, ci * Ct + (j - 1)])
                        else:
                            fifo_C[i + 1, j].put(0)

                with allo.meta_elif(j == 0):
                    for t in range(Tlength):
                        fifo_R[i, j + 1].put(
                            A[ri * Rt + (i - 1), t]
                            if flowtag
                            else A[t, ri * Rt + (i - 1)]
                        )

                # peripheral Drain
                with allo.meta_elif(i == Rt + 1 and j > 0):
                    for t in range(Tlength):
                        if flowtag:
                            c_drain: int32 = fifo_C[i, j].get()
                        else:
                            C[t, ci * Ct + (j - 1)] = (
                                C[t, ci * Ct + (j - 1)] + fifo_C[i, j].get()
                            )

                with allo.meta_elif(j == Ct + 1 and i > 0):
                    for t in range(Tlength):
                        r_drain: int32 = fifo_R[i, j].get()

                # main Compute
                with allo.meta_else():
                    local_S = 0 if flowtag else B[ri * Rt + (i - 1), ci * Ct + (j - 1)]

                    for t in range(Tlength):
                        # Flow IN
                        s: int32 = local_S  # omit peripheral pe
                        r: int32 = fifo_R[i, j].get()
                        c: int32 = fifo_C[i, j].get()
                        # Core MAC
                        act: int32 = r
                        weight: int32 = c if flowtag else s
                        psum: int32 = s if flowtag else c
                        acc: int32 = act * weight + psum
                        # FLOW OUT
                        local_S = acc if flowtag else s  # *
                        fifo_R[i, j + 1].put(r)
                        fifo_C[i + 1, j].put(c if flowtag else acc)

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
    flowtag = True
    mod = df.build(top)

    if hls.is_available("vitis_hls"):
        mod(A_flat, B_flat, flowtag, C_flat)
        print(C_flat)
        C_truth = np.dot(A, B).flatten()
        print(C_truth)
        np.testing.assert_allclose(C_flat, C_truth, atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_unified_systolic()
