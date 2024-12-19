# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, bool
from allo.ir.utils import MockBuffer
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 4, 4, 4
Rt, Ct = 2, 2
P0, P1 = Rt + 2, Ct + 2


@df.region()
def top():
    # interconnect
    fifo_R = df.array(df.pipe(dtype=int32, shape=(), depth=16), shape=(P0, P1 - 1))
    fifo_C = df.array(df.pipe(dtype=int32, shape=(), depth=16), shape=(P0 - 1, P1))
    inst_broad = df.array(df.pipe(dtype=bool, shape=(), depth=4), shape=(P1 - 1,))
    inst_chain = df.array(df.pipe(dtype=bool, shape=(), depth=4), shape=(P0 - 1, P1))

    @df.kernel(mapping=[P0, P1])
    def gemm(A: int32[M, K], B: int32[K, N], inst: bool, C: int32[M, N]):

        i, j = df.get_pid()

        # --------------------------------------------------------
        # Decode and Dispatch

        with allo.meta_if(i == 0 and j == 0):
            tag: bool = inst
            inst_broad[j].put(tag)
            inst_chain[i, j].put(tag)

        with allo.meta_else():
            with allo.meta_if(i == 0):
                flowtag: bool = inst_broad[j - 1].get()
            with allo.meta_else():
                flowtag: bool = inst_chain[i - 1, j].get()

            with allo.meta_if(i == 0 and j != P1 - 1):
                inst_broad[j].put(flowtag)
            with allo.meta_if(i != P0 - 1):
                inst_chain[i, j].put(flowtag)

        # --------------------------------------------------------
        # Computation

        with allo.meta_if(i in {0, Rt + 1} and j in {0, Ct + 1}):
            pass

        with allo.meta_else():
            # --------------------------------------------------------
            # Parameters
            Rtimes: int32 = M // Rt if flowtag else K // Rt
            Ctimes: int32 = N // Ct
            Tlength: int32 = K if flowtag else M
            Czero: int32 = 0

            for ri in range(Rtimes, name="row_loop"):
                for ci in range(Ctimes, name="column_loop"):

                    # peripheral Load
                    with allo.meta_if(i == 0):
                        for t in range(Tlength):
                            if flowtag:
                                fifo_C[i, j].put(B[t, ci * Ct + (j - 1)])
                            else:
                                fifo_C[i, j].put(Czero)

                    with allo.meta_elif(j == 0):
                        for t in range(Tlength):
                            fifo_R[i, j].put(
                                A[ri * Rt + (i - 1), t]
                                if flowtag
                                else A[t, ri * Rt + (i - 1)]
                            )

                    # peripheral Drain
                    with allo.meta_elif(i == Rt + 1 and j > 0):
                        for t in range(Tlength):
                            if flowtag:
                                c_drain: int32 = fifo_C[i - 1, j].get()
                            else:
                                C[t, ci * Ct + (j - 1)] = (
                                    C[t, ci * Ct + (j - 1)] + fifo_C[i - 1, j].get()
                                )

                    with allo.meta_elif(j == Ct + 1 and i > 0):
                        for t in range(Tlength):
                            r_drain: int32 = fifo_R[i, j - 1].get()

                    # main Compute
                    with allo.meta_else():
                        local_S: int32 = (
                            0 if flowtag else B[ri * Rt + (i - 1), ci * Ct + (j - 1)]
                        )

                        for t in range(Tlength):
                            # Flow IN
                            s: int32 = local_S  # omit peripheral pe
                            r: int32 = fifo_R[i, j - 1].get()
                            c: int32 = fifo_C[i - 1, j].get()
                            # Core MAC
                            acti: int32 = r
                            weight: int32 = c if flowtag else s
                            psum: int32 = s if flowtag else c
                            accu: int32 = acti * weight + psum
                            # FLOW OUT
                            local_S = accu if flowtag else s  # *
                            fifo_R[i, j].put(r)
                            fifo_C[i, j].put(c if flowtag else accu)

                        if flowtag:
                            C[ri * Rt + (i - 1), ci * Ct + (j - 1)] = local_S
                        else:
                            pass


def schedule_unified_systolic(s):
    s.partition(MockBuffer(s.top_func_name, "A"), dim=0)
    s.partition(MockBuffer(s.top_func_name, "B"), dim=0)
    s.partition(MockBuffer(s.top_func_name, "C"), dim=0)
    return s


def test_unified_systolic():

    A = np.random.randint(-8, 8, (M, K)).astype(np.int32)
    B = np.random.randint(-8, 8, (K, N)).astype(np.int32)

    C = np.zeros((M, N), dtype=np.int32)

    if hls.is_available("vitis_hls"):

        s = df.customize(top)
        schedule_unified_systolic(s)

        # csim test
        print(" Csim Test ".center(60, "*"))
        mod = s.build(target="vitis_hls", mode="csim", project="top.prj")
        C_truth = np.dot(A, B)
        print(C_truth)

        flowtag1: bool = False
        mod(A, B, flowtag1, C)
        print(C)
        np.testing.assert_allclose(C, C_truth, atol=1e-5)
        print("Csim: Weight-stationary Mode Passed!")

        flowtag2: bool = True
        C = np.zeros((M, N), dtype=np.int32)
        mod(A, B, flowtag2, C)
        print(C)
        np.testing.assert_allclose(C, C_truth, atol=1e-5)
        print("Csim: Output-stationary Mode Passed!")

        # csyn test
        print(" Csyn Test ".center(60, "*"))
        mod_csyn = s.build(target="vitis_hls", mode="csyn", project="df-uni-csyn.prj")
        mod_csyn()
        print("Design: C-Synthesizable!")

        # hw_emu test
        print(" Hw_emu Test ".center(60, "*"))
        mod_hwemu = s.build(
            target="vitis_hls", mode="hw_emu", project="df-uni-hwemu.prj"
        )
        C = np.zeros((M, N), dtype=np.int32)
        mod_hwemu(A, B, flowtag1, C)
        print(C)
        np.testing.assert_allclose(C, C_truth, atol=1e-5)
        print("Hw_emu: Weight-stationary Mode Passed!")

        C = np.zeros((M, N), dtype=np.int32)
        mod_hwemu(A, B, flowtag2, C)
        print(C)
        np.testing.assert_allclose(C, C_truth, atol=1e-5)
        print("Hw_emu: Output-stationary Mode Passed!")


if __name__ == "__main__":
    test_unified_systolic()
