# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32, bool, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


@df.region()
def unified_gemm_simple():
    # interconnect
    fifo_R: Stream[int32, 16][P0, P1 - 1]
    fifo_C: Stream[int32, 16][P0 - 1, P1]
    inst_broad: Stream[bool, 4][P1 - 1]
    inst_chain: Stream[bool, 4][P0 - 1, P1]

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

        with allo.meta_if(i in {0, U + 1} and j in {0, U + 1}):
            pass

        with allo.meta_else():
            # --------------------------------------------------------
            # Parameters
            Tlength: int32 = K if flowtag else M
            Czero: int32 = 0

            # peripheral Load
            with allo.meta_if(i == 0):
                for t in range(Tlength):
                    if flowtag:
                        fifo_C[i, j].put(B[t, j - 1])
                    else:
                        fifo_C[i, j].put(Czero)

            with allo.meta_elif(j == 0):
                for t in range(Tlength):
                    fifo_R[i, j].put(A[i - 1, t] if flowtag else A[t, i - 1])

            # peripheral Drain
            with allo.meta_elif(i == U + 1 and j > 0):
                for t in range(Tlength):
                    if flowtag:
                        c_drain: int32 = fifo_C[i - 1, j].get()
                    else:
                        C[t, j - 1] = fifo_C[i - 1, j].get()

            with allo.meta_elif(j == U + 1 and i > 0):
                for t in range(Tlength):
                    r_drain: int32 = fifo_R[i, j - 1].get()

            # main Compute
            with allo.meta_else():
                local_S: int32 = 0 if flowtag else B[i - 1, j - 1]

                for t in range(Tlength):
                    # Flow In
                    s: int32 = local_S  # omit peripheral pe
                    r: int32 = fifo_R[i, j - 1].get()
                    c: int32 = fifo_C[i - 1, j].get()
                    # Core MAC
                    acti: int32 = r
                    weight: int32 = c if flowtag else s
                    psum: int32 = s if flowtag else c
                    accu: int32 = acti * weight + psum
                    # Flow Out
                    local_S = accu if flowtag else s  # *
                    fifo_R[i, j].put(r)
                    fifo_C[i, j].put(c if flowtag else accu)

                if flowtag:
                    C[i - 1, j - 1] = local_S


@df.region()
def unified_gemm_daisy_chain():
    L2_R: Stream[UInt(U * 16), 4][P0 - 1]
    L2_C: Stream[UInt(N * 16), 4][P1 - 1]

    L1_S: Stream[UInt(U * 16), 4][U + 1, N]
    L2_S_in: Stream[UInt(U * 16), 4][N]
    L2_S_out: Stream[UInt(U * 16), 4][N]

    fifo_R: Stream[int16, 4][U, N]
    fifo_C: Stream[int16, 4][U + 1, N]  # Additional one for partial sum in WS

    inst_broad: Stream[bool, 4][P1 - 1]
    inst_chain: Stream[bool, 4][P0 - 1, P1]

    @df.kernel(mapping=[P0, P1])
    def gemm(A: int16[M, K], B: int16[K, N], inst: bool, C: int16[M, N]):

        # --------------------------------------------------------
        # Parameters
        i, j = df.get_pid()
        Rtimes: int16 = U
        Ctimes: int16 = N
        Tlength: int16 = U
        Czero: int16 = 0

        # --------------------------------------------------------
        # Instruction Decode and Dispatch

        with allo.meta_if(i == 0 and j == 0):
            flowtag: bool = inst
            inst_broad[j].put(flowtag)
            inst_chain[i, j].put(flowtag)

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

        # corner kernels
        with allo.meta_if(i == 0 and j == 0):
            if not flowtag:
                # pack weight
                for n in range(N):
                    packed_S_in: UInt(U * 16) = 0
                    for k in range(U):
                        packed_S_in[k * 16 : (k + 1) * 16] = B[k, n]
                    L2_S_in[0].put(packed_S_in)

            for u in range(U):
                # pack data Row
                packed_R: UInt(U * 16) = 0
                if flowtag:
                    for m in range(U):
                        packed_R[m * 16 : (m + 1) * 16] = A[m, u]
                else:
                    for k in range(U):
                        packed_R[k * 16 : (k + 1) * 16] = A[u, k]
                L2_R[1].put(packed_R)
                # pack data Column
                packed_C: UInt(N * 16) = 0
                if flowtag:
                    for n in range(N):
                        packed_C[n * 16 : (n + 1) * 16] = B[u, n]
                else:
                    for n in range(N):
                        packed_C[n * 16 : (n + 1) * 16] = Czero
                L2_C[1].put(packed_C)

        with allo.meta_elif(i == P0 - 1 and j == P1 - 1):
            for n in range(N):
                packed_S_out = L2_S_out[N - 1].get()
                for m in range(M):
                    C[m, n] = packed_S_out[m * 16 : (m + 1) * 16]

        with allo.meta_elif(i in {0, P0 - 1} and j in {0, P1 - 1}):
            pass

        # peripheral kernels
        with allo.meta_elif(j == 0):
            # i > 0, the first column
            for u in range(U):
                r = L2_R[i].get()
                # unpack data
                fifo_R[i - 1, 0].put(r[16 * (i - 1) : 16 * i])
                with allo.meta_if(i < U):
                    L2_R[i + 1].put(r)

        with allo.meta_elif(i == 0):
            # j > 0, the first row
            if not flowtag:
                L1_S[0, j - 1].put(L2_S_in[j - 1].get())
                with allo.meta_if(j != P1 - 2):
                    for ind in range(N - j):
                        L2_S_in[j].put(L2_S_in[j - 1].get())

            for u in range(U):
                c = L2_C[j].get()
                fifo_C[0, j - 1].put(c[16 * (j - 1) : 16 * j])
                with allo.meta_if(j < N):
                    L2_C[j + 1].put(c)

        with allo.meta_elif(i == P0 - 1):
            if flowtag:  # OS
                c_C = L1_S[i - 1, N - j].get()
                L2_S_out[j - 1].put(c_C)
                with allo.meta_if(j != 1):
                    for ind in range(j - 1):
                        L2_S_out[j - 1].put(L2_S_out[j - 2].get())

            else:  # WS
                with allo.meta_if(j != 1):
                    for ind in range(j - 1):
                        L2_S_out[j - 1].put(L2_S_out[j - 2].get())

                c_C: UInt(U * 16) = 0
                for m in range(U):
                    c_C[m * 16 : (m + 1) * 16] = fifo_C[U, j - 1].get()
                L2_S_out[j - 1].put(c_C)

        with allo.meta_elif(j == P1 - 1):
            pass

        # main body
        with allo.meta_else():
            local_s: int16 = 0

            # Stationary Cache-In
            if not flowtag:
                packed_tmp: UInt(U * 16) = L1_S[i - 1, j - 1].get()
                local_s = packed_tmp[16 * (i - 1) : 16 * i]
                with allo.meta_if(i < U):
                    L1_S[i, j - 1].put(packed_tmp)

            for u in range(U):
                # Flow In
                r: int16 = fifo_R[i - 1, j - 1].get()
                c: int16 = fifo_C[i - 1, j - 1].get()
                # Core MAC
                acti: int16 = r
                weight: int16 = c if flowtag else local_s
                psum: int16 = local_s if flowtag else c
                accu: int16 = acti * weight + psum
                if flowtag:
                    local_s = accu
                # Flow Out
                with allo.meta_if(j < N):
                    fifo_R[i - 1, j].put(r)
                with allo.meta_if(i < U):
                    fifo_C[i, j - 1].put(c if flowtag else accu)
                with allo.meta_if(i == U):
                    if not flowtag:
                        fifo_C[i, j - 1].put(accu)

            # Stationary Cache-Out
            if flowtag:
                with allo.meta_if(i == 1):
                    packed_tmp: UInt(U * 16) = 0
                with allo.meta_else():
                    packed_tmp: UInt(U * 16) = L1_S[i - 1, j - 1].get()

                packed_c: UInt(U * 16) = 0
                for m in range(U):
                    if m == i - 1:
                        packed_c[m * 16 : (m + 1) * 16] = local_s
                    else:
                        packed_c[m * 16 : (m + 1) * 16] = packed_tmp[
                            m * 16 : (m + 1) * 16
                        ]
                L1_S[i, j - 1].put(packed_c)


@df.region()
def unified_gemm_tiling():
    # interconnect
    fifo_R: Stream[int32, 16][P0, P1 - 1]
    fifo_C: Stream[int32, 16][P0 - 1, P1]

    inst_broad: Stream[bool, 4][P1 - 1]
    inst_chain: Stream[bool, 4][P0 - 1, P1]

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
                            # Flow In
                            s: int32 = local_S  # omit peripheral pe
                            r: int32 = fifo_R[i, j - 1].get()
                            c: int32 = fifo_C[i - 1, j].get()
                            # Core MAC
                            acti: int32 = r
                            weight: int32 = c if flowtag else s
                            psum: int32 = s if flowtag else c
                            accu: int32 = acti * weight + psum
                            # Flow Out
                            local_S = accu if flowtag else s  # *
                            fifo_R[i, j].put(r)
                            fifo_C[i, j].put(c if flowtag else accu)

                        if flowtag:
                            C[ri * Rt + (i - 1), ci * Ct + (j - 1)] = local_S


def schedule_unified_systolic(s):
    s.partition(f"{s.top_func_name}:A", dim=0)
    s.partition(f"{s.top_func_name}:B", dim=0)
    s.partition(f"{s.top_func_name}:C", dim=0)
    return s


U = 4  # Require for same size in two dimension if not tiling
M, N, K = U, 4, U
P0, P1 = U + 2, U + 2


def test_unified_simple():

    A = np.random.randint(-8, 8, (M, K)).astype(np.int32)
    B = np.random.randint(-8, 8, (K, N)).astype(np.int32)

    C = np.zeros((M, N), dtype=np.int32)

    if hls.is_available("vitis_hls"):

        s = df.customize(unified_gemm_simple)
        schedule_unified_systolic(s)

        # csim test
        print(" Csim Test ".center(60, "*"))
        mod = s.build(target="vitis_hls", mode="csim", project="df-uni-simple-csim.prj")
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

        # hw_emu test
        print(" Hw_emu Test ".center(60, "*"))
        mod_hwemu = s.build(
            target="vitis_hls", mode="hw_emu", project="df-uni-simple-hwemu.prj"
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


def test_unified_daisy_chain():
    A = np.random.randint(0, 8, (M, K), dtype=np.int16)
    B = np.random.randint(0, 8, (K, N), dtype=np.int16)
    C = np.zeros((M, N), dtype=np.int16)

    if hls.is_available("vitis_hls"):
        # csim test
        print(" Csim Test ".center(60, "*"))
        mod = df.build(
            unified_gemm_daisy_chain,
            target="vitis_hls",
            mode="csim",
            project="df-uni-daisy-csim.prj",
        )
        C_truth = np.dot(A, B)
        print(C_truth)

        flowtag1: bool = False
        mod(A, B, flowtag1, C)
        print(C)
        np.testing.assert_allclose(C, C_truth, atol=1e-5)
        print("Csim: Weight-stationary Mode Passed!")

        flowtag2: bool = True
        C = np.zeros((M, N), dtype=np.int16)
        mod(A, B, flowtag2, C)
        print(C)
        np.testing.assert_allclose(C, C_truth, atol=1e-5)
        print("Csim: Output-stationary Mode Passed!")

        # hw_emu test
        print(" Hw_emu Test ".center(60, "*"))
        mod_hwemu = df.build(
            unified_gemm_daisy_chain,
            target="vitis_hls",
            mode="hw_emu",
            project="df-uni-daisy-hwemu.prj",
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


M, N, K = 4, 4, 4
Rt, Ct = 2, 2
P0, P1 = Rt + 2, Ct + 2


def test_unified_tiling():

    A = np.random.randint(-8, 8, (M, K)).astype(np.int32)
    B = np.random.randint(-8, 8, (K, N)).astype(np.int32)

    C = np.zeros((M, N), dtype=np.int32)

    if hls.is_available("vitis_hls"):

        s = df.customize(unified_gemm_tiling)
        schedule_unified_systolic(s)

        # csim test
        print(" Csim Test ".center(60, "*"))
        mod = s.build(target="vitis_hls", mode="csim", project="df-uni-tiling-csim.prj")
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

        # hw_emu test
        print(" Hw_emu Test ".center(60, "*"))
        mod_hwemu = s.build(
            target="vitis_hls", mode="hw_emu", project="df-uni-tiling-hwemu.prj"
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
    U = 4  # Require for same size in two dimension if not tiling
    M, N, K = U, 4, U
    P0, P1 = U + 2, U + 2
    test_unified_simple()
    test_unified_daisy_chain()

    M, N, K = 4, 4, 4
    Rt, Ct = 2, 2
    P0, P1 = Rt + 2, Ct + 2
    test_unified_tiling()
