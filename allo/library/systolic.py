# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unused-import

import allo
from allo.ir.types import int32, index


def PE_kernel[
    TyA, TyB, TyC, K: int32, Mt: int32, Nt: int32
](
    A_in: "TyA[K]",
    B_in: "TyB[K]",
    A_out: "TyA[K]",
    B_out: "TyB[K]",
    C: "TyC[Mt, Nt]",
    i: index,
    j: index,
):
    v: TyC = 0
    for k in range(K):
        a: TyA = A_in[k]
        b: TyB = B_in[k]
        v += a * b
        A_out[k] = a
        B_out[k] = b
    C[i, j] = v


def systolic_tile[
    TyA, TyB, TyC, K: int32, Mt: int32, Nt: int32
](A: "TyA[Mt, K]", B: "TyB[K, Nt]", C: "TyC[Mt, Nt]"):
    A_fifo: TyA[Mt, Nt + 1, K]
    B_fifo: TyB[Nt, Mt + 1, K]
    A_drain: TyA[Mt]
    B_drain: TyB[Nt]

    for k in range(K, name="data_load"):
        # Can be fully unrolled inside this loop,
        # once A and B are correctly partitioned
        for m in range(Mt):
            A_fifo[m, 0, k] = A[m, k]
        for n in range(Nt):
            B_fifo[n, 0, k] = B[k, n]
    for i, j in allo.grid(Mt, Nt, name="PE"):
        PE_kernel[TyA, TyB, TyC, K, Mt, Nt](
            A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j
        )
    for k in range(K, name="data_drain"):
        for m in range(Mt):
            A_drain[m] = A_fifo[m, Nt, k]
        for n in range(Nt):
            B_drain[n] = B_fifo[n, Mt, k]


def systolic[
    TyA, TyB, TyC, M: int32, K: int32, N: int32, Mt: int32, Nt: int32
](A: "TyA[M, K]", B: "TyB[K, N]", C: "TyC[M, N]"):
    local_A: TyA[Mt, K]
    local_B: TyB[K, Nt]
    local_C: TyC[Mt, Nt]

    # k needs not be tiled, since it is temporal dimension
    for mi, ni in allo.grid(M // Mt, N // Nt, name="outer_tile"):
        # reversed traversal, better for cascading systolic arrays with FIFOs
        # corresponds to the order of the previous `store_C_tile` output
        for ak, ai in allo.grid(K, Mt, name="load_A_tile"):
            # reuse along the ni dimension
            if ni == 0:
                local_A[ai, ak] = A[mi * Mt + ai, ak]
        for bk, bj in allo.grid(K, Nt, name="load_B_tile"):
            # reuse along the mi dimension
            # since the inner access order is different from the outer one,
            # we cannot cache as a line buffer
            local_B[bk, bj] = B[bk, ni * Nt + bj]
        systolic_tile[TyA, TyB, TyC, K, Mt, Nt](
            local_A,
            local_B,
            local_C,
        )
        # reversed traversal, better for cascading systolic arrays with FIFOs
        for sj, si in allo.grid(Nt, Mt, name="store_C_tile"):
            C[mi * Mt + si, ni * Nt + sj] = local_C[si, sj]
