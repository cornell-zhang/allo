# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unused-import

from .. import dsl
from ..ir.types import int8, int32, index, Int
from ..ir.utils import MockBuffer


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
    # Be careful, need to use high precision for accumulation
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
    for i, j in dsl.grid(Mt, Nt, name="PE"):
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
    for mi, ni in dsl.grid(M // Mt, N // Nt, name="outer_tile"):
        # reversed traversal, better for cascading systolic arrays with FIFOs
        # corresponds to the order of the previous `store_C_tile` output
        for ak, ai in dsl.grid(K, Mt, name="load_A_tile"):
            # reuse along the ni dimension
            if ni == 0:
                local_A[ai, ak] = A[mi * Mt + ai, ak]
        for bk, bj in dsl.grid(K, Nt, name="load_B_tile"):
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
        for sj, si in dsl.grid(Nt, Mt, name="store_C_tile"):
            C[mi * Mt + si, ni * Nt + sj] = local_C[si, sj]


def packed_systolic[
    TyA: Int,
    TyB: Int,
    TyC: Int,
    M: int32,
    K: int32,
    N: int32,
    Mt: int32,
    Nt: int32,
    P: int32,  # packing factor
](
    A: "Int(TyA.bits * P)[M, K // P]",
    B: "Int(TyB.bits * P)[K, N // P]",
    C: "Int(TyC.bits * P)[M, N // P]",
):
    local_A: TyA[Mt, K]
    local_B: TyB[K, Nt]
    local_C: TyC[Mt, Nt]

    # k needs not be tiled, since it is temporal dimension
    for mi, ni in dsl.grid(M // Mt, N // Nt, name="outer_tile"):
        # reversed traversal, better for cascading systolic arrays with FIFOs
        # corresponds to the order of the previous `store_C_tile` output
        for ak, ai in dsl.grid(K // P, Mt, name="load_A_tile"):
            # reuse along the ni dimension
            if ni == 0:
                a: Int(TyA.bits * P) = A[mi * Mt + ai, ak]
                for p in range(P):
                    local_A[ai, ak * P + p] = a[p * TyA.bits : (p + 1) * TyA.bits]
        for bk, bj in dsl.grid(K, Nt // P, name="load_B_tile"):
            # reuse along the mi dimension
            # since the inner access order is different from the outer one,
            # we cannot cache as a line buffer
            b: Int(TyB.bits * P) = B[bk, ni * Nt // P + bj]
            for p in range(P):
                local_B[bk, bj * P + p] = b[p * TyB.bits : (p + 1) * TyB.bits]
        systolic_tile[TyA, TyB, TyC, K, Mt, Nt](
            local_A,
            local_B,
            local_C,
        )
        # reversed traversal, better for cascading systolic arrays with FIFOs
        for sj, si in dsl.grid(Nt // P, Mt, name="store_C_tile"):
            c: Int(TyC.bits * P) = 0
            for p in range(P):
                # pylint: disable=unsupported-assignment-operation
                c[p * TyC.bits : (p + 1) * TyC.bits] = local_C[si, sj * P + p]
            C[mi * Mt + si, ni * Nt // P + sj] = c


def schedule_systolic(s):
    assert len(s.inst_list) in {8, 9}
    s.partition(s.local_C, dim=0)  # required, otherwise it will fail dataflow checking
    s.partition(s.local_A, dim=1)
    s.partition(s.local_B, dim=2)
    load_A_loop = s.get_loops(s.top_func_name)["outer_tile"]["ai"]
    s.pipeline(load_A_loop)
    load_B_loop = s.get_loops(s.top_func_name)["outer_tile"]["bj"]
    s.pipeline(load_B_loop)
    store_C_loop = s.get_loops(s.top_func_name)["outer_tile"]["si"]
    s.pipeline(store_C_loop)
    tile_loop = s.get_loops(s.top_func_name)["outer_tile"]["ni"]
    s.dataflow(tile_loop)
    pe = s.unfold("systolic_tile:PE", [0, 1])  # specify which are spatial loops
    s.to(MockBuffer("systolic_tile", "A_fifo"), pe, axis=1, depth=s.inst_list[-2] + 1)
    s.to(MockBuffer("systolic_tile", "B_fifo"), pe, axis=0, depth=s.inst_list[-1] + 1)
    return s
