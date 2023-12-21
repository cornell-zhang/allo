# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unused-import, unsupported-assignment-operation

from .. import dsl
from ..ir.types import int8, int16, int32, index, Int, UInt
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
    A: "Int(TyA.bits * P)[M // P, K]",
    B: "Int(TyB.bits * P)[K, N // P]",
    C: "Int(TyC.bits * P)[M // P, N]",
):
    local_A: TyA[Mt, K]
    local_B: TyB[K, Nt]
    local_C: TyC[Mt, Nt]

    # k needs not be tiled, since it is temporal dimension
    for mi, ni in dsl.grid(M // Mt, N // Nt, name="outer_tile"):
        # reversed traversal, better for cascading systolic arrays with FIFOs
        # corresponds to the order of the previous `store_C_tile` output
        for ak, ai in dsl.grid(K, Mt // P, name="load_A_tile"):
            # reuse along the ni dimension
            if ni == 0:
                a: Int(TyA.bits * P) = A[mi * Mt // P + ai, ak]
                for p in range(P):
                    local_A[ai * P + p, ak] = a[p * TyA.bits : (p + 1) * TyA.bits]
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
        for sj, si in dsl.grid(Nt, Mt // P, name="store_C_tile"):
            c: Int(TyC.bits * P) = 0
            for p in range(P):
                # pylint: disable=unsupported-assignment-operation
                c[p * TyC.bits : (p + 1) * TyC.bits] = local_C[si * P + p, sj]
            C[mi * Mt // P + si, ni * Nt + sj] = c


def PE_kernel_packed_int8xint8[
    K: int32, Mt: int32, Nt: int32
](
    A_in: "int8[K]",  # not bit-packed
    B_in: "int16[K]",  # bit-packed, each element is 8 bits
    A_out: "int8[K]",
    B_out: "int16[K]",
    C: "int32[Mt, Nt // 2]",  # bit-packed, each element is 16 bits
    i: index,
    j: index,
):
    v: int32 = 0
    for k in range(K):
        a: int8 = A_in[k]
        b_packed: int16 = B_in[k]
        b0: int8 = b_packed[0:8]
        b1: int8 = b_packed[8:16]
        s0: UInt(1) = a[7] ^ b0[7]
        s1: UInt(1) = a[7] ^ b1[7]
        au: UInt(8) = dsl.abs(a)
        b0u: UInt(8) = dsl.abs(b0)
        b1u: UInt(8) = dsl.abs(b1)
        # DSP48E1: 18x27 multiplier -> 45-bit result
        op0: UInt(18) = 0
        op1: UInt(27) = 0
        op0[0:8] = au
        op1[0:8] = b0u
        op1[17:25] = b1u
        res: UInt(45) = op0 * op1
        res0u: UInt(16) = res[0:16]
        res1u: UInt(16) = res[17:33]
        res0: int16 = -res0u if s0 else res0u
        res1: int16 = -res1u if s1 else res1u
        v[0:16] += res0
        v[16:32] += res1
        A_out[k] = a
        B_out[k] = b_packed
    C[i, j] = v


def systolic_tile_packed_int8xint8[
    K: int32, Mt: int32, Nt: int32
](A: "int8[Mt, K]", B: "int16[K, Nt // 2]", C: "int32[Mt, Nt // 2]"):
    A_fifo: int8[Mt, Nt // 2 + 1, K]
    B_fifo: int16[Nt // 2, Mt + 1, K]
    A_drain: int8[Mt]
    B_drain: int16[Nt // 2]

    for k in range(K, name="data_load"):
        # Can be fully unrolled inside this loop,
        # once A and B are correctly partitioned
        for m in range(Mt):
            A_fifo[m, 0, k] = A[m, k]
        for n in range(Nt // 2):
            B_fifo[n, 0, k] = B[k, n]
    for i, j in dsl.grid(Mt, Nt // 2, name="PE"):
        PE_kernel_packed_int8xint8[K, Mt, Nt](
            A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j
        )
    for k in range(K, name="data_drain"):
        for m in range(Mt):
            A_drain[m] = A_fifo[m, Nt // 2, k]
        for n in range(Nt // 2):
            B_drain[n] = B_fifo[n, Mt, k]


def packed_int8xint8_systolic[
    M: int32,
    K: int32,
    N: int32,
    Mt: int32,
    Nt: int32,
    P: int32,  # packing factor
](A: "Int(8 * P)[M // P, K]", B: "Int(8 * P)[K, N // P]", C: "Int(8 * P)[M // P, N]"):
    local_A: int8[Mt, K]
    local_B: int16[K, Nt // 2]
    local_C: int32[Mt, Nt // 2]

    # k needs not be tiled, since it is temporal dimension
    for mi, ni in dsl.grid(M // Mt, N // Nt, name="outer_tile"):
        # reversed traversal, better for cascading systolic arrays with FIFOs
        # corresponds to the order of the previous `store_C_tile` output
        for ak, ai in dsl.grid(K, Mt // P, name="load_A_tile"):
            # reuse along the ni dimension
            if ni == 0:
                a: Int(8 * P) = A[mi * Mt // P + ai, ak]
                for p in range(P):
                    local_A[ai * P + p, ak] = a[p * 8 : (p + 1) * 8]
        for bk, bj in dsl.grid(K, Nt // P, name="load_B_tile"):
            # reuse along the mi dimension
            # since the inner access order is different from the outer one,
            # we cannot cache as a line buffer
            b: Int(8 * P) = B[bk, ni * Nt // P + bj]
            for p in range(P // 2):
                local_B[bk, bj * P + p] = b[p * 16 : (p + 1) * 16]
        systolic_tile_packed_int8xint8[K, Mt, Nt](
            local_A,
            local_B,
            local_C,
        )
        # reversed traversal, better for cascading systolic arrays with FIFOs
        for sj, si in dsl.grid(Nt // 2, Mt // P, name="store_C_tile"):
            c0: Int(8 * P) = 0
            c1: Int(8 * P) = 0
            for p in range(P):
                # pylint: disable=unsupported-assignment-operation
                x: int32 = local_C[si * P + p, sj]
                c0[p * 8 : (p + 1) * 8] = x[0:16]
                c1[p * 8 : (p + 1) * 8] = x[16:32]
            C[mi * Mt // P + si, ni * Nt + 2 * sj] = c0
            C[mi * Mt // P + si, ni * Nt + 2 * sj + 1] = c1


def schedule_systolic(s):
    if s.top_func_name == "systolic":
        assert len(s.inst_list) == 8
        tile_name = "systolic_tile"
        M0, M1 = s.inst_list[-2], s.inst_list[-1]
    elif s.top_func_name == "packed_systolic":
        assert len(s.inst_list) == 9
        tile_name = "systolic_tile"
        M0, M1 = s.inst_list[-3], s.inst_list[-2]
    elif s.top_func_name == "packed_int8xint8_systolic":
        assert len(s.inst_list) == 6
        tile_name = "systolic_tile_packed_int8xint8"
        M0, M1 = s.inst_list[-3], s.inst_list[-2]
    else:
        raise ValueError(
            f"Cannot apply `schedule_systolic` to function: {s.top_func_name}"
        )
    s.partition(s.local_C, dim=0)  # required, otherwise it will fail dataflow checking
    s.partition(s.local_A, dim=1)
    s.partition(s.local_B, dim=2)
    load_A_loop = s.get_loops(s.top_func_name)["outer_tile"]["ai"]
    if str(load_A_loop.loop.attributes["upper_bound"]) == "affine_map<() -> (1)>":
        load_A_loop = s.get_loops(s.top_func_name)["outer_tile"]["ak"]
    s.pipeline(load_A_loop)
    load_B_loop = s.get_loops(s.top_func_name)["outer_tile"]["bj"]
    if str(load_B_loop.loop.attributes["upper_bound"]) == "affine_map<() -> (1)>":
        load_B_loop = s.get_loops(s.top_func_name)["outer_tile"]["bk"]
    s.pipeline(load_B_loop)
    store_C_loop = s.get_loops(s.top_func_name)["outer_tile"]["si"]
    if str(store_C_loop.loop.attributes["upper_bound"]) == "affine_map<() -> (1)>":
        store_C_loop = s.get_loops(s.top_func_name)["outer_tile"]["sj"]
    s.pipeline(store_C_loop)
    tile_loop = s.get_loops(s.top_func_name)["outer_tile"]["ni"]
    s.dataflow(tile_loop)
    pe = s.unfold(f"{tile_name}:PE", [0, 1])  # specify which are spatial loops
    s.to(MockBuffer(tile_name, "A_fifo"), pe, axis=1, depth=M0 + 1)
    s.to(MockBuffer(tile_name, "B_fifo"), pe, axis=0, depth=M1 + 1)
    return s
