# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import Stream
import allo.dataflow as df
from allo.memory import Layout


def gen_gemm_mapping_primitive(Pm, Pn, Pk, col_num=4, row_num=4):
    # chain on k dimension
    mapping_primitives = []
    bases: list[list[str]] = []
    for i in range(Pm):
        bases.append([])
        for j in range(Pn):
            base = f"gemm_0_{i}_{j}"
            for k in range(1, Pk):
                mapping_primitives.append(("chain", [base, f"gemm_{k}_{i}_{j}"]))
                base += f"-gemm_{k}_{i}_{j}"
            bases[i].append(base)

    if Pn // col_num < 1 or Pm // row_num < 1:
        col_num, row_num = row_num, col_num
    if Pn < col_num:
        col_num = Pn
    if Pm < row_num:
        row_num = Pm
    if Pn // col_num > 1 or Pm // row_num > 1:
        for i in range(row_num):
            for j in range(col_num):
                bundle_list = []
                for p in range(Pm // row_num):
                    for q in range(Pn // col_num):
                        bundle_list.append(bases[i + row_num * p][j + col_num * q])
                mapping_primitives.append(("bundle", bundle_list))

    return mapping_primitives


def gen_gemm_mapping_primitive_v2(Pm, Pn, Pk, col_num=4, row_num=4):
    # chain on k dimension
    mapping_primitives = []
    bases: list[list[str]] = []
    for i in range(Pm):
        bases.append([])
        for j in range(Pn):
            base_ping = f"gemm_0_{i}_{j}"
            for k in range(1, Pk // 2):
                mapping_primitives.append(("chain", [base_ping, f"gemm_{k}_{i}_{j}"]))
                base_ping += f"-gemm_{k}_{i}_{j}"
            base_pong = f"gemm_{Pk//2}_{i}_{j}"
            for k in range(1, Pk // 2):
                mapping_primitives.append(
                    ("chain", [base_pong, f"gemm_{Pk//2+ k}_{i}_{j}"])
                )
                base_pong += f"-gemm_{Pk//2 + k}_{i}_{j}"
            bases[i].append((base_ping, base_pong))

    col_num //= 2
    if Pn // col_num < 1 or Pm // row_num < 1:
        col_num, row_num = row_num, col_num
    if Pn < col_num:
        col_num = Pn
    if Pm < row_num:
        row_num = Pm
    if Pn // col_num > 1 or Pm // row_num > 1:
        for i in range(row_num):
            for j in range(col_num):
                bundle_list = []
                for p in range(Pm // row_num):
                    for q in range(Pn // col_num):
                        bundle_list.append(bases[i + row_num * p][j + col_num * q])
                mapping_primitives.append(("bundle", bundle_list))

    return mapping_primitives


def GEMM(M, N, K, Pm, Pn, Pk, TyI, TyO, col_num=4, row_num=4):
    Mt, Nt = M // Pm, N // Pn

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
        pipe: Stream[TyO[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn], args=[A, B, C])
        def gemm(
            local_A: TyI[M, K] @ LyA, local_B: TyI[K, N] @ LyB, local_C: TyO[M, N] @ LyC
        ):
            pk, pm, pn = df.get_pid()
            C_in: TyO[Mt, Nt]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0
            C_out: TyO[Mt, Nt] = allo.add(allo.matmul(local_A, local_B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                local_C[:, :] = C_out

    return top, gen_gemm_mapping_primitive(Pm, Pn, Pk, col_num, row_num)
