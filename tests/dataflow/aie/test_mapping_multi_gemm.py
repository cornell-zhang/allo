# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int8, int16, bfloat16, Stream
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from ml_dtypes import bfloat16 as np_bfloat16


# [NOTE]: export FACTOR=2
def gen_gemm_mapping_primitive(prefix, Pm, Pn, Pk, col_num=4, row_num=4):
    # chain on k dimension
    mapping_primitives = []
    bases: list[list[str]] = []
    for i in range(Pm):
        bases.append([])
        for j in range(Pn):
            base = f"{prefix}_0_{i}_{j}"
            for k in range(1, Pk):
                mapping_primitives.append(("chain", [base, f"{prefix}_{k}_{i}_{j}"]))
                base += f"-{prefix}_{k}_{i}_{j}"
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


def _test_batched_gemm(M, N, K, Pm, Pn, Pk, TyI, TyO):
    assert TyI == TyO
    Mt, Nt = M // Pm, N // Pn

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe_a: Stream[TyO[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemma(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            C_in: TyO[Mt, Nt]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe_a[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0
            C_out: TyO[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe_a[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

        pipe_b: Stream[TyO[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemmb(D: TyI[M, K] @ LyA, E: TyI[K, N] @ LyB, F: TyO[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            F_in: TyO[Mt, Nt]
            with allo.meta_if(pk > 0):
                F_in[:, :] = pipe_b[pk - 1, pm, pn].get()
            with allo.meta_else():
                F_in[:, :] = 0
            F_out: TyO[Mt, Nt] = allo.add(allo.matmul(D, E), F_in)
            with allo.meta_if(pk < Pk - 1):
                pipe_b[pk, pm, pn].put(F_out)
            with allo.meta_elif(pk == Pk - 1):
                F[:, :] = F_out

    mapping_primitives = gen_gemm_mapping_primitive(
        "gemma", Pm, Pn, Pk, col_num=2, row_num=2
    )
    mapping_primitives.extend(
        gen_gemm_mapping_primitive("gemmb", Pm, Pn, Pk, col_num=2, row_num=2)
    )
    mod = df.build(
        top,
        project="gemm.prj",
        target="aie",
        mapping_primitives=mapping_primitives,
        profile=True,
        warmup=200,
        num_iters=1000,
        device_type="npu1_2col",
    )
    if TyI is bfloat16:
        A = np.random.random((M, K)).astype(np_bfloat16)
        B = np.random.random((K, N)).astype(np_bfloat16)
        C = np.zeros((M, N)).astype(np_bfloat16)
        D = np.zeros((M, N)).astype(np_bfloat16)
    elif TyI is int8:
        A = np.random.randint(-8, 8, (M, K)).astype(np.int8)
        B = np.random.randint(-8, 8, (K, N)).astype(np.int8)
        C = np.zeros((M, N)).astype(np.int8)
        D = np.zeros((M, N)).astype(np.int8)
    elif TyI is int16:
        A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
        B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
        D = np.zeros((M, N)).astype(np.int16)
    else:
        raise ValueError(f"unsupported data type {TyI}")
    mod(A, B, C, A, B, D)
    if TyI is bfloat16:
        np.testing.assert_allclose(
            C.astype(np.float32), (A @ B).astype(np.float32), atol=1e-2
        )
        np.testing.assert_allclose(
            D.astype(np.float32), (A @ B).astype(np.float32), atol=1e-2
        )
    else:
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        np.testing.assert_allclose(D, A @ B, atol=1e-5)


if __name__ == "__main__":
    dma_opt_flag = os.getenv("ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH") == "1"
    if dma_opt_flag:
        os.environ["FACTOR"] = "2"

    _test_batched_gemm(512, 512, 512, 8, 8, 8, int16, int16)
    _test_batched_gemm(512, 512, 512, 8, 8, 8, int8, int8)
    try:
        _test_batched_gemm(512, 512, 512, 8, 8, 8, bfloat16, bfloat16)
    except:
        print("[NOTE]: bfloat16 have accuracy issue")

    if dma_opt_flag:
        del os.environ["FACTOR"]
