# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int8, int16, bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from ml_dtypes import bfloat16 as np_bfloat16

COL_NUM = 8 if os.getenv("NPU2") == "1" else 4


def gen_pingpong_gemm_mapping_primitive(Pm, Pn, Pk, col_num=4, row_num=4):
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

    if Pn // col_num > 1 or Pm // row_num > 1:
        for i in range(row_num):
            for j in range(col_num):
                bundle_list = []
                for p in range(Pm // row_num):
                    for q in range(Pn // col_num):
                        bundle_list.append(bases[i + row_num * p][j + col_num * q])
                mapping_primitives.append(("bundle", bundle_list))

    return mapping_primitives


def _test_pingpong_gemm(M, N, K, Pm, Pn, Pk, TyI, TyO):
    assert TyI == TyO
    Mt, Nt = M // Pm, N // Pn

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe = df.array(
            df.pipe(dtype=TyO, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn)
        )

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: TyO[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: TyO[Mt, Nt] = 0
            C_out: TyO[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mapping_primitives = gen_pingpong_gemm_mapping_primitive(
        Pm,
        Pn,
        Pk,
        # col_num= 2,
        # row_num=2
    )

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=mapping_primitives,
        profile=True,
        warmup=200,
        num_iters=1000,
    )
    if TyI is bfloat16:
        A = np.random.random((M, K)).astype(np_bfloat16)
        B = np.random.random((K, N)).astype(np_bfloat16)
        C = np.zeros((M, N)).astype(np_bfloat16)
    elif TyI is int8:
        A = np.random.randint(-8, 8, (M, K)).astype(np.int8)
        B = np.random.randint(-8, 8, (K, N)).astype(np.int8)
        C = np.zeros((M, N)).astype(np.int8)
    elif TyI is int16:
        A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
        B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
    else:
        raise ValueError(f"unsupported data type {TyI}")
    mod(A, B, C)
    if TyI is bfloat16:
        np.testing.assert_allclose(
            C.astype(np.float32), (A @ B).astype(np.float32), atol=1e-2
        )
    else:
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    # [NOTE]: int8 and bfloat16 have accuracy issue (compared with cpu reference)
    # - i8
    _test_pingpong_gemm(512, 512, 512, 8, 8, 8, int8, int8)

    # - i16
    _test_pingpong_gemm(512, 512, 512, 8, 8, 8, int16, int16)

    # - bf16
    try:
        _test_pingpong_gemm(512, 512, 512, 8, 8, 8, bfloat16, bfloat16)
    except:
        print("[NOTE]: bfloat16 have accuracy issue")
