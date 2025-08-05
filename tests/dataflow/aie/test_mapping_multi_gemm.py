# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int8, int16, bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from ml_dtypes import bfloat16 as np_bfloat16


def _test_batched_gemm(M, N, K, Pm, Pn, Pk, TyI, TyO):
    assert TyI == TyO
    Mt, Nt = M // Pm, N // Pn

    LyA = Layout("S2S3")
    LyB = Layout("S3S1")
    LyC = Layout("S2S1")

    @df.region()
    def top():
        pipe = df.array(
            df.pipe(dtype=TyO, shape=(Mt, Nt), depth=2), shape=(4, Pk - 1, Pm, Pn)
        )

        @df.kernel(mapping=[4, Pk, Pm, Pn])
        def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            pb, pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: TyO[Mt, Nt] = pipe[pb, pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: TyO[Mt, Nt] = 0
            C_out: TyO[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pb, pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mod = df.build(
        top,
        target="aie-mlir",
        profile=True,
        warmup=200,
        num_iters=1000,
        device_type="npu1_1col",
    )
    if TyI is int16:
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
    _test_batched_gemm(64, 64, 64, 2, 2, 1, int16, int16)
