# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.experimental.external_kernel import ExternalModule


def _test_pingpong_gemm_4x4x4(TyI, TyO):
    """
    Manually do reduction for split_k
    """
    M, N, K = 128, 128, 512
    matmul = ExternalModule(
        top=f"matmul_scalar_{TyI}_{TyO}",
        impl_path="matmul_scalar.cc",
        input_idx=[0, 1, 2],
        output_idx=[3],
    )
    Pm, Pn, Pk = 4, 4, 4
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

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
                C_local: TyO[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_local: TyO[Mt, Nt] = 0
            matmul(A, B, C_local, C_local)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_local)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_local

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            ("chain", ["gemm_0_0_0", "gemm_1_0_0"]),
            ("chain", ["gemm_0_0_0-gemm_1_0_0", "gemm_2_0_0"]),
            ("chain", ["gemm_0_0_0-gemm_1_0_0-gemm_2_0_0", "gemm_3_0_0"]),
            ("chain", ["gemm_0_0_1", "gemm_1_0_1"]),
            ("chain", ["gemm_0_0_1-gemm_1_0_1", "gemm_2_0_1"]),
            ("chain", ["gemm_0_0_1-gemm_1_0_1-gemm_2_0_1", "gemm_3_0_1"]),
            ("chain", ["gemm_0_0_2", "gemm_1_0_2"]),
            ("chain", ["gemm_0_0_2-gemm_1_0_2", "gemm_2_0_2"]),
            ("chain", ["gemm_0_0_2-gemm_1_0_2-gemm_2_0_2", "gemm_3_0_2"]),
            ("chain", ["gemm_0_0_3", "gemm_1_0_3"]),
            ("chain", ["gemm_0_0_3-gemm_1_0_3", "gemm_2_0_3"]),
            ("chain", ["gemm_0_0_3-gemm_1_0_3-gemm_2_0_3", "gemm_3_0_3"]),
            ("chain", ["gemm_0_1_0", "gemm_1_1_0"]),
            ("chain", ["gemm_0_1_0-gemm_1_1_0", "gemm_2_1_0"]),
            ("chain", ["gemm_0_1_0-gemm_1_1_0-gemm_2_1_0", "gemm_3_1_0"]),
            ("chain", ["gemm_0_1_1", "gemm_1_1_1"]),
            ("chain", ["gemm_0_1_1-gemm_1_1_1", "gemm_2_1_1"]),
            ("chain", ["gemm_0_1_1-gemm_1_1_1-gemm_2_1_1", "gemm_3_1_1"]),
            ("chain", ["gemm_0_1_2", "gemm_1_1_2"]),
            ("chain", ["gemm_0_1_2-gemm_1_1_2", "gemm_2_1_2"]),
            ("chain", ["gemm_0_1_2-gemm_1_1_2-gemm_2_1_2", "gemm_3_1_2"]),
            ("chain", ["gemm_0_1_3", "gemm_1_1_3"]),
            ("chain", ["gemm_0_1_3-gemm_1_1_3", "gemm_2_1_3"]),
            ("chain", ["gemm_0_1_3-gemm_1_1_3-gemm_2_1_3", "gemm_3_1_3"]),
            ("chain", ["gemm_0_2_0", "gemm_1_2_0"]),
            ("chain", ["gemm_0_2_0-gemm_1_2_0", "gemm_2_2_0"]),
            ("chain", ["gemm_0_2_0-gemm_1_2_0-gemm_2_2_0", "gemm_3_2_0"]),
            ("chain", ["gemm_0_2_1", "gemm_1_2_1"]),
            ("chain", ["gemm_0_2_1-gemm_1_2_1", "gemm_2_2_1"]),
            ("chain", ["gemm_0_2_1-gemm_1_2_1-gemm_2_2_1", "gemm_3_2_1"]),
            ("chain", ["gemm_0_2_2", "gemm_1_2_2"]),
            ("chain", ["gemm_0_2_2-gemm_1_2_2", "gemm_2_2_2"]),
            ("chain", ["gemm_0_2_2-gemm_1_2_2-gemm_2_2_2", "gemm_3_2_2"]),
            ("chain", ["gemm_0_2_3", "gemm_1_2_3"]),
            ("chain", ["gemm_0_2_3-gemm_1_2_3", "gemm_2_2_3"]),
            ("chain", ["gemm_0_2_3-gemm_1_2_3-gemm_2_2_3", "gemm_3_2_3"]),
            ("chain", ["gemm_0_3_0", "gemm_1_3_0"]),
            ("chain", ["gemm_0_3_0-gemm_1_3_0", "gemm_2_3_0"]),
            ("chain", ["gemm_0_3_0-gemm_1_3_0-gemm_2_3_0", "gemm_3_3_0"]),
            ("chain", ["gemm_0_3_1", "gemm_1_3_1"]),
            ("chain", ["gemm_0_3_1-gemm_1_3_1", "gemm_2_3_1"]),
            ("chain", ["gemm_0_3_1-gemm_1_3_1-gemm_2_3_1", "gemm_3_3_1"]),
            ("chain", ["gemm_0_3_2", "gemm_1_3_2"]),
            ("chain", ["gemm_0_3_2-gemm_1_3_2", "gemm_2_3_2"]),
            ("chain", ["gemm_0_3_2-gemm_1_3_2-gemm_2_3_2", "gemm_3_3_2"]),
            ("chain", ["gemm_0_3_3", "gemm_1_3_3"]),
            ("chain", ["gemm_0_3_3-gemm_1_3_3", "gemm_2_3_3"]),
            ("chain", ["gemm_0_3_3-gemm_1_3_3-gemm_2_3_3", "gemm_3_3_3"]),
        ],
        profile=True,
        warmup=200,
        num_iters=1000,
    )
    A = np.random.randint(-16, 16, (M, K)).astype(np.int16)
    B = np.random.randint(-16, 16, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_pingpong_gemm_4x4x4(int16, int32)
