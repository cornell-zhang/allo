# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout


def _test_split_k_explicit_gather_gemm_1x1x4():

    Ty = int16
    M, N, K = 32, 32, 64
    Pk = 4

    LyA = Layout("RS0")
    LyB = Layout("S0R")

    @df.region()
    def top():
        pipe = df.array(df.pipe(dtype=Ty, shape=(M, N), depth=2), shape=(Pk,))

        @df.kernel(mapping=[Pk])
        def partial_gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB):
            pk = df.get_pid()
            pipe[pk].put(allo.matmul(A, B))

        @df.kernel(mapping=[1])
        def acc(C: Ty[M, N]):
            C_: Ty[M, N] = 0
            # gather
            buffer: Ty[Pk, M, N]
            with allo.meta_for(Pk) as i:
                buffer[i, :, :] = pipe[i].get()
            # accumulate
            for i in range(Pk):
                C_[:, :] += buffer[i]
            C[:, :] = C_

    mod = df.build(
        top,
        target="aie",
        mapping_primitives=[
            (
                "bundle",
                [
                    "partial_gemm_0",
                    "partial_gemm_1",
                    "partial_gemm_2",
                    "partial_gemm_3",
                ],
            ),
        ],
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")

if __name__ == "__main__":
    os.environ["FORCE_UNROLL_INDEX"] = "1"
    _test_split_k_explicit_gather_gemm_1x1x4()
    del os.environ["FORCE_UNROLL_INDEX"]