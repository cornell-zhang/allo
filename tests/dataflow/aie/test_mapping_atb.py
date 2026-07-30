# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
import allo.dataflow as df
from allo.backend.aie import is_available
from allo.ir.types import int16, Stream
import numpy as np

Ty = int16
M, N, K = 64, 16, 16
RHO_VALUES = [1, 2, 4, 8]  # Change rho value here


def make_atb_top(rho):
    assert M % rho == 0
    Ma = M // rho

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe_a: Stream[Ty[Ma, K], 2][rho]
        pipe_b: Stream[Ty[K, N], 2][rho]
        pipe_c: Stream[Ty[Ma, N], 2][rho]

        @df.kernel(mapping=[1], args=[A])
        def load_a(local_A: Ty[M, K]):
            with allo.meta_for(rho) as i:
                pipe_a[i].put(local_A[i * Ma : (i + 1) * Ma, :])

        @df.kernel(mapping=[1], args=[B])
        def load_b(local_B: Ty[K, N]):
            with allo.meta_for(rho) as i:
                pipe_b[i].put(local_B)

        @df.kernel(mapping=[rho])
        def compute():
            pk = df.get_pid()
            local_A: Ty[Ma, K] = pipe_a[pk].get()
            local_B: Ty[K, N] = pipe_b[pk].get()
            pipe_c[pk].put(allo.matmul(local_A, local_B))

        @df.kernel(mapping=[1], args=[C])
        def store_c(local_C: Ty[M, N]):
            with allo.meta_for(rho) as i:
                local_C[i * Ma : (i + 1) * Ma, :] = pipe_c[i].get()

    return top


def run_atb(rho):
    top = make_atb_top(rho)
    mapping_primitives = None
    if rho > 1:
        mapping_primitives = [("bundle", [f"compute_{i}" for i in range(rho)])]

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)

    if is_available():
        os.environ["FORCE_UNROLL_INDEX"] = "1"
        mod = df.build(top, target="aie", mapping_primitives=mapping_primitives)
        mod(A, B, C)
        del os.environ["FORCE_UNROLL_INDEX"]
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print(f"rho={rho} PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    for rho in RHO_VALUES:
        run_atb(rho)
