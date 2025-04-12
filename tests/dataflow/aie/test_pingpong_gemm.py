# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, float32
import allo.dataflow as df
import numpy as np

Ty = int32
M, N, K = 16, 16, 16
Pm, Pn, Pk = 1, 1, 2
Mt, Nt, Kt = M // Pm, N // Pn, K // Pk


@df.region()
def top():
    pipe = df.array(df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(Pk-1, Pm, Pn))

    @df.kernel(mapping=[Pk, Pm, Pn])
    def gemm(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pk, pm, pn = df.get_pid()
        with allo.meta_if(pk > 0):
            C_in: Ty[Mt, Nt] = pipe[pk-1, pm, pn].get()
        with allo.meta_else():
            C_in: Ty[Mt, Nt] = 0
        C_out: Ty[Mt, Nt] = 0
        # S1S0 x S0S2 -> S1S2
        C_out[:, :] = (
            allo.matmul(
                A[pm * Mt : (pm + 1) * Mt, pk * Kt : (pk + 1) * Kt],
                B[pk * Kt : (pk + 1) * Kt, pn * Nt : (pn + 1) * Nt],
            )
            + C_in
        )
        with allo.meta_if(pk < Pk - 1):
            pipe[pk, pm, pn].put(C_out)
        with allo.meta_elif(pk == Pk - 1):
            C[pm * Mt : (pm + 1) * Mt, pn * Nt : (pn + 1) * Nt] = C_out


def test_cooperative_gemm():
    mod = df.build(top, target="aie")
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    test_cooperative_gemm()
