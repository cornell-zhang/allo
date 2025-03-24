# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, float32
import allo.dataflow as df
import numpy as np

Ty = int32
M, N, K = 16, 16, 16
P0, P1 = 2, 2
Mt, Nt = M // P0, N // P1


@df.region()
def top():
    pipe = df.array(df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(P0, P1))

    @df.kernel(mapping=[P0, P1])
    def gemm0(A: Ty[M, K], B: Ty[K, N]):
        p0, p1 = df.get_pid()
        C_out: Ty[Mt, Nt] = 0
        C_out[:, :] = allo.matmul(
            A[p0 * Mt : (p0 + 1) * Mt, : K // 2], B[: K // 2, p1 * Nt : (p1 + 1) * Nt]
        )
        pipe[p0, p1].put(C_out)

    @df.kernel(mapping=[P0, P1])
    def gemm1(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        p0, p1 = df.get_pid()
        C_out: Ty[Mt, Nt] = pipe[p0, p1].get()
        C[p0 * Mt : (p0 + 1) * Mt, p1 * Nt : (p1 + 1) * Nt] = (
            allo.matmul(
                A[p0 * Mt : (p0 + 1) * Mt, K // 2 :],
                B[K // 2 :, p1 * Nt : (p1 + 1) * Nt],
            )
            + C_out
        )


def test_cooperative_gemm():
    mod = df.build(top, target="aie")
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    test_cooperative_gemm()
