# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import allo.dataflow as df
from allo.ir.types import int32
from allo.memory import Layout
import numpy as np

LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")

def _test_cannon():
    Ty = int32
    M, K, N = 16, 16, 16
    P = 2
    m, k, n = M // P, K // P, N // P

    @df.region()
    def top():
        A_pipe = df.array(df.pipe(dtype=Ty, shape=(m, k), depth=2), shape=(P, P))
        B_pipe = df.array(df.pipe(dtype=Ty, shape=(k, n), depth=2), shape=(P, P))

        @df.kernel(mapping=[P, P])
        def cannon(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):

            C_local: Ty[m, n] = 0

            with allo.meta_for(P) as l:
                A_local: Ty[m, k] = 0
                B_local: Ty[k, n] = 0

                for i in range(m):
                    for j in range(k):
                        A_local[i, j] = A[i, l * k + j]

                for i in range(k):
                    for j in range(n):
                        B_local[i, j] = B[l * k + i, j]

                C_local[:, :] += allo.matmul(A_local, B_local)

            C[:, :] = C_local

    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)

    mod = df.build(top, target="aie")

    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")

if __name__ == "__main__":
    _test_cannon()