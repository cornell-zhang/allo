# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import allo.dataflow as df
from allo.ir.types import int32
from allo.memory import Layout
import numpy as np

LyA = Layout("S0S1")
LyB = Layout("S0S1")
LyC = Layout("S0S1")


def _test_cannon():
    Ty = int32
    M, K, N = 32, 32, 32
    P = 2
    m, k, n = M // P, K // P, N // P

    @df.region()
    def top():
        A_init = df.array(df.pipe(dtype=Ty, shape=(m, k), depth=2), shape=(P, P))
        B_init = df.array(df.pipe(dtype=Ty, shape=(k, n), depth=2), shape=(P, P))

        A_pipe = df.array(df.pipe(dtype=Ty, shape=(m, k), depth=2), shape=(P, P))
        B_pipe = df.array(df.pipe(dtype=Ty, shape=(k, n), depth=2), shape=(P, P))

        @df.kernel(mapping=[P, P])
        def init(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB):
            pi, pj = df.get_pid()

            A_init[(pi - pj) % P, pj].put(A)
            B_init[pi, (pj - pi) % P].put(B)

        @df.kernel(mapping=[P, P])
        def cannon(C: Ty[M, N] @ LyC):
            pi, pj = df.get_pid()

            A_out: Ty[m, k] = A_init[pi, pj].get()
            B_out: Ty[k, n] = B_init[pi, pj].get()

            C[:, :] = allo.matmul(A_out, B_out)

            A_pipe[(pi - 1) % P, pj].put(A_out)
            B_pipe[pi, (pj - 1) % P].put(B_out)

            with allo.meta_for(P - 1) as i:
                A_out: Ty[m, k] = A_pipe[pi, pj].get()
                B_out: Ty[k, n] = B_pipe[pi, pj].get()

                C[:, :] += allo.matmul(A_out, B_out)
                if i < P - 1:
                    A_pipe[(pi - 1) % P, pj].put(A_out)
                    B_pipe[pi, (pj - 1) % P].put(B_out)

    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)

    mod = df.build(top, target="aie-mlir")

    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_cannon()
