# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import allo.dataflow as df
from allo.ir.types import int32, Stream
from allo.memory import Layout
import numpy as np

# RRxRS->RS
# RSxSR->RR
LyW1 = Layout("RS0")
LyW2 = Layout("S0R")


def _test_tensor_parallelism():
    Ty = int32
    M, K, N, L = 8, 8, 8, 8
    P0 = 2
    Nt = N // P0

    @df.region()
    def top():
        Y: Stream[Ty[M, Nt], 2][P0]
        part_Z: Stream[Ty[M, L], 2][P0]

        @df.kernel(mapping=[P0])
        def gemm0(X: Ty[M, K], W1: Ty[K, N] @ LyW1):
            pn = df.get_pid()
            Y[pn].put(allo.matmul(X, W1))

        @df.kernel(mapping=[P0])
        def gemm1(W2: Ty[N, L] @ LyW2):
            pn = df.get_pid()
            part_Z[pn].put(allo.matmul(Y[pn].get(), W2))

        @df.kernel(mapping=[1])
        def acc(Z: Ty[M, L]):
            Z_out: Ty[M, L] = 0
            with allo.meta_for(P0) as i:
                Z_out[:, :] += part_Z[i].get()
            Z[:, :] = Z_out

    X = np.random.randint(0, 64, (M, K)).astype(np.int32)
    W1 = np.random.randint(0, 64, (K, N)).astype(np.int32)
    W2 = np.random.randint(0, 64, (N, L)).astype(np.int32)

    mod = df.build(top, target="aie")
    Z = np.zeros((M, L)).astype(np.int32)
    mod(X, W1, W2, Z)
    np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
    print("PASSED!")

    # simulator broken right now
    # sim_mod = df.build(top, target="simulator")
    # Z = np.zeros((M, L)).astype(np.int32)
    # sim_mod(X, W1, W2, Z)
    # np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
    # print("Dataflow Simulator Passed!")


if __name__ == "__main__":
    _test_tensor_parallelism()
