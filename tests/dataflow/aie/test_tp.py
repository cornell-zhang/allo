# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo.dataflow as df
from allo.ir.types import int16
import numpy as np
import allo
from allo.memory import Layout

# RRxRS->RS
# RSxSR->RR
LyW1 = Layout("RS0")
LyW2 = Layout("S0R")


def _test_tensor_parallelism():
    Ty = int16
    M, K, N, L = 32, 32, 32, 32
    P0 = 1
    Nt = N // P0

    @df.region()
    def top():
        Y = df.array(df.pipe(dtype=Ty, shape=(M, Nt), depth=2), shape=(P0,))
        part_Z = df.array(df.pipe(dtype=Ty, shape=(M, L), depth=2), shape=(P0,))

        @df.kernel(mapping=[P0])
        def gemm0(X: Ty[M, K], W1: Ty[K, N] @ LyW1):
            Y[0].put(allo.matmul(X, W1))

        @df.kernel(mapping=[P0])
        def gemm1(W2: Ty[N, L] @ LyW2):
            part_Z[0].put(allo.matmul(Y[0].get(), W2))

        @df.kernel(mapping=[P0])
        def acc(Z: Ty[M, L]):
            Z[:, :] = part_Z[0].get()

    X = np.random.randint(0, 64, (M, K)).astype(np.int16)
    W1 = np.random.randint(0, 64, (K, N)).astype(np.int16)
    W2 = np.random.randint(0, 64, (N, L)).astype(np.int16)

    mod = df.build(top, target="aie")
    Z = np.zeros((M, L)).astype(np.int16)
    mod(X, W1, W2, Z)
    np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
    print("PASSED!")

    # AIE broken right now
    # sim_mod = df.build(top, target="simulator")
    # Z = np.zeros((M, L)).astype(np.int16)
    # sim_mod(X, W1, W2, Z)
    # np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
    # print("Dataflow Simulator Passed!")


if __name__ == "__main__":
    _test_tensor_parallelism()
