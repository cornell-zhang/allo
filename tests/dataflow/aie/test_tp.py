# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import allo.dataflow as df
from allo.ir.types import int32, Stream
from allo.memory import Layout
import numpy as np
from allo.backend.aie import is_available

S = Layout.Shard
R = Layout.Replicate
# RRxRS->RS
# RSxSR->RR
LyW1 = [R, S(0)]
LyW2 = [S(0), R]


def test_tensor_parallelism():
    Ty = int32
    M, K, N, L = 8, 8, 8, 8
    P0 = 2
    Nt = N // P0

    @df.region()
    def top(X: Ty[M, K], W1: Ty[K, N], W2: Ty[N, L], Z: Ty[M, L]):
        Y: Stream[Ty[M, Nt], 2][P0]
        part_Z: Stream[Ty[M, L], 2][P0]

        @df.kernel(mapping=[P0], args=[X, W1])
        def gemm0(local_X: Ty[M, K], local_W1: Ty[K, N] @ LyW1):
            pn = df.get_pid()
            Y[pn].put(allo.matmul(local_X, local_W1))

        @df.kernel(mapping=[P0], args=[W2])
        def gemm1(local_W2: Ty[N, L] @ LyW2):
            pn = df.get_pid()
            part_Z[pn].put(allo.matmul(Y[pn].get(), local_W2))

        @df.kernel(mapping=[1], args=[Z])
        def acc(local_Z: Ty[M, L]):
            Z_out: Ty[M, L] = 0
            with allo.meta_for(P0) as i:
                Z_out[:, :] += part_Z[i].get()
            local_Z[:, :] = Z_out

    X = np.random.randint(0, 64, (M, K)).astype(np.int32)
    W1 = np.random.randint(0, 64, (K, N)).astype(np.int32)
    W2 = np.random.randint(0, 64, (N, L)).astype(np.int32)
    if is_available():
        mod = df.build(top, target="aie")
        Z = np.zeros((M, L)).astype(np.int32)
        mod(X, W1, W2, Z)
        np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    # simulator broken right now
    # sim_mod = df.build(top, target="simulator")
    # Z = np.zeros((M, L)).astype(np.int32)
    # sim_mod(X, W1, W2, Z)
    # np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
    # print("Dataflow Simulator Passed!")


if __name__ == "__main__":
    test_tensor_parallelism()
