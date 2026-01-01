# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import allo
import allo.dataflow as df
from allo.ir.types import int32, Stream
from allo.memory import Layout
import numpy as np
from allo.backend.aie import is_available

# RRxRS->RS
# RSxSR->RR
LyW1 = Layout("RS0")
LyW2 = Layout("S0R")


def test_tp_v1():
    """
    aie backend by default tries to avoid unrolling `meta_for` to optimize code size.
    When the iterator of a rolled `meta_for` is used as the index of a pipe,
    current virtual mapping (especially chain) faces significant restrictions.

    If you prefer to sacrifice code size in exchange for using more mapping primitives,
    you can set `FORCE_UNROLL_INDEX` to prevent `meta_for` with index-based iterators from being optimized.
    """
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

    os.environ["FORCE_UNROLL_INDEX"] = "1"
    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("chain", ["gemm1_1", "acc_0"]),
            ],
        )
        Z = np.zeros((M, L)).astype(np.int32)
        mod(X, W1, W2, Z)
        np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("chain", ["gemm1_1", "acc_0"]),
                ("chain", ["gemm1_0", "gemm1_1-acc_0"]),
            ],
        )
        Z = np.zeros((M, L)).astype(np.int32)
        mod(X, W1, W2, Z)
        np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("chain", ["gemm1_1", "acc_0"]),
                ("chain", ["gemm1_0", "gemm1_1-acc_0"]),
            ],
        )
        Z = np.zeros((M, L)).astype(np.int32)
        mod(X, W1, W2, Z)
        np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
    del os.environ["FORCE_UNROLL_INDEX"]


def test_tp_v2():
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
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("bundle", [("gemm0_0", "gemm1_0"), ("gemm0_1", "gemm1_1")]),
            ],
        )
        Z = np.zeros((M, L)).astype(np.int32)
        mod(X, W1, W2, Z)
        np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_tp_v1()
    test_tp_v2()
