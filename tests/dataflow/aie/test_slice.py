# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout


def _test_store_slice():

    Ty = int16
    N = 32
    Pk = 4

    @df.region()
    def top_v1():

        @df.kernel(mapping=[1])
        def core(A: Ty[N], B: Ty[Pk, N]):
            for i in range(Pk):
                B[i, :] = A

    A = np.random.randint(0, 64, (N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    mod = df.build(top_v1, target="aie")
    mod(A, B)
    np.testing.assert_allclose(A, B[0, :], atol=1e-5)
    np.testing.assert_allclose(A, B[1, :], atol=1e-5)
    np.testing.assert_allclose(A, B[2, :], atol=1e-5)
    np.testing.assert_allclose(A, B[3, :], atol=1e-5)
    print("PASSED!")

    @df.region()
    def top_v2():

        @df.kernel(mapping=[1])
        def core(A: Ty[N], B: Ty[Pk, N]):
            for i in range(0, Pk, 2):
                B[i, :] = allo.add(A, 1)
            for i in range(1, Pk, 2):
                B[i, :] = allo.add(A, -1)

    mod = df.build(top_v2, target="aie")
    mod(A, B)
    np.testing.assert_allclose(A + 1, B[0, :], atol=1e-5)
    np.testing.assert_allclose(A - 1, B[1, :], atol=1e-5)
    np.testing.assert_allclose(A + 1, B[2, :], atol=1e-5)
    np.testing.assert_allclose(A - 1, B[3, :], atol=1e-5)
    print("PASSED!")


def _test_load_slice():

    Ty = int16
    N = 32
    Pk = 4

    @df.region()
    def top():

        @df.kernel(mapping=[1])
        def core(A: Ty[Pk, N], B: Ty[Pk, N]):
            for i in range(Pk):
                B[i, :] = A[i, :]

    A = np.random.randint(0, 64, (Pk, N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    mod = df.build(top, target="aie")
    mod(A, B)
    np.testing.assert_allclose(A, B, atol=1e-5)
    print("PASSED!")


def _test_store_slice1():

    Ty = int16
    N = 32
    Pk = 4

    @df.region()
    def top():
        pipe = df.array(df.pipe(dtype=Ty, shape=(N,), depth=2), shape=(Pk,))

        @df.kernel(mapping=[Pk])
        def src(A: Ty[N]):
            pk = df.get_pid()
            pipe[pk].put(A)

        @df.kernel(mapping=[1])
        def dst(B: Ty[Pk, N]):
            B[:, :] = df.gather([pipe[0], pipe[1], pipe[2], pipe[3]])

    A = np.random.randint(0, 64, (N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    mod_v1 = df.build(top, target="aie")
    mod_v1(A, B)
    np.testing.assert_allclose(A, B[0, :], atol=1e-5)
    np.testing.assert_allclose(A, B[1, :], atol=1e-5)
    np.testing.assert_allclose(A, B[2, :], atol=1e-5)
    np.testing.assert_allclose(A, B[3, :], atol=1e-5)
    print("PASSED!")

    mod_v2 = df.build(
        top,
        target="aie",
        mapping_primitives=[("bundle", ["src_0", "src_1", "src_2", "src_3"])],
    )
    mod_v2(A, B)
    np.testing.assert_allclose(A, B[0, :], atol=1e-5)
    np.testing.assert_allclose(A, B[1, :], atol=1e-5)
    np.testing.assert_allclose(A, B[2, :], atol=1e-5)
    np.testing.assert_allclose(A, B[3, :], atol=1e-5)
    print("PASSED!")


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
                # [NOTE]: will be left as UB
                buffer[i, :, :] = pipe[i].get()
            """
            buffer: Ty[Pk, M, N] = allo.gather(pipe[:].get())
            """
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
    # _test_store_slice()
    # _test_load_slice()
    _test_store_slice1()
    # _test_split_k_explicit_gather_gemm_1x1x4()
