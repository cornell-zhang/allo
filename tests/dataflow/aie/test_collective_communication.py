# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16, int32, Stream
import allo.dataflow as df
import numpy as np
from allo.memory import Layout


def _test_gather():

    Ty = int16
    N = 32
    Pk = 4

    @df.region()
    def top_v1():
        pipe: Stream[Ty[N], 2][Pk]

        @df.kernel(mapping=[Pk])
        def src(A: Ty[N]):
            pk = df.get_pid()
            pipe[pk].put(A)

        @df.kernel(mapping=[1])
        def dst(B: Ty[Pk, N]):
            # gather a list of pipes
            B[:, :] = df.gather([pipe[0], pipe[1], pipe[2], pipe[3]])

    A = np.random.randint(0, 64, (N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    mod_v1 = df.build(top_v1, target="aie")
    mod_v1(A, B)
    np.testing.assert_allclose(A, B[0, :], atol=1e-5)
    np.testing.assert_allclose(A, B[1, :], atol=1e-5)
    np.testing.assert_allclose(A, B[2, :], atol=1e-5)
    np.testing.assert_allclose(A, B[3, :], atol=1e-5)
    print("PASSED!")

    mod_v2 = df.build(
        top_v1,
        target="aie",
        mapping_primitives=[("bundle", ["src_0", "src_1", "src_2", "src_3"])],
    )
    mod_v2(A, B)
    np.testing.assert_allclose(A, B[0, :], atol=1e-5)
    np.testing.assert_allclose(A, B[1, :], atol=1e-5)
    np.testing.assert_allclose(A, B[2, :], atol=1e-5)
    np.testing.assert_allclose(A, B[3, :], atol=1e-5)
    print("PASSED!")

    @df.region()
    def top_v2():
        pipe: Stream[Ty[N], 2][Pk]

        @df.kernel(mapping=[Pk])
        def src(A: Ty[N]):
            pk = df.get_pid()
            pipe[pk].put(A)

        @df.kernel(mapping=[1])
        def dst(B: Ty[Pk, N]):
            # gather slice
            B[:, :] = df.gather(pipe[:])

    A = np.random.randint(0, 64, (N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    mod_v1 = df.build(top_v2, target="aie")
    mod_v1(A, B)
    np.testing.assert_allclose(A, B[0, :], atol=1e-5)
    np.testing.assert_allclose(A, B[1, :], atol=1e-5)
    np.testing.assert_allclose(A, B[2, :], atol=1e-5)
    np.testing.assert_allclose(A, B[3, :], atol=1e-5)
    print("PASSED!")

    mod_v2 = df.build(
        top_v2,
        target="aie",
        mapping_primitives=[("bundle", ["src_0", "src_1", "src_2", "src_3"])],
    )
    mod_v2(A, B)
    np.testing.assert_allclose(A, B[0, :], atol=1e-5)
    np.testing.assert_allclose(A, B[1, :], atol=1e-5)
    np.testing.assert_allclose(A, B[2, :], atol=1e-5)
    np.testing.assert_allclose(A, B[3, :], atol=1e-5)
    print("PASSED!")


def _test_gather_matmul():

    TyI, TyO = int16, int32
    M, N, K = 32, 32, 16
    Pk = 2

    @df.region()
    def top():
        pipe: Stream[TyI[M, K], 2][Pk]

        @df.kernel(mapping=[Pk])
        def src(A: TyI[M, K]):
            pk = df.get_pid()
            pipe[pk].put(A)

        @df.kernel(mapping=[1])
        def dst(B: TyI[K, N], C: TyO[M, N]):
            tmp_A: TyI[Pk, M, K] = df.gather(pipe[:])
            # [NOTE]: test using buffer slice as mixed precision matmul input
            C_: TyO[M, N] = allo.matmul(tmp_A[0], B)
            C[:, :] = C_

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int32)

    mod = df.build(top, target="aie")
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)


def _test_split_k_explicit_gather_gemm_1x1x4():

    Ty = int16
    M, N, K = 16, 16, 64
    Pk = 4

    LyA = Layout("RS0")
    LyB = Layout("S0R")

    @df.region()
    def top_v1():
        pipe: Stream[Ty[M, N], 2][Pk]

        @df.kernel(mapping=[Pk])
        def partial_gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB):
            pk = df.get_pid()
            pipe[pk].put(allo.matmul(A, B))

        @df.kernel(mapping=[1])
        def acc(C: Ty[M, N]):
            C_: Ty[M, N] = 0
            # gather
            buffer: Ty[Pk, M, N] = df.gather([pipe[0], pipe[1], pipe[2], pipe[3]])
            # accumulate
            for i in range(Pk):
                C_[:, :] += buffer[i]
            C[:, :] = C_

    mod = df.build(
        top_v1,
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

    @df.region()
    def top_v2():
        pipe: Stream[Ty[M, N], 2][Pk]

        @df.kernel(mapping=[Pk])
        def partial_gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB):
            pk = df.get_pid()
            pipe[pk].put(allo.matmul(A, B))

        @df.kernel(mapping=[1])
        def acc(C: Ty[M, N]):
            C_: Ty[M, N] = 0
            # gather
            buffer: Ty[Pk, M, N] = df.gather(pipe[:])
            # accumulate
            for i in range(Pk):
                # [NOTE]: test using buffer slice in builtin external kernel
                C_[:, :] += buffer[i]
            C[:, :] = C_

    mod = df.build(
        top_v2,
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


def _test_scatter():

    Ty = int16
    N = 32
    Pk = 2

    @df.region()
    def top_v1():
        pipe_1: Stream[Ty[N], 2][Pk]
        pipe_2: Stream[Ty[N], 2][Pk]

        @df.kernel(mapping=[1])
        def src(A: Ty[Pk, N]):
            df.scatter(A, [pipe_1[0], pipe_1[1]])

        @df.kernel(mapping=[Pk])
        def inter():
            pk = df.get_pid()
            pipe_2[pk].put(allo.add(pipe_1[pk].get(), 1))

        @df.kernel(mapping=[1])
        def dst(B: Ty[Pk, N]):
            B[:, :] = df.gather(pipe_2[:])

    A = np.random.randint(0, 64, (Pk, N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    mod_v1 = df.build(top_v1, target="aie")
    mod_v1(A, B)
    np.testing.assert_allclose(A + 1, B, atol=1e-5)
    print("PASSED!")

    mod_v2 = df.build(
        top_v1, target="aie", mapping_primitives=[("bundle", ["inter_0", "inter_1"])]
    )
    mod_v2(A, B)
    np.testing.assert_allclose(A + 1, B, atol=1e-5)
    print("PASSED!")

    @df.region()
    def top_v2():
        pipe_1: Stream[Ty[N], 2][Pk]
        pipe_2: Stream[Ty[N], 2][Pk]

        @df.kernel(mapping=[1])
        def src(A: Ty[Pk, N]):
            df.scatter(A, pipe_1[:])

        @df.kernel(mapping=[Pk])
        def inter():
            pk = df.get_pid()
            pipe_2[pk].put(allo.add(pipe_1[pk].get(), 1))

        @df.kernel(mapping=[1])
        def dst(B: Ty[Pk, N]):
            B[:, :] = df.gather(pipe_2[:])

    A = np.random.randint(0, 64, (Pk, N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    mod_v1 = df.build(top_v2, target="aie")
    mod_v1(A, B)
    np.testing.assert_allclose(A + 1, B, atol=1e-5)
    print("PASSED!")

    mod_v2 = df.build(
        top_v2, target="aie", mapping_primitives=[("bundle", ["inter_0", "inter_1"])]
    )
    mod_v2(A, B)
    np.testing.assert_allclose(A + 1, B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_gather()
    _test_gather_matmul()
    _test_scatter()
    _test_split_k_explicit_gather_gemm_1x1x4()
