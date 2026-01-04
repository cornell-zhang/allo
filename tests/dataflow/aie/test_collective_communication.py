# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16, int32, Stream
import allo.dataflow as df
import numpy as np
from allo.memory import MemLayout
from allo.backend.aie import is_available


def test_gather():

    Ty = int16
    N = 32
    Pk = 4

    @df.region()
    def top_v1(A: Ty[N], B: Ty[Pk, N]):
        pipe: Stream[Ty[N], 2][Pk]

        @df.kernel(mapping=[Pk], args=[A])
        def src(local_A: Ty[N]):
            pk = df.get_pid()
            pipe[pk].put(local_A)

        @df.kernel(mapping=[1], args=[B])
        def dst(local_B: Ty[Pk, N]):
            # gather a list of pipes
            local_B[:, :] = df.gather([pipe[0], pipe[1], pipe[2], pipe[3]])

    A = np.random.randint(0, 64, (N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    if is_available():
        mod_v1 = df.build(top_v1, target="aie")
        mod_v1(A, B)
        np.testing.assert_allclose(A, B[0, :], atol=1e-5)
        np.testing.assert_allclose(A, B[1, :], atol=1e-5)
        np.testing.assert_allclose(A, B[2, :], atol=1e-5)
        np.testing.assert_allclose(A, B[3, :], atol=1e-5)
        print("V1 PASSED!")

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
        print("V2 PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    @df.region()
    def top_v2(A: Ty[N], B: Ty[Pk, N]):
        pipe: Stream[Ty[N], 2][Pk]

        @df.kernel(mapping=[Pk], args=[A])
        def src(local_A: Ty[N]):
            pk = df.get_pid()
            pipe[pk].put(local_A)

        @df.kernel(mapping=[1], args=[B])
        def dst(local_B: Ty[Pk, N]):
            # gather slice
            local_B[:, :] = df.gather(pipe[:])

    A = np.random.randint(0, 64, (N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)
    if is_available():
        mod_v1 = df.build(top_v2, target="aie")
        mod_v1(A, B)
        np.testing.assert_allclose(A, B[0, :], atol=1e-5)
        np.testing.assert_allclose(A, B[1, :], atol=1e-5)
        np.testing.assert_allclose(A, B[2, :], atol=1e-5)
        np.testing.assert_allclose(A, B[3, :], atol=1e-5)
        print("V1 PASSED!")

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
        print("V2 PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_gather_matmul():

    TyI, TyO = int16, int32
    M, N, K = 32, 32, 16
    Pk = 2

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
        pipe: Stream[TyI[M, K], 2][Pk]

        @df.kernel(mapping=[Pk], args=[A])
        def src(local_A: TyI[M, K]):
            pk = df.get_pid()
            pipe[pk].put(local_A)

        @df.kernel(mapping=[1], args=[B, C])
        def dst(local_B: TyI[K, N], local_C: TyO[M, N]):
            tmp_A: TyI[Pk, M, K] = df.gather(pipe[:])
            # [NOTE]: test using buffer slice as mixed precision matmul input
            C_: TyO[M, N] = allo.matmul(tmp_A[0], local_B)
            local_C[:, :] = C_

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int32)

    if is_available():
        mod = df.build(top, target="aie")
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_split_k_explicit_gather_gemm_1x1x4():

    Ty = int16
    M, N, K = 16, 16, 64
    Pk = 4

    LyA = MemLayout("RS0")
    LyB = MemLayout("S0R")

    @df.region()
    def top_v1(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[M, N], 2][Pk]

        @df.kernel(mapping=[Pk], args=[A, B])
        def partial_gemm(local_A: Ty[M, K] @ LyA, local_B: Ty[K, N] @ LyB):
            pk = df.get_pid()
            pipe[pk].put(allo.matmul(local_A, local_B))

        @df.kernel(mapping=[1], args=[C])
        def acc(local_C: Ty[M, N]):
            C_: Ty[M, N] = 0
            # gather
            buffer: Ty[Pk, M, N] = df.gather([pipe[0], pipe[1], pipe[2], pipe[3]])
            # accumulate
            for i in range(Pk):
                C_[:, :] += buffer[i]
            local_C[:, :] = C_

    if is_available():
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
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    @df.region()
    def top_v2(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[M, N], 2][Pk]

        @df.kernel(mapping=[Pk], args=[A, B])
        def partial_gemm(local_A: Ty[M, K] @ LyA, local_B: Ty[K, N] @ LyB):
            pk = df.get_pid()
            pipe[pk].put(allo.matmul(local_A, local_B))

        @df.kernel(mapping=[1], args=[C])
        def acc(local_C: Ty[M, N]):
            C_: Ty[M, N] = 0
            # gather
            buffer: Ty[Pk, M, N] = df.gather(pipe[:])
            # accumulate
            for i in range(Pk):
                # [NOTE]: test using buffer slice in builtin external kernel
                C_[:, :] += buffer[i]
            local_C[:, :] = C_

    if is_available():
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
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_scatter():

    Ty = int16
    N = 32
    Pk = 2

    @df.region()
    def top_v1(A: Ty[Pk, N], B: Ty[Pk, N]):
        pipe_1: Stream[Ty[N], 2][Pk]
        pipe_2: Stream[Ty[N], 2][Pk]

        @df.kernel(mapping=[1], args=[A])
        def src(local_A: Ty[Pk, N]):
            df.scatter(local_A, [pipe_1[0], pipe_1[1]])

        @df.kernel(mapping=[Pk])
        def inter():
            pk = df.get_pid()
            pipe_2[pk].put(allo.add(pipe_1[pk].get(), 1))

        @df.kernel(mapping=[1], args=[B])
        def dst(local_B: Ty[Pk, N]):
            local_B[:, :] = df.gather(pipe_2[:])

    A = np.random.randint(0, 64, (Pk, N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)

    if is_available():
        mod_v1 = df.build(top_v1, target="aie")
        mod_v1(A, B)
        np.testing.assert_allclose(A + 1, B, atol=1e-5)
        print("V1 PASSED!")

        mod_v2 = df.build(
            top_v1,
            target="aie",
            mapping_primitives=[("bundle", ["inter_0", "inter_1"])],
        )
        mod_v2(A, B)
        np.testing.assert_allclose(A + 1, B, atol=1e-5)
        print("V2 PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    @df.region()
    def top_v2(A: Ty[Pk, N], B: Ty[Pk, N]):
        pipe_1: Stream[Ty[N], 2][Pk]
        pipe_2: Stream[Ty[N], 2][Pk]

        @df.kernel(mapping=[1], args=[A])
        def src(local_A: Ty[Pk, N]):
            df.scatter(local_A, pipe_1[:])

        @df.kernel(mapping=[Pk])
        def inter():
            pk = df.get_pid()
            pipe_2[pk].put(allo.add(pipe_1[pk].get(), 1))

        @df.kernel(mapping=[1], args=[B])
        def dst(local_B: Ty[Pk, N]):
            local_B[:, :] = df.gather(pipe_2[:])

    A = np.random.randint(0, 64, (Pk, N)).astype(np.int16)
    B = np.zeros((Pk, N)).astype(np.int16)
    if is_available():
        mod_v1 = df.build(top_v2, target="aie")
        mod_v1(A, B)
        np.testing.assert_allclose(A + 1, B, atol=1e-5)
        print("V1 PASSED!")

        mod_v2 = df.build(
            top_v2,
            target="aie",
            mapping_primitives=[("bundle", ["inter_0", "inter_1"])],
        )
        mod_v2(A, B)
        np.testing.assert_allclose(A + 1, B, atol=1e-5)
        print("V2 PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_gather()
    test_gather_matmul()
    test_scatter()
    test_split_k_explicit_gather_gemm_1x1x4()
