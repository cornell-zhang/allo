# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16, int32, Stream
import allo.dataflow as df
import numpy as np
from allo.memory import MemLayout
from allo.backend.aie import is_available


def test_vector_scalar_add():
    Ty = int32
    M = 1024

    @df.region()
    def top(A: Ty[M], B: Ty[M]):
        @df.kernel(mapping=[1], args=[A, B])
        def core(local_A: Ty[M], local_B: Ty[M]):
            # [NOTE]: this will be optimized as `for i in range(M):` (though such usage is not recommended)
            with allo.meta_for(M) as i:
                local_B[i] = local_A[i] + 1

    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)
    if is_available():
        mod = df.build(top, target="aie")
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_gather():

    Ty = int16
    M, N, K = 32, 32, 64
    Pk = 2

    LyA = MemLayout("RS0")
    LyB = MemLayout("S0R")

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[M, N], 2][Pk]

        @df.kernel(mapping=[Pk], args=[A, B])
        def partial_gemm(local_A: Ty[M, K] @ LyA, local_B: Ty[K, N] @ LyB):
            pk = df.get_pid()
            pipe[pk].put(allo.matmul(local_A, local_B))

        @df.kernel(mapping=[1], args=[C])
        def acc(local_C: Ty[M, N]):
            C_: Ty[M, N] = 0
            with allo.meta_for(Pk) as i:
                C_[:, :] += pipe[i].get()
            local_C[:, :] = C_

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    if is_available():
        mod_v1 = df.build(top, target="aie")
        mod_v1(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("V1 PASSED!")

        mod_v2 = df.build(
            top,
            target="aie",
            mapping_primitives=[
                (
                    "bundle",
                    [
                        "partial_gemm_0",
                        "partial_gemm_1",
                    ],
                ),
            ],
        )
        mod_v2(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("V2 PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_scatter():
    Ty = int32
    M = 1024
    P = 4
    Ly = MemLayout("S0")

    @df.region()
    def top(A: Ty[M], B: Ty[M]):
        pipe: Stream[Ty[M // P], 2][P]

        @df.kernel(mapping=[1])
        def prod():
            Acc: Ty[M // P] = 1
            with allo.meta_for(P) as i:
                pipe[i].put(Acc)

        @df.kernel(mapping=[P], args=[A, B])
        def core(local_A: Ty[M] @ Ly, local_B: Ty[M] @ Ly):
            pk = df.get_pid()
            local_B[:] = allo.add(local_A, pipe[pk].get())

    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)

    if is_available():
        mod_v1 = df.build(top, target="aie")
        mod_v1(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("V1 PASSED!")

        mod_v2 = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("bundle", ["core_0", "core_1", "core_2", "core_3"]),
            ],
        )
        mod_v2(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("V2 PASSED!")

        os.environ["FORCE_UNROLL_INDEX"] = "0"
        mod_v3 = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("bundle", ["core_0", "core_1", "core_2", "core_3"]),
                ("chain", ["prod_0", "core_0x4"]),
            ],
        )
        mod_v3(A, B)
        del os.environ["FORCE_UNROLL_INDEX"]
        np.testing.assert_allclose(B, A + 1)
        print("V3 PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_scatter2():
    Ty = int32
    M = 1024
    P = 4
    Ly = MemLayout("S0")

    @df.region()
    def top(Inc: Ty[M // P], A: Ty[M], B: Ty[M]):
        pipe: Stream[Ty[M // P], 2][P]

        @df.kernel(mapping=[1], args=[Inc])
        def prod(local_Inc: Ty[M // P] @ Ly):
            with allo.meta_for(P) as i:
                pipe[i].put(local_Inc)

        @df.kernel(mapping=[P], args=[A, B])
        def core(local_A: Ty[M] @ Ly, local_B: Ty[M] @ Ly):
            pk = df.get_pid()
            local_B[:] = allo.add(local_A, pipe[pk].get())

    Inc = np.ones(M // P).astype(np.int32)
    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)
    if is_available():
        mod_v1 = df.build(top, target="aie")
        mod_v1(Inc, A, B)
        np.testing.assert_allclose(B, A + 1)
        print("V1 PASSED!")

        mod_v2 = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("bundle", ["core_0", "core_1", "core_2", "core_3"]),
            ],
        )
        mod_v2(Inc, A, B)
        np.testing.assert_allclose(B, A + 1)
        print("V2 PASSED!")

        os.environ["FORCE_UNROLL_INDEX"] = "0"
        mod_v3 = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("bundle", ["core_0", "core_1", "core_2", "core_3"]),
                ("chain", ["prod_0", "core_0x4"]),
            ],
        )
        mod_v3(Inc, A, B)
        del os.environ["FORCE_UNROLL_INDEX"]
        np.testing.assert_allclose(B, A + 1)
        print("V3 PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_vector_scalar_add()
    test_gather()
    test_scatter()
    test_scatter2()
