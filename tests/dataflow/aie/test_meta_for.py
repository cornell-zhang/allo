# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout


def _test_vector_scalar_add():
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M]):
            # [NOTE]: this will be optimized as `for i in range(M):` (though such usage is not recommended)
            with allo.meta_for(M) as i:
                B[i] = A[i] + 1

    A = np.random.randint(0, 100, M).astype(np.int32)
    mod = df.build(top, target="aie")
    B = np.zeros(M).astype(np.int32)
    mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")


def _test_gather():

    Ty = int16
    M, N, K = 32, 32, 64
    Pk = 2

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
            with allo.meta_for(Pk) as i:
                C_[:, :] += pipe[i].get()
            C[:, :] = C_

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod_v1 = df.build(top, target="aie")
    mod_v1(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")

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
    print("PASSED!")


def _test_scatter():
    Ty = int32
    M = 1024
    P = 4
    Ly = Layout("S0")

    @df.region()
    def top():
        pipe = df.array(df.pipe(dtype=Ty, shape=(M // P,), depth=2), shape=(P,))

        @df.kernel(mapping=[1])
        def prod():
            Acc: Ty[M // P] = 1
            with allo.meta_for(P) as i:
                pipe[i].put(Acc)

        @df.kernel(mapping=[P])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly):
            pk = df.get_pid()
            B[:] = allo.add(A, pipe[pk].get())

    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)

    mod_v1 = df.build(top, target="aie")
    mod_v1(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")

    mod_v2 = df.build(
        top,
        target="aie",
        mapping_primitives=[
            ("bundle", ["core_0", "core_1", "core_2", "core_3"]),
        ],
    )
    mod_v2(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")

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
    print("PASSED!")


def _test_scatter2():
    Ty = int32
    M = 1024
    P = 4
    Ly = Layout("S0")

    @df.region()
    def top():
        pipe = df.array(df.pipe(dtype=Ty, shape=(M // P,), depth=2), shape=(P,))

        @df.kernel(mapping=[1])
        def prod(Inc: Ty[M // P] @ Ly):
            with allo.meta_for(P) as i:
                pipe[i].put(Inc)

        @df.kernel(mapping=[P])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly):
            pk = df.get_pid()
            B[:] = allo.add(A, pipe[pk].get())

    Inc = np.ones(M // P).astype(np.int32)
    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)

    mod_v1 = df.build(top, target="aie")
    mod_v1(Inc, A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")

    mod_v2 = df.build(
        top,
        target="aie",
        mapping_primitives=[
            ("bundle", ["core_0", "core_1", "core_2", "core_3"]),
        ],
    )
    mod_v2(Inc, A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")

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
    print("PASSED!")


if __name__ == "__main__":
    _test_vector_scalar_add()
    _test_gather()
    _test_scatter()
    _test_scatter2()
