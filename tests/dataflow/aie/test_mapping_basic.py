# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import Stream, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie import is_available

S = Layout.Shard
R = Layout.Replicate


def test_vector_scalar_add_v1():
    Ly = [S(0)]
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top(A: Ty[M], B: Ty[M]):
        @df.kernel(mapping=[4], args=[A, B])
        def core(local_A: Ty[M] @ Ly, local_B: Ty[M] @ Ly):
            local_B[:] = allo.add(local_A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)
    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[("bundle", ["core_0", "core_1", "core_2", "core_3"])],
        )
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_vector_scalar_add_v2():
    Ly = [S(0)]
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top(A: Ty[M], B: Ty[M]):
        @df.kernel(mapping=[4], args=[A, B])
        def core(local_A: Ty[M] @ Ly, local_B: Ty[M] @ Ly):
            local_B[:] = allo.add(local_A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)
    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("bundle", ["core_0", "core_1"]),
                ("bundle", ["core_2", "core_3"]),
            ],
        )
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_vector_scalar_add_v3():
    Ly = [S(0)]
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top(A: Ty[M], B: Ty[M]):
        @df.kernel(mapping=[4], args=[A, B])
        def core(local_A: Ty[M] @ Ly, local_B: Ty[M] @ Ly):
            local_B[:] = allo.add(local_A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)
    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("bundle", ["core_0", "core_1"]),  # -> bundled_node_name: core_0
                ("bundle", ["core_2", "core_3"]),  # -> bundled_node_name: core_2
                (
                    "bundle",
                    ["core_0x2", "core_2x2"],
                ),  # name after bundled, may be confusing
            ],
        )
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_producer_consumer():
    Ty = int32
    M, N, K = 16, 16, 16

    @df.region()
    def top(A: Ty[M, N], B: Ty[M, N]):
        pipe: Stream[Ty[M, N], 1]

        @df.kernel(mapping=[1], args=[A])
        def producer(local_A: Ty[M, N]):
            pipe.put(local_A)

        @df.kernel(mapping=[1], args=[B])
        def consumer(local_B: Ty[M, N]):
            data = pipe.get()
            for i, j in allo.grid(M, N):
                # computation
                local_B[i, j] = data[i, j] + 1

    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.zeros((M, N), dtype=np.int32)
    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[("chain", ["producer_0", "consumer_0"])],
        )
        mod(A, B)
        np.testing.assert_allclose(A + 1, B, atol=1e-5)
        print("Passed!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_vector_scalar_add_v1()
    test_vector_scalar_add_v2()
    test_vector_scalar_add_v3()
    test_producer_consumer()
