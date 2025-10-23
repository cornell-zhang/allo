# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import Stream, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout


def _test_vector_scalar_add_v1():
    Ly = Layout("S0")
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[4])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    mod = df.build(
        top,
        target="aie",
        mapping_primitives=[("bundle", ["core_0", "core_1", "core_2", "core_3"])],
    )
    B = np.zeros(M).astype(np.int32)
    mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")


def _test_vector_scalar_add_v2():
    Ly = Layout("S0")
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[4])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    mod = df.build(
        top,
        target="aie",
        mapping_primitives=[
            ("bundle", ["core_0", "core_1"]),
            ("bundle", ["core_2", "core_3"]),
        ],
    )
    B = np.zeros(M).astype(np.int32)
    mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")


def _test_vector_scalar_add_v3():
    Ly = Layout("S0")
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[4])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
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
    B = np.zeros(M).astype(np.int32)
    mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")


def _test_producer_consumer():
    Ty = int32
    M, N, K = 16, 16, 16

    @df.region()
    def top():
        pipe: Stream[Ty[M, N], 1]

        @df.kernel(mapping=[1])
        def producer(A: Ty[M, N]):
            pipe.put(A)

        @df.kernel(mapping=[1])
        def consumer(B: Ty[M, N]):
            data = pipe.get()
            for i, j in allo.grid(M, N):
                # computation
                B[i, j] = data[i, j] + 1

    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.zeros((M, N), dtype=np.int32)

    mod = df.build(
        top,
        target="aie",
        mapping_primitives=[("chain", ["producer_0", "consumer_0"])],
    )
    mod(A, B)
    np.testing.assert_allclose(A + 1, B, atol=1e-5)
    print("Passed!")


if __name__ == "__main__":
    _test_vector_scalar_add_v1()
    _test_vector_scalar_add_v2()
    _test_vector_scalar_add_v3()
    _test_producer_consumer()
