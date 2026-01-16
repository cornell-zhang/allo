# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df
import numpy as np
from allo.backend.aie import is_available

Ty = int32
M, N, K = 16, 16, 16


@df.region()
def top(A: Ty[M, N], B: Ty[M, N]):
    pipe: Stream[Ty, 4]

    @df.kernel(mapping=[1], args=[A])
    def producer(local_A: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # load data
            out: Ty = local_A[i, j]
            # send data
            pipe.put(out)

    @df.kernel(mapping=[1], args=[B])
    def consumer(local_B: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # receive data
            data = pipe.get()
            # computation
            local_B[i, j] = data + 1


def test_producer_consumer():
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.zeros((M, N), dtype=np.int32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("Dataflow Simulator Passed!")

    if is_available():
        mod = df.build(top, target="aie")
        mod(A, B)
        np.testing.assert_allclose(A + 1, B, atol=1e-5)
        print("Passed!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_producer_consumer()
