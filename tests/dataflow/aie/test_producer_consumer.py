# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int32
import allo.dataflow as df
import numpy as np

Ty = int32
M, N, K = 16, 16, 16


@df.region()
def top():
    pipe = df.pipe(dtype=Ty, shape=(), depth=4)

    @df.kernel(mapping=[1])
    def producer(A: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # load data
            out: Ty = A[i, j]
            # send data
            pipe.put(out)

    @df.kernel(mapping=[1])
    def consumer(B: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # receive data
            data = pipe.get()
            # computation
            B[i, j] = data + 1


def test_producer_consumer():
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.zeros((M, N), dtype=np.int32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("Dataflow Simulator Passed!")

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        mod(A, B)
        np.testing.assert_allclose(A + 1, B, atol=1e-5)
        print("Passed!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_producer_consumer()
