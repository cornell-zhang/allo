# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

Ty = float32
M, N, K = 16, 16, 16


@df.kernel(mapping=[1])
def producer(A: Ty[M, N]):
    pipe: Stream[Ty] = df.pipe(src="producer", dst="consumer")
    for i, j in allo.grid(M, N):
        # load data
        out: Ty = A[i, j]
        # send data
        pipe.put(out)


@df.kernel(mapping=[1])
def consumer(B: Ty[M, N]):
    pipe: Stream[Ty] = df.pipe(src="producer", dst="consumer")
    for i, j in allo.grid(M, N):
        # receive data
        data = pipe.get()
        # computation
        B[i, j] = data + 1


def test_producer_consumer():
    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)
    top = df.build([producer, consumer])
    if hls.is_available("vitis_hls"):
        top(A, B)
        np.testing.assert_allclose(A + 1, B)
        print("Passed!")


if __name__ == "__main__":
    test_producer_consumer()
