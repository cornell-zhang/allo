# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

Ty = float32
M, N, K = 16, 16, 16


@df.region()
def top():
    pipe0 = df.pipe(dtype=Ty, shape=(), depth=4)
    pipe1 = df.pipe(dtype=Ty, shape=(), depth=2)

    @df.kernel(mapping=[1])
    def producer(A: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # load data
            out: Ty = A[i, j]
            # send data
            pipe0.put(out)

    @df.kernel(mapping=[1])
    def consumer(B: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # receive data
            data = pipe0.get()
            # computation
            B[i, j] = data + 1


if __name__ == "__main__":
    mod = df.build(top)
