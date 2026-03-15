# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, Stream
from allo import spmw
import allo.backend.hls as hls
from src.hls import to_hls
import numpy as np
import tempfile


def test_producer_consumer():
    Ty = float32
    M, N, K = 16, 16, 16

    @spmw.unit()
    def top(A: Ty[M, N], B: Ty[M, N]):
        pipe: Stream[Ty, 4]

        @spmw.work(grid=[1])
        def producer():
            for i, j in allo.grid(M, N):
                # load data
                out: Ty = A[i, j]
                # send data
                pipe.put(out)

        @spmw.work(grid=[1])
        def consumer():
            for i, j in allo.grid(M, N):
                # receive data
                data = pipe.get()
                # computation
                B[i, j] = data + 1

    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)

    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = to_hls(top, project=tmpdir)
            mod(A, B)
            np.testing.assert_allclose(A + 1, B)
            print("Passed!")


if __name__ == "__main__":
    test_producer_consumer()
