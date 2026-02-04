# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


def test_producer_consumer():
    Ty = float32
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

    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("Dataflow Simulator Passed!")


def test_double_put():
    Ty = float32
    M, N, K = 16, 16, 16

    @df.region()
    def top(A: Ty[M, N], B: Ty[M, N]):
        pipe: Stream[Ty, 4]

        @df.kernel(mapping=[1], args=[A])
        def producer(local_A: Ty[M, N]):
            for i, j in allo.grid(M, N):
                if j % 2 == 0:
                    # double put
                    pipe.put(local_A[i, j])
                    pipe.put(local_A[i, j + 1])
                else:
                    pass

        @df.kernel(mapping=[1], args=[B])
        def consumer(local_B: Ty[M, N]):
            for i, j in allo.grid(M, N):
                # normal get
                data = pipe.get()
                local_B[i, j] = data + 1

    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("Dataflow Simulator Passed!")

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = df.build(top, target="vitis_hls", project=tmpdir)
        assert "v32, v32" not in mod.hls_code


if __name__ == "__main__":
    test_producer_consumer()
    test_double_put()
