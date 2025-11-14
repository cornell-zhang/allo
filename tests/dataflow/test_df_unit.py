# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import UInt, float32, index, Stream
import allo.dataflow as df
import numpy as np


def test_uint():
    B, M, N = 2, 2, 2

    @df.region()
    def top():
        stream: Stream[UInt(B * 8), 4]

        @df.kernel(mapping=[1])
        def load(A: UInt(B * 8)[M, N]):
            for mt, nt in allo.grid(M, N):
                stream.put(A[mt, nt])

        @df.kernel(mapping=[1])
        def store(B: UInt(B * 8)[M, N]):
            for mt, nt in allo.grid(M, N):
                B[mt, nt] = stream.get()

    mod = df.build(top, target="vitis_hls", project="top.prj", wrap_io=False)
    print(mod.hls_code)
    assert "hls::stream< uint16_t >" in mod.hls_code
    assert "hls::stream< int16_t >" not in mod.hls_code
    assert " int16_t" not in mod.hls_code


def test_func_index():
    Ty = float32
    M, N = 4, 4

    def index_calculation(x: index) -> index:
        res: index = x - 1
        return res

    @df.region()
    def top():
        pipe: Stream[Ty, 4]

        @df.kernel(mapping=[1])
        def producer(A: Ty[M, N]):
            for i, j in allo.grid(M, N):
                out: Ty = A[i, index_calculation(j) + 1]
                pipe.put(out)

        @df.kernel(mapping=[1])
        def consumer(B: Ty[M, N]):
            for i, j in allo.grid(M, N):
                data = pipe.get()
                B[i, j] = data + 1

    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)
    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B)
    np.testing.assert_allclose(A + 1, B)
    print("Dataflow Simulator Passed!")


if __name__ == "__main__":
    test_uint()
    test_func_index()
