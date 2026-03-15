# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.hls import to_hls
import allo
from allo.ir.types import int32, Stream
import allo.backend.hls as hls
from allo import spmw
import tempfile


def test_scalar_stream_1():
    @spmw.unit()
    def top1(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32]

        @spmw.work(grid=[1])
        def producer():
            pipe.put(A[0, 0])

        @spmw.work(grid=[1])
        def consumer():
            B[0, 0] = pipe.get()

    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = to_hls(top1, project=tmpdir)
            np_A = np.random.randint(0, 100, (16, 16), dtype=np.int32)
            np_B = np.zeros((16, 16), dtype=np.int32)
            s(np_A, np_B)
            assert np_A[0][0] == np_B[0][0]


def test_scalar_stream_2():
    @spmw.unit()
    def top2(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32]

        @spmw.work(grid=[1])
        def producer():
            for i, j in allo.grid(16, 16):
                pipe.put(A[i, j])

        @spmw.work(grid=[1])
        def consumer():
            for i, j in allo.grid(16, 16):
                B[i, j] = pipe.get()

    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = to_hls(top2, project=tmpdir)
            np_A = np.random.randint(0, 100, (16, 16), dtype=np.int32)
            np_B = np.zeros((16, 16), dtype=np.int32)
            s(np_A, np_B)
            assert np.array_equal(np_A, np_B)


def test_tensor_stream():
    @spmw.unit()
    def top(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32][16, 16]

        @spmw.work(grid=[1])
        def producer():
            with allo.meta_for(16) as i:
                with allo.meta_for(16) as j:
                    pipe[i, j].put(A[i, j])

        @spmw.work(grid=[1])
        def consumer():
            with allo.meta_for(16) as i:
                with allo.meta_for(16) as j:
                    B[i, j] = pipe[i, j].get()

    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = to_hls(top, project=tmpdir)
            np_A = np.random.randint(0, 100, (16, 16), dtype=np.int32)
            np_B = np.zeros((16, 16), dtype=np.int32)
            s(np_A, np_B)
            assert np.array_equal(np_A, np_B)


if __name__ == "__main__":
    test_scalar_stream_1()
    test_scalar_stream_2()
    test_tensor_stream()
