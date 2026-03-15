# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tempfile
from src.hls import to_hls
import allo.backend.hls as hls
from allo.ir.types import int32, ConstExpr, index
from allo import spmw


def test_get_wid_1D_1():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            for i in range(1024):
                B[i] = A[i] + 1

    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = to_hls(top, project=tmpdir)
            A = np.random.rand(
                1024,
            ).astype(np.int32)
            B = np.zeros((1024,), dtype=np.int32)
            mod(A, B)
            np.testing.assert_allclose(A + 1, B)
            print("Passed!")


def test_get_wid_1D_2():
    vlen = 1024
    P = 4
    tlen = vlen // P

    @spmw.unit()
    def top(A: int32[vlen], B: int32[vlen]):
        @spmw.work(grid=[P])
        def core():
            x = spmw.axes()
            pi: ConstExpr[index] = x.id
            for i in range(tlen * pi, tlen * (pi + 1)):
                B[i] = A[i] + 1

    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = to_hls(top, project=tmpdir)
            A = np.random.rand(
                1024,
            ).astype(np.int32)
            B = np.zeros((1024,), dtype=np.int32)
            mod(A, B)
            np.testing.assert_allclose(A + 1, B)
            print("Passed!")


if __name__ == "__main__":
    test_get_wid_1D_1()
    test_get_wid_1D_2()
