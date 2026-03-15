# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo.backend.hls as hls
from src.hls import to_hls
import tempfile
import allo
from allo.ir.types import int32
from allo import spmw


def test():
    @spmw.unit()
    def vadd(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            for i in allo.grid(1024):
                B[i] = A[i] + 1

    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            vadd(A, B)

    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = np.random.randint(0, 100, (1024,), dtype=np.int32)
            B = np.random.randint(0, 100, (1024,), dtype=np.int32)
            mod = to_hls(top, project=tmpdir)
            mod(A, B)
            np.testing.assert_allclose(A + 1, B)


if __name__ == "__main__":
    test()
