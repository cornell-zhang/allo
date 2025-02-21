# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import UInt
import allo.dataflow as df


def test_uint():
    B, M, N = 2, 2, 2

    @df.region()
    def top():
        stream = df.pipe(dtype=UInt(B * 8), shape=(), depth=4)

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


if __name__ == "__main__":
    test_uint()
