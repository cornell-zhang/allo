# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import uint256, uint32, uint8, bool, int8, int16, int32
import allo.dataflow as df
from allo.utils import get_np_struct_type
from allo.backend import hls

VLEN = 256
ELEN = 32

import numpy as np

np_256 = get_np_struct_type(VLEN)


def test_vadd():
    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def VEC(
            A: uint256[1],
            B: uint256[1],
            C: uint256[1],
        ):
            for i in allo.grid(VLEN // ELEN, name="vec_nest"):
                C[0][i * ELEN : (i + 1) * ELEN] = (
                    A[0][i * ELEN : (i + 1) * ELEN] + B[0][i * ELEN : (i + 1) * ELEN]
                )

    A = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    B = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    C = np.zeros(VLEN // ELEN).astype(np.uint32)
    packed_A = np.ascontiguousarray(A).view(np_256)
    packed_B = np.ascontiguousarray(B).view(np_256)
    packed_C = np.ascontiguousarray(C).view(np_256)

    mod = df.build(top, target="simulator")
    mod(packed_A, packed_B, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(A + B, unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED!")

    s = df.customize(top)
    # unroll the lanes
    nest_loop_i = s.get_loops("VEC_0")["vec_nest"]["i"]
    s.unroll(nest_loop_i)
    print(s.module)

    if hls.is_available("vitis_hls"):
        print("Starting Test...")
        mod = s.build(
            target="vitis_hls",
            mode="hw_emu",
            project=f"vadd.prj",
            wrap_io=False,
        )
        mod(packed_A, packed_B, packed_C)
        unpacked_C = packed_C.view(np.uint32)
        np.testing.assert_allclose(A + B, unpacked_C, rtol=1e-5, atol=1e-5)
        print(unpacked_C)
        print("Passed Test!")


if __name__ == "__main__":
    test_vadd()
