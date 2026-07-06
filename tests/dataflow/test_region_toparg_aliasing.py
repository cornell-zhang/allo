# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for cornell-zhang/allo#592.

When a ``@df.region`` contains multiple ``@df.kernel``s that reuse the same
local parameter name while being bound to distinct region inputs, the top-level
function interface must expose one memref per distinct region input. The old
code deduplicated top-level arguments by the kernel-local parameter name, so two
kernels both naming their input ``x`` collapsed their distinct region inputs
(``a`` and ``b``) into a single top-level buffer, raising::

    AssertionError: # of input arguments mismatch, got 2 but expected 1
"""

import allo.dataflow as df
from allo.ir.types import int32
import numpy as np


def test_region_toparg_aliasing():
    @df.region()
    def top(a: int32[4], b: int32[4]):  # two distinct region inputs
        @df.kernel(mapping=[1], args=[a])
        def k_a(x: int32[4]):  # local param name "x"
            for i in range(4):
                x[i] = 1

        @df.kernel(mapping=[1], args=[b])
        def k_b(x: int32[4]):  # same local param name "x" -> aliasing bug
            for i in range(4):
                x[i] = 2

    mod = df.build(top, target="simulator")

    a = np.zeros(4, dtype=np.int32)
    b = np.zeros(4, dtype=np.int32)

    # The region has two inputs, so two arrays are passed. The aliasing bug
    # collapsed them into a single top-level argument, so this call raised
    # "got 2 but expected 1" before the fix.
    mod(a, b)

    np.testing.assert_array_equal(a, np.ones(4, dtype=np.int32))
    np.testing.assert_array_equal(b, np.full(4, 2, dtype=np.int32))


if __name__ == "__main__":
    test_region_toparg_aliasing()
    print("PASSED: test_region_toparg_aliasing")
