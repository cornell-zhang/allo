# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import allo
from allo.ir.types import int32


def test_get_bit():
    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] + 1)[0]
        return B

    s = allo.customize(kernel, verbose=True)
    print(s.module)
    np_A = np.random.randint(10, size=(10,))
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), (np_A + 1) & 1, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_get_bit()
