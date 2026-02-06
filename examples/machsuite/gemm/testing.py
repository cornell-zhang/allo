# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import allo
import numpy as np


_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from gemm_ncubed import gemm


def test_gemm_ncubed(psize="small"):
    # gemm_ncubed is hardcoded to 64x64, no size parameters to patch
    N = 64
    np.random.seed(42)
    m1 = np.random.randint(0, 100, (N, N)).astype(np.float32)
    m2 = np.random.randint(0, 100, (N, N)).astype(np.float32)
    s = allo.customize(gemm)

    mod = s.build(target="llvm")

    actual = mod(m1, m2)
    check = np.matmul(m1, m2)
    np.testing.assert_allclose(actual, check, rtol=1e-5, atol=1e-5)
    print("PASS!")


if __name__ == "__main__":
    test_gemm_ncubed("full")
