# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import allo
import numpy as np


_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from gemm_ncubed import gemm
import gemm_blocked


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


def test_gemm_blocked(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["gemm_blocked"][psize]

    # Patch module constants before customize()
    for key, val in params.items():
        setattr(gemm_blocked, key, val)

    M = gemm_blocked.M
    N = gemm_blocked.N
    K = gemm_blocked.K

    np.random.seed(42)
    m1 = np.random.randint(0, 10, (M, K)).astype(np.int32)
    m2 = np.random.randint(0, 10, (K, N)).astype(np.int32)

    s = allo.customize(gemm_blocked.bbgemm)
    mod = s.build(target="llvm")

    actual = mod(m1, m2)
    check = np.matmul(m1.astype(np.int64), m2.astype(np.int64)).astype(np.int32)
    np.testing.assert_allclose(actual, check, rtol=1e-5, atol=1e-5)
    print("PASS!")


if __name__ == "__main__":
    test_gemm_ncubed("full")
    test_gemm_blocked("full")
