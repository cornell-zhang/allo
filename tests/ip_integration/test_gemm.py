# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import allo


def test_gemm():
    mod = allo.backend.ip.load_hls(
        top="gemm",
        headers=["gemm.h"],
        impls=["gemm.cpp"],
        signature="float32[16, 16], float32[16, 16], float32[16, 16]",
        link_hls=False,
    )
    a = np.random.random((16, 16)).astype(np.float32)
    b = np.random.random((16, 16)).astype(np.float32)
    c = np.zeros((16, 16)).astype(np.float32)
    mod(a, b, c)
    np.testing.assert_allclose(np.matmul(a, b), c, atol=1e-6)
    print("Passed!")


if __name__ == "__main__":
    pytest.main([__file__])
