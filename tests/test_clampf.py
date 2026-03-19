# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import float32
import pytest

torch = pytest.importorskip("torch")
def clampf_kernel(x: float32[4, 4]) -> float32[4, 4]:
    return clampf(x, -1.0, 1.0)


def test_clampf_mlir_and_execution():
    s = allo.customize(clampf_kernel)

    mlir_text = str(s.module)
    assert "math.clampf" in mlir_text

    mod = s.build()
    x = np.array(
        [
            [-2.0, -0.5, 0.2, 1.5],
            [3.0, -4.0, 0.0, 0.9],
            [1.2, -1.1, 5.0, -3.3],
            [0.7, -0.8, 1.0, -1.0],
        ],
        dtype=np.float32,
    )
    out = mod(x)
    expected = np.clip(x, -1.0, 1.0)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
