# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import float32


def clampf_kernel(x: float32[4, 4]) -> float32[4, 4]:
    return clampf(x, -1.0, 1.0)


def test_clampf_mlir_and_execution():
    s = allo.customize(clampf_kernel)

    mlir_text = str(s.module)
    assert "arith.maximumf" in mlir_text
    assert "arith.minimumf" in mlir_text
    assert "math.clampf" not in mlir_text

    mod = s.build(target="llvm")

    x = np.array(
        [
            [-100.0, -1.0, -0.25, 0.0],
            [0.25, 0.75, 1.0, 100.0],
            [-2.5, -1.5, 1.5, 2.5],
            [9.0, -9.0, 0.5, -0.5],
        ],
        dtype=np.float32,
    )

    out = np.asarray(mod(x))
    golden = np.clip(x, -1.0, 1.0)

    assert out.shape == golden.shape
    np.testing.assert_allclose(
        out,
        golden,
        rtol=1e-6,
        atol=1e-6,
        err_msg="clampf lowering/execution does not match numpy.clip",
    )
