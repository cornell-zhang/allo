# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from allo.ir.types import int4, int8
import allo.dataflow as df
from allo.library.aie.modules.gemm import GEMM
import numpy as np
from allo.backend.aie import is_available


@pytest.mark.parametrize(
    "M, N, K, Pm, Pn, Pk, TyI, TyO",
    [
        (1024, 1024, 1024, 1024 // 64, 1024 // 64, 1024 // 64, int8, int8),
    ],
)
def test_pingpong_gemm(M, N, K, Pm, Pn, Pk, TyI, TyO):
    top, mapping_primitives = GEMM(M, N, K, Pm, Pn, Pk, TyI, TyO)
    assert (TyI is int4 or TyI is int8) and TyO is int8, (
        "This test only supports these data type combinations. "
        "Please refer to examples/aie/gemm.py for gemm examples with other data types."
    )

    if is_available():
        os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
        mod = df.build(
            top,
            project="top.prj",
            target="aie",
            mapping_primitives=mapping_primitives,
            profile=True,
            warmup=200,
            num_iters=1000,
        )

        A = np.random.randint(-4, 4, (M, K)).astype(np.int8)
        B = np.random.randint(-4, 4, (K, N)).astype(np.int8)
        C = np.zeros((M, N)).astype(np.int8)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
        del os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"]
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024
    m, n, k = 64, 128, 64
    # - i8
    test_pingpong_gemm(M, N, K, M // m, N // n, K // k, int8, int8)
    # - i4
    dir_path = os.path.dirname(os.path.abspath(__file__))
    os.environ["ALLO_EXTERNAL_KERNEL_DIR"] = (
        f"{dir_path}/../../../allo/library/aie/kernels/"
    )
    test_pingpong_gemm(M, N, K, M // m, N // n, K // k, int4, int8)
    del os.environ["ALLO_EXTERNAL_KERNEL_DIR"]
