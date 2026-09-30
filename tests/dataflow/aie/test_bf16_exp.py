# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from allo.ir.types import bfloat16
from ml_dtypes import bfloat16 as np_bfloat16
import allo.dataflow as df
import numpy as np
from allo.backend.aie.external_kernel import ExternalModule
from allo.backend.aie import is_available


def test_exp():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.abspath(
        os.path.join(
            dir_path, "../../../allo/library/aie/kernels/softmax_bf16_aie2p.cc"
        )
    )
    exp_ = ExternalModule(
        top="exp_bf16",
        impl_path=kernel_path,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = bfloat16
    LEN = 1024

    @df.region()
    def top(A: Ty[LEN], B: Ty[LEN]):
        @df.kernel(mapping=[1], args=[A, B])
        def core(local_A: Ty[LEN], local_B: Ty[LEN]):
            exp_(local_A, local_B)

    A = (np.random.rand(LEN).astype(np.float32) * 4.0 - 2.0).astype(np_bfloat16)
    ref = np.exp(A.astype(np.float32)).astype(np_bfloat16)

    if os.getenv("NPU2") == "1" and is_available():
        mod = df.build(top, target="aie")
        B = np.zeros(LEN).astype(np_bfloat16)
        mod(A, B)
        np.testing.assert_allclose(
            B.astype(np.float32), ref.astype(np.float32), rtol=8e-2
        )
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_softmax():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    if os.getenv("NPU2") == "1":
        kernel_path = os.path.abspath(
            os.path.join(
                dir_path, "../../../allo/library/aie/kernels/softmax_bf16_aie2p.cc"
            )
        )
    else:
        kernel_path = os.path.abspath(
            os.path.join(dir_path, "../../../allo/library/aie/kernels/softmax_bf16.cc")
        )

    softmax_ = ExternalModule(
        top="vector_softmax_bf16",
        impl_path=kernel_path,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = bfloat16
    LEN = 1024

    @df.region()
    def top(A: Ty[LEN], B: Ty[LEN]):
        @df.kernel(mapping=[1], args=[A, B])
        def core(local_A: Ty[LEN], local_B: Ty[LEN]):
            softmax_(local_A, local_B)

    # safe softmax
    A = (np.random.rand(LEN).astype(np.float32) * 4.0 - 2.0).astype(np_bfloat16)
    A_f32 = A.astype(np.float32)
    shifted = np.exp(A_f32 - A_f32.max())
    ref = (shifted / shifted.sum()).astype(np_bfloat16)

    if os.getenv("NPU2") == "1" and is_available():
        mod = df.build(top, target="aie")
        B = np.zeros(LEN).astype(np_bfloat16)
        mod(A, B)
        np.testing.assert_allclose(
            B.astype(np.float32), ref.astype(np.float32), rtol=8e-2, atol=1e-4
        )
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_exp()
    test_softmax()
