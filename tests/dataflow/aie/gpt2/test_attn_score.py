# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn as nn
from allo.ir.types import float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.experimental.external_kernel import ExternalModule
from allo.ir.types import float32

KERNEL_LIB_PATH = "../../../../allo/backend/experimental/kernels/"

LyA = Layout("S0R")
LyB = Layout("S1R")
LyC = Layout("S0S1")

head_dim = 64
seq_len = 32 * 4


def _test_attn_score():
    """
    Test the computation of attention scores using the external module
    `transpose_matmul_with_scale`. This module performs the core operation
    used in GPT-2's attention:

        attn_score = (Q @ K^T) / sqrt(d)

    where Q is the query matrix, K is the key matrix, and d is the head dimension.
    """

    attn_score = ExternalModule(
        top="transpose_matmul_with_scale",
        impl_path=KERNEL_LIB_PATH + "transpose_matmul_with_scale.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32

    @df.region()
    def top():
        @df.kernel(mapping=[4, 4])
        def core(
            A: Ty[seq_len, head_dim] @ LyA,
            B: Ty[seq_len, head_dim] @ LyB,
            C: Ty[seq_len, seq_len] @ LyC,
        ):
            attn_score(A, B, C)

    input_tensor_a = torch.randn(seq_len, head_dim, dtype=torch.float32)
    input_tensor_b = torch.randn(seq_len, head_dim, dtype=torch.float32)
    output = (input_tensor_a @ input_tensor_b.T) * 0.125
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        output_allo = np.zeros((seq_len, seq_len)).astype(np.float32)
        mod(input_tensor_a.cpu().numpy(), input_tensor_b.cpu().numpy(), output_allo)
        np.testing.assert_allclose(output_allo, output, rtol=1e-2)
        print("PASSED!")


if __name__ == "__main__":
    _test_attn_score()
