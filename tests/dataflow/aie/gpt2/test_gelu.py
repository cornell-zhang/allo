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

Ly = Layout("R")
LyA = Layout("S0R")

batch_size = 16
hidden_size = 512


def gelu_approx_np(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def _test_gelu():

    # TODO: excessive error
    gelu = ExternalModule(
        top="gelu",
        impl_path="gelu.cc",
        input_idx=[0],
        output_idx=[1],
    )

    Ty = float32
    M, N = batch_size, hidden_size

    @df.region()
    def top():
        @df.kernel(mapping=[4])
        def core(A: Ty[M, N] @ LyA, C: Ty[M, N] @ LyA):
            gelu(A, C)

    input_tensor = torch.randn(batch_size, hidden_size, dtype=torch.float32)
    output = gelu_approx_np(input_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        output_allo = np.zeros((batch_size, hidden_size)).astype(np.float32)
        mod(input_tensor.cpu().numpy(), output_allo)
        np.testing.assert_allclose(output_allo, output, rtol=1e-1)
        print("PASSED!")


if __name__ == "__main__":
    _test_gelu()
