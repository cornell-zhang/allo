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

batch_size = 1
hidden_size = 32


def softmax_approx_np(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x)
    x_sum = np.sum(x, axis=1, keepdims=True)
    return x / x_sum


def _test_softmax():

    # TODO: excessive error
    softmax = ExternalModule(
        top="softmax",
        impl_path="softmax.cc",
        input_idx=[0],
        output_idx=[1],
    )

    Ty = float32
    M, N = batch_size, hidden_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M, N] @ LyA, C: Ty[M, N] @ LyA):
            softmax(A, C)

    input_tensor = (
        torch.randn(batch_size, hidden_size, dtype=torch.float32).cpu().numpy()
    )
    output = softmax_approx_np(input_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        output_allo = np.zeros((batch_size, hidden_size)).astype(np.float32)
        mod(input_tensor, output_allo)
        print(output)
        np.testing.assert_allclose(output_allo, output, rtol=1e-3)
        print("PASSED!")


if __name__ == "__main__":
    _test_softmax()
