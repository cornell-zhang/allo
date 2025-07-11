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

seq_len = 16
hidden_size = 512


def layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: shape [..., dim] - input tensor
    weight: shape [dim] - scale parameter γ
    eps: small constant for numerical stability
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * weight


def _test_layer_norm():

    norm = ExternalModule(
        top="layer_norm",
        impl_path="norm.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    M, N = seq_len, hidden_size

    @df.region()
    def top():
        @df.kernel(mapping=[4])
        def core(A: Ty[M, N] @ LyA, B: Ty[N] @ Ly, C: Ty[M, N] @ LyA):
            norm(A, B, C)

    input_tensor = torch.randn(seq_len, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    output = layernorm(input_tensor, weight)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        output_allo = np.zeros((seq_len, hidden_size)).astype(np.float32)
        mod(input_tensor.cpu().numpy(), weight.cpu().numpy(), output_allo)
        np.testing.assert_allclose(output_allo, output, rtol=1e-2)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, weight):
        norm = x.norm(dim=-1, keepdim=True)  # L2 norm along last dim
        rms = norm / (x.shape[-1] ** 0.5)
        return x / (rms + self.eps) * weight


def _test_rms_norm():

    norm = ExternalModule(
        top="rms_norm",
        impl_path="norm.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    M, N = seq_len, hidden_size

    @df.region()
    def top():
        @df.kernel(mapping=[4])
        def core(A: Ty[M, N] @ LyA, B: Ty[N] @ Ly, C: Ty[M, N] @ LyA):
            norm(A, B, C)

    input_tensor = torch.randn(seq_len, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    rms_norm = RMSNorm()
    output = rms_norm(input_tensor, weight)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        output_allo = np.zeros((seq_len, hidden_size)).astype(np.float32)
        mod(input_tensor.cpu().numpy(), weight.cpu().numpy(), output_allo)
        np.testing.assert_allclose(output_allo, output, rtol=1e-2)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    _test_layer_norm()
    _test_rms_norm()
