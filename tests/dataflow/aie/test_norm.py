# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import torch.nn as nn
from allo.ir.types import float32
import allo.dataflow as df
import numpy as np
from allo.memory import MemLayout
from allo.backend.aie.external_kernel import ExternalModule
from allo.backend.aie import is_available

Ly = MemLayout("R")
LyA = MemLayout("S0R")

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


@pytest.mark.parametrize("enable_trace", [False, True])
def test_layer_norm(enable_trace: bool):
    dir_path = os.path.dirname(os.path.abspath(__file__))

    norm = ExternalModule(
        top="layer_norm",
        impl_path=f"{dir_path}/norm.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    M, N = seq_len, hidden_size

    @df.region()
    def top(A: Ty[M, N], B: Ty[N], C: Ty[M, N]):
        @df.kernel(mapping=[4], args=[A, B, C])
        def core(local_A: Ty[M, N] @ LyA, local_B: Ty[N] @ Ly, local_C: Ty[M, N] @ LyA):
            norm(local_A, local_B, local_C)

    input_tensor = torch.randn(seq_len, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    output = layernorm(input_tensor, weight)

    if is_available():
        if enable_trace:
            mod = df.build(
                top,
                target="aie",
                trace=[("core", (0,)), ("core", (1,))],
                trace_size=65536,
            )
        else:
            mod = df.build(top, target="aie")
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


def test_rms_norm():
    dir_path = os.path.dirname(os.path.abspath(__file__))

    norm = ExternalModule(
        top="rms_norm",
        impl_path=f"{dir_path}/norm.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    M, N = seq_len, hidden_size

    @df.region()
    def top(A: Ty[M, N], B: Ty[N], C: Ty[M, N]):
        @df.kernel(mapping=[4], args=[A, B, C])
        def core(local_A: Ty[M, N] @ LyA, local_B: Ty[N] @ Ly, local_C: Ty[M, N] @ LyA):
            norm(local_A, local_B, local_C)

    input_tensor = torch.randn(seq_len, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    rms_norm = RMSNorm()
    output = rms_norm(input_tensor, weight)

    if is_available():
        mod = df.build(top, target="aie")
        output_allo = np.zeros((seq_len, hidden_size)).astype(np.float32)
        mod(input_tensor.cpu().numpy(), weight.cpu().numpy(), output_allo)
        np.testing.assert_allclose(output_allo, output, rtol=1e-2)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_single_row_layer_norm():
    dir_path = os.path.dirname(os.path.abspath(__file__))

    norm = ExternalModule(
        top="single_row_layer_norm",
        impl_path=f"{dir_path}/norm.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    M, N = 4, 512

    @df.region()
    def top(A: Ty[M, N], B: Ty[N], C: Ty[M, N]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def core(local_A: Ty[M, N], local_B: Ty[N], local_C: Ty[M, N]):
            for i in range(M):
                # [NOTE]: test using buffer slice as customized external kernel arguments
                norm(local_A[i], local_B, local_C[i])

    input_tensor = torch.randn(M, N, dtype=torch.float32)
    weight = torch.randn(N, dtype=torch.float32)
    output = layernorm(input_tensor, weight)

    if is_available():
        mod = df.build(top, target="aie")
        output_allo = np.zeros((M, N)).astype(np.float32)
        mod(input_tensor.cpu().numpy(), weight.cpu().numpy(), output_allo)
        np.testing.assert_allclose(output_allo, output, rtol=1e-2)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_layer_norm(enable_trace=False)
    test_rms_norm()
    test_single_row_layer_norm()
    test_layer_norm(enable_trace=True)
