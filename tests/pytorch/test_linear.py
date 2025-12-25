# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import pytest
import allo
from allo.library.systolic import systolic
from allo.ir.types import int8
import numpy as np
import allo.backend.hls as hls

device = "cpu"
N, L, D = 1, 4, 4


def test_int8_linear():
    try:
        import torch
        from torch import nn
    except ImportError:
        print("PyTorch not found, skipping...")
        return

    class TestModule(nn.Module):
        def __init__(self, hidden_size: int, bias: bool = True):
            super().__init__()
            self.fc = nn.Linear(hidden_size, 4 * hidden_size, bias=bias, device=device)
            self.fc.weight = nn.Parameter(
                torch.randint(
                    -10,
                    10,
                    (hidden_size * 4, hidden_size),
                    device=device,
                    dtype=torch.float32,
                )
            )
            if bias:
                self.fc.bias = nn.Parameter(
                    torch.randint(
                        -10, 10, (hidden_size * 4,), device=device, dtype=torch.float32
                    )
                )

        def forward(self, hidden_states: torch.Tensor):
            return self.fc(hidden_states)

    model = TestModule(D, bias=False)
    model.eval()
    x = torch.randint(-10, 10, (L, D), device=device, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)

    def linear(X: int8[L, D], W: int8[D, 4 * D]) -> int8[L, 4 * D]:
        Z: int8[L, 4 * D]
        systolic[int8, int8, int8, L, D, 4 * D, 2, 2](X, W, Z)
        return Z

    s_linear = allo.customize(linear)
    mod = s_linear.build()
    np_x = x.numpy().astype(np.int8)
    # important to have "ascontiguousarray"
    np_w = model.fc.weight.detach().T.numpy().astype(np.int8)
    np_w = np.ascontiguousarray(np_w)
    allo_out = mod(np_x, np_w)
    np.testing.assert_allclose(allo_out, np_x @ np_w, atol=1e-3)
    print("Passed Numpy test!")
    np.testing.assert_allclose(allo_out, out.numpy().astype(np.int8), atol=1e-3)
    print("Passed PyTorch test!")

    s_linear.compose(systolic, instantiate=[int8, int8, int8, L, D, 4 * D, 2, 2])
    with tempfile.TemporaryDirectory() as tmpdir:
        hls_mod = s_linear.build(
            target="vitis_hls",
            mode="csim",
            project=tmpdir,
        )
        csim_out = np.zeros((L, 4 * D), dtype=np.int8)
        if not hls.is_available("vitis_hls"):
            print("Vitis HLS not found, skipping...")
            return
        hls_mod(np_x, np_w, csim_out)
        np.testing.assert_allclose(csim_out, allo_out, atol=1e-3)
        print("Passed HLS csim test!")


if __name__ == "__main__":
    pytest.main([__file__])
