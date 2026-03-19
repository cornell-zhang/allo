# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import allo

torch.set_grad_enabled(False)
torch.manual_seed(0)


def assert_fx_has_qdq(model):
    traced = torch.fx.symbolic_trace(model)
    graph_str = str(traced.graph)
    assert "torch.quantize_per_tensor" in graph_str
    assert "dequantize" in graph_str
    return graph_str


class QDQMLPInlineQInt8(nn.Module):
    def __init__(self, in_dim=16, hid=32, out_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = torch.quantize_per_tensor(x, 0.05, 0, torch.qint8).dequantize()
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.quantize_per_tensor(x, 0.05, 0, torch.qint8).dequantize()
        x = self.fc2(x)
        return x


class QDQMLPAttrQInt8(nn.Module):
    def __init__(self, in_dim=16, hid=32, out_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid, out_dim)

        self.scale = 0.05
        self.zero_point = 0
        self.dtype = torch.qint8

    def forward(self, x):
        x = torch.quantize_per_tensor(
            x, self.scale, self.zero_point, self.dtype
        ).dequantize()
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.quantize_per_tensor(
            x, self.scale, self.zero_point, self.dtype
        ).dequantize()
        x = self.fc2(x)
        return x


class QDQMLPAttrQUInt8(nn.Module):
    def __init__(self, in_dim=16, hid=32, out_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid, out_dim)

        self.scale = 0.05
        self.zero_point = 17
        self.dtype = torch.quint8

    def forward(self, x):
        x = torch.quantize_per_tensor(
            x, self.scale, self.zero_point, self.dtype
        ).dequantize()
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.quantize_per_tensor(
            x, self.scale, self.zero_point, self.dtype
        ).dequantize()
        x = self.fc2(x)
        return x


@pytest.mark.parametrize(
    "model_cls,input_shape",
    [
        (QDQMLPInlineQInt8, (4, 16)),
        (QDQMLPAttrQInt8, (4, 16)),
        (QDQMLPAttrQUInt8, (4, 16)),
    ],
)
def test_qdq_fx_graph_contains_expected_ops(model_cls, input_shape):
    model = model_cls().eval()
    graph_str = assert_fx_has_qdq(model)

    assert "call_function[target=torch.quantize_per_tensor]" in graph_str
    assert "call_method[target=dequantize]" in graph_str


@pytest.mark.parametrize(
    "model_cls,input_shape",
    [
        (QDQMLPInlineQInt8, (4, 16)),
        (QDQMLPAttrQInt8, (4, 16)),
        (QDQMLPAttrQUInt8, (4, 16)),
    ],
)
def test_qdq_mlir_generation(model_cls, input_shape):
    model = model_cls().eval()
    x = torch.randn(*input_shape, dtype=torch.float32)

    sched = allo.frontend.from_pytorch(
        model,
        example_inputs=[x],
        target="mlir",
        verbose=False,
    )
    assert sched is not None

    mlir_text = str(sched.module)

    # frontend/QDQ lowering evidence
    assert "math.roundeven" in mlir_text
    assert "arith.maximumf" in mlir_text
    assert "arith.minimumf" in mlir_text

    # dequantize remains a float boundary in generated program
    assert "quantize_per_tensor" in mlir_text
    assert "dequantize" in mlir_text

    # old unsupported clamp op should not appear anymore
    assert "math.clampf" not in mlir_text


@pytest.mark.parametrize(
    "model_cls,input_shape",
    [
        (QDQMLPInlineQInt8, (4, 16)),
        (QDQMLPAttrQInt8, (4, 16)),
        (QDQMLPAttrQUInt8, (4, 16)),
    ],
)
def test_qdq_llvm_cpu_matches_pytorch(model_cls, input_shape):
    model = model_cls().eval()
    x = torch.randn(*input_shape, dtype=torch.float32)

    golden = model(x).detach().numpy()

    llvm_mod = allo.frontend.from_pytorch(
        model,
        example_inputs=[x],
        target="llvm",
        verbose=False,
    )
    out = np.asarray(llvm_mod(x.detach().numpy()))

    assert out.shape == golden.shape
    np.testing.assert_allclose(out, golden, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("batch", [1, 4, 7])
def test_qdq_batch_variation(batch):
    model = QDQMLPAttrQInt8().eval()
    x = torch.randn(batch, 16, dtype=torch.float32)

    golden = model(x).detach().numpy()

    llvm_mod = allo.frontend.from_pytorch(
        model,
        example_inputs=[x],
        target="llvm",
        verbose=False,
    )
    out = np.asarray(llvm_mod(x.detach().numpy()))

    assert out.shape == golden.shape
    np.testing.assert_allclose(out, golden, rtol=1e-4, atol=1e-5)


def test_non_quantized_fp32_still_works():
    class FP32MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 8)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = FP32MLP().eval()
    x = torch.randn(4, 16, dtype=torch.float32)

    golden = model(x).detach().numpy()

    llvm_mod = allo.frontend.from_pytorch(
        model,
        example_inputs=[x],
        target="llvm",
        verbose=False,
    )
    out = np.asarray(llvm_mod(x.detach().numpy()))

    assert out.shape == golden.shape
    np.testing.assert_allclose(out, golden, rtol=1e-5, atol=1e-6)
