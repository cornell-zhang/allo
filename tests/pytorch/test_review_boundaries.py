# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for FP32 compatibility and explicit quantization boundaries."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from allo.frontend.pytorch import (
    QuantizationConfig,
    approximate_multiplier_shift,
    from_pytorch,
)


class PlainLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(2.0)
            self.linear.bias.fill_(0.5)

    def forward(self, x):
        return self.linear(x)


def test_fp32_linear_scalar_dtype_option_remains_supported():
    model = PlainLinear().eval()
    x = torch.tensor([[1.0], [-2.0]])
    module = from_pytorch(model, (x,), op_dtypes={"linear": "float32"})
    np.testing.assert_array_equal(module(x.numpy()), model(x).detach().numpy())


@pytest.mark.parametrize("key", ["inputs", "default"])
def test_explicit_input_dtype_option_is_honored(key):
    class Add(torch.nn.Module):
        def forward(self, x):
            return x + x

    model = Add().eval()
    example = torch.tensor([[0.25]], dtype=torch.float32)
    module = from_pytorch(
        model,
        (example,),
        op_dtypes={key: "float64", "outputs": "float64"},
    )
    # This is the caller-selected ABI; the tracing example stays float32.
    actual = np.asarray(module(np.array([[0.25]], dtype=np.float64)))
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, [[0.5]])


class QDQConsumer(torch.nn.Module):
    def __init__(self, consumer):
        super().__init__()
        self.consumer = consumer

    def forward(self, x):
        y = torch.quantize_per_tensor(x, 0.25, 0, torch.qint8).dequantize()
        if self.consumer == "mul":
            return y * 2.0
        if self.consumer == "relu":
            return torch.nn.functional.relu(y)
        return y + y


@pytest.mark.parametrize("mode", ["early", "delayed", "fused"])
@pytest.mark.parametrize("consumer", ["mul", "relu"])
def test_qdq_is_preserved_before_float_consumers(mode, consumer):
    model = QDQConsumer(consumer).eval()
    x = torch.tensor([[0.3, -0.3, 40.0, -40.0]], dtype=torch.float32)
    expected = model(x).detach().numpy()
    module = from_pytorch(model, (x,), qdq_lowering_mode=mode)
    np.testing.assert_array_equal(module(x.numpy()), expected)


def test_fused_add_does_not_invent_an_output_quantization_boundary():
    model = QDQConsumer("add").eval()
    x = torch.tensor([[25.0]], dtype=torch.float32)
    expected = model(x).detach().numpy()
    assert expected.item() == 50.0
    module = from_pytorch(model, (x,), qdq_lowering_mode="fused")
    np.testing.assert_array_equal(module(x.numpy()), expected)


def test_native_qdq_linear_does_not_silently_round_the_folded_weight():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, x):
            x = torch.quantize_per_tensor(x, 0.25, 0, torch.qint8).dequantize()
            y = self.linear(x)
            return torch.quantize_per_tensor(y, 0.25, 0, torch.qint8).dequantize()

    model = Model().eval()
    x = torch.tensor([[1.0]], dtype=torch.float32)
    expected = model(x).detach().numpy()
    assert expected.item() == 1.0
    module = from_pytorch(
        model,
        (x,),
        qdq_lowering_mode="fused",
        op_dtypes={"linear": ("int8", "int8", "int32")},
    )
    np.testing.assert_array_equal(module(x.numpy()), expected)


@pytest.mark.parametrize("mode", ["early", "delayed", "fused"])
def test_int_repr_returns_integer_storage(mode):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.quantize_per_tensor(x, 0.25, -3, torch.qint8).int_repr()

    model = Model().eval()
    x = torch.tensor([[0.3, -0.3, 40.0]], dtype=torch.float32)
    expected = model(x).numpy()
    module = from_pytorch(model, (x,), qdq_lowering_mode=mode)
    actual = np.asarray(module(x.numpy()))
    assert actual.dtype == np.int8
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("float_observer", [True, False])
def test_fused_add_preserves_all_consumers(float_observer):
    class Model(torch.nn.Module):
        def forward(self, x):
            q = torch.quantize_per_tensor(x, 0.25, 0, torch.qint8).dequantize()
            total = q + q
            a = torch.quantize_per_tensor(total, 0.25, 0, torch.qint8).dequantize()
            if float_observer:
                return a, total
            b = torch.quantize_per_tensor(total, 1.0, 0, torch.qint8).dequantize()
            return a, b

    model = Model().eval()
    x = torch.tensor([[25.0]], dtype=torch.float32)
    expected = model(x)
    module = from_pytorch(model, (x,), qdq_lowering_mode="fused")
    actual = module(x.numpy())
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want.numpy())


@pytest.mark.parametrize("mode", ["early", "delayed", "fused"])
def test_consecutive_qdq_keeps_both_rounding_boundaries(mode):
    class Model(torch.nn.Module):
        def forward(self, x):
            y = torch.quantize_per_tensor(x, 1.0, 0, torch.qint8).dequantize()
            return torch.quantize_per_tensor(y, 0.25, 0, torch.qint8).dequantize()

    model = Model().eval()
    x = torch.tensor([[0.6, -0.6]], dtype=torch.float32)
    module = from_pytorch(model, (x,), qdq_lowering_mode=mode)
    np.testing.assert_array_equal(module(x.numpy()), model(x).numpy())


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ({"weight_dtype": "uint8"}, NotImplementedError, "signed int8"),
        ({"weight_dtype": "int16"}, NotImplementedError, "signed int8"),
        ({"activation_dtype": "int16"}, NotImplementedError, "int8 or uint8"),
        ({"accumulator_dtype": "int64"}, ValueError, "signed int32"),
        ({"accumulator_dtype": "uint32"}, ValueError, "signed int32"),
    ],
)
def test_unsupported_native_configs_fail_explicitly(config, error, message):
    with pytest.raises(error, match=message):
        QuantizationConfig(**config)


def test_unrepresentable_explicit_qdq_weight_is_rejected():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.1)

        def forward(self, x):
            y = torch.quantize_per_tensor(x, 0.25, 0, torch.qint8).dequantize()
            return self.linear(y)

    with pytest.raises(NotImplementedError, match="exactly representable int8"):
        from_pytorch(
            Model().eval(),
            (torch.ones(1, 1),),
            target="mlir",
            qdq_lowering_mode="fused",
            op_dtypes={"linear": ("int8", "int8", "int32")},
        )


def test_unsupported_requantization_shift_is_rejected():
    with pytest.raises(NotImplementedError, match="shift"):
        approximate_multiplier_shift(2.0**-70)


@pytest.mark.parametrize("mode", ["early", "delayed", "fused"])
def test_large_finite_quantization_saturates_before_integer_conversion(mode):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.quantize_per_tensor(x, 1.0, 0, torch.qint8).dequantize()

    x = torch.tensor([[1.0e12, -1.0e12]], dtype=torch.float32)
    module = from_pytorch(Model().eval(), (x,), qdq_lowering_mode=mode)
    np.testing.assert_array_equal(module(x.numpy()), [[127.0, -128.0]])
