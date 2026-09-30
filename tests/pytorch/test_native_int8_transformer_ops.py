# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused certification for native INT8 transformer operators."""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import ShapeProp

import allo
from allo.frontend.pytorch import QuantizationConfig, TorchBuilder
from allo.frontend.tracer import AlloTracer
from allo.ir.types import int8, int32, int64, uint8
from examples.torch.smollm2 import (
    SmolLM2Config,
    SmolLM2DecoderLayerHarness,
    smollm2_leaf_modules,
)


allo_nn = importlib.import_module("allo.library.nn")


REQUIRED_TRANSFORMER_KERNELS = {
    "rms_norm3d",
    "silu3d",
    "rope3d",
    "positioned_rope3d",
    "repeat_interleave3d",
    "causal_softmax3d",
    "embedding2d",
    "kv_cache_update3d",
    "qmul3d",
    "qmatmul3d",
    "qsilu3d",
    "qrope3d",
    "qpositioned_rope3d",
    "qcausal_softmax3d",
    "qrms_norm3d",
    "qembedding2d",
    "qkv_cache_update3d",
    "requantize_per_channel2d",
    "requantize_per_channel3d",
}


def round_shift_ties_even(value: int, shift: int) -> int:
    if shift < 0:
        return value << -shift
    if shift == 0:
        return value
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        quotient += 1
    return -quotient if value < 0 else quotient


def per_channel_requantization_reference(
    values,
    multipliers,
    shifts,
    input_zero_point,
    output_zero_point,
    qmin,
    qmax,
):
    expected = np.empty(values.shape, dtype=np.int8)

    for index in np.ndindex(values.shape):
        channel = index[-1]
        centered = int(values[index]) - input_zero_point
        scaled = centered * int(multipliers[channel])
        rounded = round_shift_ties_even(
            scaled,
            int(shifts[channel]),
        )
        quantized = rounded + output_zero_point
        expected[index] = np.clip(
            quantized,
            qmin,
            qmax,
        )

    return expected


def build_source(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
    tracer = AlloTracer(
        model,
        concrete_args={},
        leaf_modules=smollm2_leaf_modules(),
    )
    graph = tracer.trace()
    gm = GraphModule(tracer.root, graph, model.__class__.__name__)
    ShapeProp(gm).propagate(*inputs)
    builder = TorchBuilder(
        gm,
        inputs,
        leaf_modules=smollm2_leaf_modules(),
        quant_config=QuantizationConfig(),
        qdq_lowering_mode="fused",
    )
    return builder, builder.build()


def test_transformer_kernel_symbols_are_exported_and_registered():
    exported = {name for name in REQUIRED_TRANSFORMER_KERNELS if hasattr(allo_nn, name)}
    assert exported == REQUIRED_TRANSFORMER_KERNELS
    registered = {kernel.__name__ for kernel in allo.library.KERNEL2SCHEDULE}
    assert REQUIRED_TRANSFORMER_KERNELS <= registered


def test_repeat_interleave3d_uses_grouped_query_ordering_in_llvm():
    heads, length, width, repeat = 3, 2, 4, 3
    values = np.arange(heads * length * width, dtype=np.int8).reshape(
        heads, length, width
    )
    expected = np.repeat(values, repeat, axis=0)
    schedule = allo.customize(
        allo_nn.repeat_interleave3d,
        instantiate=[int8, heads, length, width, repeat],
    )
    module = schedule.build(target="llvm")
    np.testing.assert_array_equal(module(values), expected)


def test_qmul3d_uses_int64_requantization_in_llvm():
    shape = (1, 2, 4)
    lhs = np.array([[[-8, -3, 2, 7], [11, -12, 31, -32]]], dtype=np.int8)
    rhs = np.array([[[5, -6, 7, -8], [-9, 10, -11, 12]]], dtype=np.int8)
    multiplier, shift = 3, 2
    expected = np.empty(shape, dtype=np.int8)
    for index in np.ndindex(shape):
        value = round_shift_ties_even(
            int(lhs[index]) * int(rhs[index]) * multiplier, shift
        )
        expected[index] = np.clip(value, -128, 127)
    schedule = allo.customize(
        allo_nn.qmul3d,
        instantiate=[int8, int8, int64, int8, *shape],
    )
    module = schedule.build(target="llvm")
    actual = module(
        lhs,
        rhs,
        multiplier,
        shift,
        0,
        0,
        0,
        -128,
        127,
    )
    np.testing.assert_array_equal(actual, expected)


def test_qmatmul3d_accumulates_int32_and_requantizes_int64_in_llvm():
    batch, rows, reduction, columns = 2, 3, 8, 4
    rng = np.random.default_rng(19)
    lhs = rng.integers(-16, 17, size=(batch, rows, reduction), dtype=np.int8)
    rhs = rng.integers(-12, 13, size=(batch, reduction, columns), dtype=np.int8)
    multiplier, shift = 5, 3
    accumulator = np.matmul(lhs.astype(np.int32), rhs.astype(np.int32))
    expected = np.empty((batch, rows, columns), dtype=np.int8)
    for index in np.ndindex(expected.shape):
        value = round_shift_ties_even(int(accumulator[index]) * multiplier, shift)
        expected[index] = np.clip(value, -128, 127)
    schedule = allo.customize(
        allo_nn.qmatmul3d,
        instantiate=[
            int8,
            int8,
            int32,
            int64,
            int8,
            batch,
            rows,
            reduction,
            columns,
        ],
    )
    module = schedule.build(target="llvm")
    actual = module(
        lhs,
        rhs,
        multiplier,
        shift,
        0,
        0,
        0,
        -128,
        127,
    )
    np.testing.assert_array_equal(actual, expected)


def test_qsilu_lut_includes_the_input_scale_in_llvm():
    shape = (1, 2, 4)
    qmin = -128
    input_scale, output_scale = 0.125, 0.0625
    values = np.array([[[-16, -8, -1, 0], [1, 8, 16, 32]]], dtype=np.int8)
    codes = np.arange(-128, 128, dtype=np.int32)
    real = codes * input_scale
    table = np.clip(
        np.rint((real / (1.0 + np.exp(-real))) / output_scale), -128, 127
    ).astype(np.int8)
    expected = table[values.astype(np.int16) - qmin]
    schedule = allo.customize(
        allo_nn.qsilu3d,
        instantiate=[int8, int8, *shape],
    )
    module = schedule.build(target="llvm")
    np.testing.assert_array_equal(module(values, table, qmin), expected)


def test_qcausal_softmax_uses_input_scale_and_mask_in_llvm():
    heads, rows, columns = 1, 2, 4
    input_scale = 0.125
    causal_offset = 1
    values = np.array([[[7, -2, -8, -16], [5, 3, -4, -20]]], dtype=np.int8)
    table = np.rint(
        np.exp(-np.arange(256, dtype=np.float64) * input_scale) * (1 << 20)
    ).astype(np.int32)
    output_multiplier = 255 * (1 << 20)
    expected = np.zeros((heads, rows, columns), dtype=np.uint8)
    for head in range(heads):
        for row in range(rows):
            allowed = causal_offset + row + 1
            row_values = values[head, row, :allowed].astype(np.int32)
            row_max = int(row_values.max())
            exponentials = np.asarray(
                [table[row_max - int(value)] for value in row_values],
                dtype=np.int64,
            )
            denominator = int(exponentials.sum()) << 20
            for column, exponential in enumerate(exponentials):
                numerator = int(exponential) * output_multiplier
                quotient, remainder = divmod(numerator, denominator)
                if remainder * 2 > denominator or (
                    remainder * 2 == denominator and quotient & 1
                ):
                    quotient += 1
                expected[head, row, column] = quotient

    schedule = allo.customize(
        allo_nn.qcausal_softmax3d,
        instantiate=[int8, uint8, heads, rows, columns],
    )
    module = schedule.build(target="llvm")
    actual = module(
        values,
        table,
        output_multiplier,
        20,
        causal_offset,
        0,
        0,
        0,
        255,
    )
    np.testing.assert_array_equal(actual, expected)


def test_qrms_norm_uses_widened_integer_square_root_in_llvm():
    batch, rows, width = 1, 2, 8
    values = np.array(
        [[[-24, -11, -3, 0, 4, 13, 21, 31], [40, -35, 27, -19, 11, -7, 3, 1]]],
        dtype=np.int8,
    )
    weight = np.array([110, 115, 119, 123, 127, 121, 117, 113], dtype=np.int8)
    input_scale, weight_scale, output_scale = 0.0625, 0.0078125, 0.03125
    eps = 1.0e-5
    factor = int(round(weight_scale * math.sqrt(width) / output_scale * (1 << 20)))
    eps_codes = int(round(eps * width / (input_scale**2)))
    expected = np.empty_like(values)
    for batch_index in range(batch):
        for row in range(rows):
            centered = values[batch_index, row].astype(np.int64)
            radicand = (int(np.dot(centered, centered)) + eps_codes) << 24
            denominator = math.isqrt(radicand) << 8
            for column in range(width):
                numerator = (
                    int(values[batch_index, row, column]) * int(weight[column]) * factor
                )
                quotient, remainder = divmod(abs(numerator), denominator)
                if remainder * 2 > denominator or (
                    remainder * 2 == denominator and quotient & 1
                ):
                    quotient += 1
                if numerator < 0:
                    quotient = -quotient
                expected[batch_index, row, column] = np.clip(quotient, -128, 127)

    schedule = allo.customize(
        allo_nn.qrms_norm3d,
        instantiate=[int8, int8, int64, int8, batch, rows, width],
    )
    module = schedule.build(target="llvm")
    actual = module(
        values,
        weight,
        factor,
        eps_codes,
        0,
        0,
        0,
        -128,
        127,
    )
    np.testing.assert_array_equal(actual, expected)


def test_qkv_cache_update_reconciles_scales_and_preserves_other_tokens_in_llvm():
    heads, update_length, cache_length, width = 1, 2, 5, 4
    values = np.array([[[-9, -5, 5, 9], [12, -12, 15, -15]]], dtype=np.int8)
    cache = np.arange(heads * cache_length * width, dtype=np.int8).reshape(
        heads, cache_length, width
    )
    position = 2
    expected = cache.copy()
    for head in range(heads):
        for row in range(update_length):
            for column in range(width):
                expected[head, position + row, column] = round_shift_ties_even(
                    int(values[head, row, column]), 1
                )

    schedule = allo.customize(
        allo_nn.qkv_cache_update3d,
        instantiate=[
            int8,
            int8,
            int64,
            heads,
            update_length,
            cache_length,
            width,
        ],
    )
    module = schedule.build(target="llvm")
    actual = module(
        values,
        cache,
        1,
        1,
        0,
        0,
        -128,
        127,
        position,
    )
    np.testing.assert_array_equal(actual, expected)


def test_real_sized_smollm2_layer_emits_the_complete_composed_integer_graph():
    torch.manual_seed(4)
    config = SmolLM2Config(num_hidden_layers=1)
    model = SmolLM2DecoderLayerHarness(config, sequence_length=8).eval()
    hidden_states = torch.randn(1, 8, config.hidden_size)
    builder, source = build_source(model, (hidden_states,))

    required_source = {
        "nn.qrms_norm3d[",
        "nn.linear3d[",
        "nn.qrope3d[",
        "nn.repeat_interleave3d[",
        "nn.qmatmul3d[",
        "nn.qcausal_softmax3d[",
        "nn.qsilu3d[",
        "nn.qmul3d[",
        "nn.qadd3d[",
    }
    assert all(token in source for token in required_source)
    assert source.count("nn.linear3d[") == 7
    assert source.count("nn.qrms_norm3d[") == 2
    assert source.count("nn.qmatmul3d[") == 2
    assert source.count("nn.qadd3d[") == 2
    composition_names = [name for name, _, _ in builder.composition]
    assert "qcausal_softmax3d" in composition_names
    assert math.isclose(model.layer.self_attn.scaling, 1.0 / 8.0)


def test_requantize_per_channel2d_is_bit_exact_in_llvm():
    rows, channels = 2, 5

    values = np.array(
        [
            [-20, -9, -1, 0, 1],
            [7, 15, 31, 1000, -1000],
        ],
        dtype=np.int32,
    )
    multipliers = np.array(
        [1, 3, 5, 7, 9],
        dtype=np.int64,
    )
    shifts = np.array(
        [0, 1, -1, 3, 4],
        dtype=np.int32,
    )

    input_zero_point = -3
    output_zero_point = 4
    qmin, qmax = -128, 127

    expected = per_channel_requantization_reference(
        values,
        multipliers,
        shifts,
        input_zero_point,
        output_zero_point,
        qmin,
        qmax,
    )

    schedule = allo.customize(
        allo_nn.requantize_per_channel2d,
        instantiate=[
            int32,
            int64,
            int8,
            rows,
            channels,
        ],
    )
    module = schedule.build(target="llvm")

    actual = module(
        values,
        multipliers,
        shifts,
        input_zero_point,
        output_zero_point,
        qmin,
        qmax,
    )

    np.testing.assert_array_equal(actual, expected)


def test_requantize_per_channel3d_is_bit_exact_in_llvm():
    batch, rows, channels = 2, 2, 4

    values = np.array(
        [
            [
                [-17, -8, -1, 0],
                [1, 7, 15, 31],
            ],
            [
                [63, 127, 255, 1000],
                [-63, -127, -255, -1000],
            ],
        ],
        dtype=np.int32,
    )
    multipliers = np.array(
        [3, 5, 7, 11],
        dtype=np.int64,
    )
    shifts = np.array(
        [1, 2, 3, -1],
        dtype=np.int32,
    )

    input_zero_point = 2
    output_zero_point = -5
    qmin, qmax = -128, 127

    expected = per_channel_requantization_reference(
        values,
        multipliers,
        shifts,
        input_zero_point,
        output_zero_point,
        qmin,
        qmax,
    )

    schedule = allo.customize(
        allo_nn.requantize_per_channel3d,
        instantiate=[
            int32,
            int64,
            int8,
            batch,
            rows,
            channels,
        ],
    )
    module = schedule.build(target="llvm")

    actual = module(
        values,
        multipliers,
        shifts,
        input_zero_point,
        output_zero_point,
        qmin,
        qmax,
    )

    np.testing.assert_array_equal(actual, expected)
