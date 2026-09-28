# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native PyTorch-to-Allo INT8 certification ladder.

This suite validates the full compiler route for deterministic quantized
Linear, residual-MLP, and stacked residual-FFN workloads:

    PyTorch model
      -> torch.fx GraphModule
      -> TorchBuilder
      -> generated Allo source
      -> allo.customize(...)
      -> native Allo/MLIR Schedule
      -> scheduled LLVM execution
      -> HLS code/project generation
      -> optional Vitis C simulation and hardware emulation

It also directly validates the reusable Allo integer kernels used by the route.

This is intentionally stricter than a smoke test:
- fake-quant FP32 arithmetic does not count as native INT8 lowering;
- code emission does not count as numerical verification;
- a skipped Vitis test does not count as a hardware pass;
- integer results are compared bit-for-bit;
- accumulation stress vectors expose accidental int8/int16 accumulation.

Run the normal regression suite without the expensive synthesis and
hardware-emulation gates:

    unset ALLO_RUN_VITIS_CSYN ALLO_RUN_VITIS_HW_EMU
    pytest -q tests/test_full_int8_quant.py -x -rs

Run only software tests:

    pytest -q tests/test_full_int8_quant.py -k "not vitis" -x

Require Vitis HLS and run C simulation:

    ALLO_REQUIRE_VITIS=1 \
    pytest -q tests/test_full_int8_quant.py -x

Run the expensive Vitis synthesis gate:

    ALLO_REQUIRE_VITIS=1 ALLO_RUN_VITIS_CSYN=1 \
    pytest -q tests/test_full_int8_quant.py -x

Run the full Vitis hardware-emulation gate:

    ALLO_REQUIRE_VITIS=1 ALLO_RUN_VITIS_HW_EMU=1 \
    pytest -q tests/test_full_int8_quant.py -x
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest
import torch
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import ShapeProp

import allo
from allo.backend import hls
from allo.customize import Schedule
from allo.frontend.pytorch import (
    QuantInfo,
    TorchBuilder,
    from_pytorch,
    get_qrange,
)
from allo.frontend.tracer import AlloTracer
from allo.ir.types import int8, int32

allo_nn = importlib.import_module("allo.library.nn")
gemv_lib = importlib.import_module("allo.library.gemv")
systolic_lib = importlib.import_module("allo.library.systolic")

torch.set_grad_enabled(False)

QINT8_MIN = -128
QINT8_MAX = 127

INPUT_2D = torch.tensor(
    [
        [3.0, -2.0, 1.0, -4.0],
        [-5.0, 6.0, -7.0, 8.0],
    ],
    dtype=torch.float32,
)

INPUT_3D = torch.tensor(
    [
        [
            [3.0, -2.0, 1.0, -4.0],
            [-5.0, 6.0, -7.0, 8.0],
            [9.0, -10.0, 11.0, -12.0],
        ],
        [
            [-1.0, 2.0, -3.0, 4.0],
            [5.0, -6.0, 7.0, -8.0],
            [-9.0, 10.0, -11.0, 12.0],
        ],
    ],
    dtype=torch.float32,
)

WEIGHT = torch.tensor(
    [
        [2.0, -1.0, 3.0, 1.0],
        [-2.0, 4.0, 1.0, -3.0],
        [1.0, 1.0, -2.0, 4.0],
    ],
    dtype=torch.float32,
)

BIAS = torch.tensor([3.0, -5.0, 7.0], dtype=torch.float32)

# The top-level PyTorch boundary remains float32 because Q/DQ is how the
# quantized domain is represented in the traced graph. The linear region
# itself must lower as int8 input, int8 weight, and int32 output/accumulator.
NATIVE_LINEAR_TRIPLET = ("int8", "int8", "int32")
NATIVE_LINEAR_DTYPES = {
    "inputs": "float32",
    "default": "float32",
    "linear": NATIVE_LINEAR_TRIPLET,
}


class QDQLinear(torch.nn.Module):
    """Deterministic rank-2 Q/DQ Linear with integral parameters."""

    def __init__(
        self,
        *,
        in_features: int = 4,
        out_features: int = 3,
        bias: bool = True,
        input_scale: float = 1.0,
        input_zero_point: int = 0,
        output_scale: float = 1.0,
        output_zero_point: int = 0,
        dtype: torch.dtype = torch.qint8,
        weight: torch.Tensor | None = None,
        bias_value: torch.Tensor | None = None,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.input_scale = float(input_scale)
        self.input_zero_point = int(input_zero_point)
        self.output_scale = float(output_scale)
        self.output_zero_point = int(output_zero_point)
        self.quantized_dtype = dtype

        if weight is None:
            if (in_features, out_features) != (4, 3):
                raise ValueError(
                    "A custom weight is required for non-default dimensions"
                )
            weight = WEIGHT

        with torch.no_grad():
            self.linear.weight.copy_(weight)
            if bias:
                if bias_value is None:
                    if out_features != 3:
                        raise ValueError(
                            "A custom bias is required for non-default dimensions"
                        )
                    bias_value = BIAS
                self.linear.bias.copy_(bias_value)

    def forward(self, x):
        x = torch.quantize_per_tensor(
            x,
            scale=self.input_scale,
            zero_point=self.input_zero_point,
            dtype=self.quantized_dtype,
        ).dequantize()

        y = self.linear(x)

        return torch.quantize_per_tensor(
            y,
            scale=self.output_scale,
            zero_point=self.output_zero_point,
            dtype=self.quantized_dtype,
        ).dequantize()


class QDQLinear3D(torch.nn.Module):
    """Rank-3 version used to require the Qwen-relevant Linear path."""

    def __init__(self, *, bias: bool = True):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=bias)

        with torch.no_grad():
            self.linear.weight.copy_(WEIGHT)
            if bias:
                self.linear.bias.copy_(BIAS)

    def forward(self, x):
        x = torch.quantize_per_tensor(x, 1.0, 0, torch.qint8).dequantize()
        y = self.linear(x)
        return torch.quantize_per_tensor(y, 1.0, 0, torch.qint8).dequantize()


class QDQIdentity(torch.nn.Module):
    """Q/DQ-only model for qint8/quint8 metadata and range checks."""

    def __init__(self, scale: float, zero_point: int, dtype: torch.dtype):
        super().__init__()
        self.scale = float(scale)
        self.zero_point = int(zero_point)
        self.quantized_dtype = dtype

    def forward(self, x):
        return torch.quantize_per_tensor(
            x,
            self.scale,
            self.zero_point,
            self.quantized_dtype,
        ).dequantize()


class QDQReuse(torch.nn.Module):
    """One Q/DQ value with two consumers."""

    def forward(self, x):
        q = torch.quantize_per_tensor(x, 0.25, -2, torch.qint8).dequantize()
        return q + 1.0, q + 2.0


class QDQReindex3D(torch.nn.Module):
    """Q/DQ data reindexing that must preserve integer storage metadata."""

    def forward(self, x):
        q = torch.quantize_per_tensor(x, 0.5, -2, torch.qint8).dequantize()
        q = q.view(2, 4, 3)
        q = q.reshape(2, 4, 3)
        q = q.permute(0, 2, 1)
        q = q.transpose(1, 2).contiguous()
        return torch.quantize_per_tensor(q, 0.5, -2, torch.qint8).dequantize()


class QDQRepeat3D(torch.nn.Module):
    """Q/DQ repeat that must instantiate the integer repeat kernel."""

    def forward(self, x):
        q = torch.quantize_per_tensor(x, 1.0, 0, torch.qint8).dequantize()
        q = q.repeat(2, 1, 1)
        return torch.quantize_per_tensor(q, 1.0, 0, torch.qint8).dequantize()


class QDQNonzeroZeroPointReLU3D(torch.nn.Module):
    """ReLU whose integer-domain zero is not represented by code zero."""

    def forward(self, x):
        q = torch.quantize_per_tensor(x, 1.0, 5, torch.qint8).dequantize()
        q = torch.relu(q)
        return torch.quantize_per_tensor(q, 1.0, 5, torch.qint8).dequantize()


class QDQResidual3D(torch.nn.Module):
    """Unequal-scale residual addition requiring scale reconciliation."""

    def forward(self, a, b):
        a = torch.quantize_per_tensor(a, 0.25, -3, torch.qint8).dequantize()
        b = torch.quantize_per_tensor(b, 0.125, 5, torch.qint8).dequantize()
        y = a + b
        return torch.quantize_per_tensor(y, 0.5, -1, torch.qint8).dequantize()


def trace_model(
    model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
) -> GraphModule:
    """Reproduce the tracing and ShapeProp portion of from_pytorch."""
    model.eval()
    tracer = AlloTracer(model, concrete_args={}, leaf_modules=None)
    graph = tracer.trace()
    gm = GraphModule(tracer.root, graph, model.__class__.__name__)
    ShapeProp(gm).propagate(*inputs)
    return gm


def build_builder(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    *,
    mode: str = "fused",
    op_dtypes: dict | None = None,
) -> tuple[GraphModule, TorchBuilder, str]:
    gm = trace_model(model, inputs)
    builder = TorchBuilder(
        gm,
        inputs,
        op_dtypes=op_dtypes,
        qdq_lowering_mode=mode,
    )
    source = builder.build()
    return gm, builder, source


def torch_result(model: torch.nn.Module, *inputs: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        result = model(*inputs)
    return result.detach().cpu().numpy()


def assert_native_route_text(text: str) -> None:
    lowered = text.lower()
    assert "qhls" not in lowered
    assert "qdq_hls" not in lowered
    assert "qdq_int8_hls" not in lowered


def type_bits(value) -> int | None:
    return getattr(value, "bits", None)


def quantize_reference(
    value: np.ndarray,
    *,
    scale: float,
    zero_point: int,
    qmin: int = QINT8_MIN,
    qmax: int = QINT8_MAX,
) -> np.ndarray:
    """Reference affine quantization with round-to-nearest-even."""
    scaled = np.asarray(value, dtype=np.float64) / float(scale)
    rounded = np.rint(scaled)
    shifted = rounded + int(zero_point)

    if (qmin, qmax) == (QINT8_MIN, QINT8_MAX):
        output_dtype = np.int8
    elif (qmin, qmax) == (0, 255):
        output_dtype = np.uint8
    else:
        raise ValueError(f"Unsupported quantized range: {(qmin, qmax)}")

    return np.clip(shifted, qmin, qmax).astype(output_dtype)


def dequantize_reference(
    value: np.ndarray,
    *,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    return (np.asarray(value, dtype=np.int64) - int(zero_point)) * float(scale)


def qdq_reference(
    value: np.ndarray,
    *,
    scale: float,
    zero_point: int,
    qmin: int = QINT8_MIN,
    qmax: int = QINT8_MAX,
) -> np.ndarray:
    q = quantize_reference(
        value,
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
    )
    return dequantize_reference(q, scale=scale, zero_point=zero_point)


def int8_int8_int32_linear_reference(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.int64)
    weight64 = np.asarray(weight, dtype=np.int64)
    bias64 = np.asarray(bias, dtype=np.int64)

    result = x64 @ weight64.T + bias64

    info = np.iinfo(np.int32)
    assert result.min() >= info.min
    assert result.max() <= info.max

    return result.astype(np.int32)


def find_function_regions(text: str, language: str) -> list[tuple[str, str]]:
    """Extract balanced function bodies from MLIR or emitted C/C++."""
    if language == "mlir":
        patterns = [
            (
                re.compile(
                    r"func\.func(?:\s+(?:private|public))?\s+" r"@([A-Za-z0-9_.$-]+)"
                ),
                False,
            ),
            (
                re.compile(
                    r'"func\.func"\(\)\s*<\{.*?'
                    r'\bsym_name\s*=\s*"([^"]+)".*?'
                    r"\}>\s*\(\{",
                    re.DOTALL,
                ),
                True,
            ),
        ]
    elif language == "cpp":
        patterns = [
            (
                re.compile(
                    r"(?:void|int|float|double|int8_t|int32_t|ap_int<\d+>)\s+"
                    r"([A-Za-z_][A-Za-z0-9_]*)\s*\("
                ),
                False,
            )
        ]
    else:
        raise ValueError(f"Unsupported language: {language}")

    def find_closing_brace(opening: int) -> int | None:
        depth = 0

        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return index

        return None

    regions = []

    for pattern, includes_opening_brace in patterns:
        for match in pattern.finditer(text):
            if includes_opening_brace:
                brace = match.end() - 1
            else:
                brace = text.find("{", match.end())
                if brace < 0:
                    continue

                # Pretty MLIR may place an attribute dictionary before
                # the actual function body:
                #
                # func.func @f(...) attributes {...} {
                #   ...
                # }
                header = text[match.end() : brace]
                if language == "mlir" and re.search(r"\battributes\s*$", header):
                    attributes_end = find_closing_brace(brace)
                    if attributes_end is None:
                        continue

                    body_prefix = re.match(
                        r"\s*\{",
                        text[attributes_end + 1 :],
                    )
                    if body_prefix is None:
                        continue

                    brace = attributes_end + body_prefix.end()

            end = find_closing_brace(brace)
            if end is None:
                continue

            regions.append(
                (
                    match.start(),
                    match.group(1),
                    text[match.start() : end + 1],
                )
            )

    regions.sort(key=lambda region: region[0])
    return [(name, body) for _, name, body in regions]


def find_linear_regions(text: str, language: str) -> list[str]:
    return [
        body
        for name, body in find_function_regions(text, language)
        if "linear" in name.lower() or "qlinear" in name.lower()
    ]


def contains_i8(text: str) -> bool:
    return bool(
        re.search(
            r"(?<![A-Za-z0-9_])(?:i8|int8_t|ap_int\s*<\s*8\s*>)(?![A-Za-z0-9_])",
            text,
        )
    )


def contains_i32(text: str) -> bool:
    return bool(
        re.search(
            r"(?<![A-Za-z0-9_])(?:i32|int32_t|ap_int\s*<\s*32\s*>)(?![A-Za-z0-9_])",
            text,
        )
    )


def require_extra_global(
    builder: TorchBuilder,
    *,
    name_fragment: str,
    dtype: np.dtype,
) -> np.ndarray:
    matches = [
        (name, np.asarray(value))
        for name, value in builder.extra_globals.items()
        if name_fragment in name
    ]
    assert matches, (
        f"TorchBuilder did not register an integer constant override containing "
        f"{name_fragment!r}. Merely declaring the original float constant as an "
        f"integer type is not a valid native INT8 lowering."
    )
    name, value = matches[0]
    assert value.dtype == np.dtype(
        dtype
    ), f"{name} has dtype {value.dtype}; expected {np.dtype(dtype)}"
    return value


def vitis_available_or_skip() -> None:
    available = hls.is_available("vitis_hls")
    if available:
        return
    if os.getenv("ALLO_REQUIRE_VITIS") == "1":
        pytest.fail("ALLO_REQUIRE_VITIS=1, but vitis_hls is not available")
    pytest.skip(
        "vitis_hls is unavailable; this is an environment skip, not a hardware pass"
    )


def assert_project_artifacts(project: Path) -> None:
    required = (
        "kernel.cpp",
        "kernel.h",
        "run.tcl",
        "description.json",
    )
    missing = [name for name in required if not (project / name).is_file()]
    assert not missing, f"Missing generated Vitis project artifacts: {missing}"


def assert_synthesis_report(project: Path) -> None:
    report_root = project / "out.prj" / "solution1" / "syn" / "report"
    reports = list(report_root.glob("*csynth*.xml")) + list(
        report_root.glob("*csynth*.rpt")
    )
    assert reports, f"No synthesis report found under {report_root}"


# ---------------------------------------------------------------------------
# FX and quantization metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "inputs", "placeholder_shape", "linear_shape"),
    [
        (QDQLinear(), (INPUT_2D,), (2, 4), (2, 3)),
        (QDQLinear3D(), (INPUT_3D,), (2, 3, 4), (2, 3, 3)),
    ],
)
def test_fx_graph_and_shape_propagation(
    model,
    inputs,
    placeholder_shape,
    linear_shape,
):
    gm = trace_model(model, inputs)
    nodes = list(gm.graph.nodes)

    assert nodes[0].op == "placeholder"
    assert nodes[-1].op == "output"

    quantize_nodes = [
        node for node in nodes if "quantize_per_tensor" in str(node.target)
    ]
    dequantize_nodes = [
        node
        for node in nodes
        if node.op == "call_method" and node.target == "dequantize"
    ]
    linear_nodes = [
        node for node in nodes if node.op == "call_module" and node.target == "linear"
    ]

    assert len(quantize_nodes) == 2
    assert len(dequantize_nodes) == 2
    assert len(linear_nodes) == 1

    assert tuple(nodes[0].meta["tensor_meta"].shape) == placeholder_shape
    assert tuple(linear_nodes[0].meta["tensor_meta"].shape) == linear_shape


@pytest.mark.parametrize("mode", ["early", "delayed", "fused"])
@pytest.mark.parametrize(
    ("dtype", "scale", "zero_point", "expected_range"),
    [
        (torch.qint8, 0.25, -3, (-128, 127)),
        (torch.quint8, 0.125, 17, (0, 255)),
    ],
)
def test_qdq_modes_preserve_complete_quant_info(
    mode,
    dtype,
    scale,
    zero_point,
    expected_range,
):
    model = QDQIdentity(scale, zero_point, dtype)
    _, builder, _ = build_builder(model, (INPUT_2D,), mode=mode)

    quant_info = builder.get_quant_info_map()
    assert quant_info

    matching = [
        info
        for info in quant_info.values()
        if info.dtype == dtype
        and info.scale == pytest.approx(scale)
        and info.zero_point == zero_point
    ]
    assert matching

    for info in matching:
        assert isinstance(info, QuantInfo)
        assert (info.qmin, info.qmax) == expected_range
        assert info.qmin < info.qmax


@pytest.mark.parametrize(
    ("dtype", "zero_point", "type_name"),
    [
        (torch.qint8, -3, "int8"),
        (torch.quint8, 17, "uint8"),
    ],
)
def test_qdq_materialization_emits_typed_boundaries(
    dtype,
    zero_point,
    type_name,
):
    _, _, source = build_builder(
        QDQIdentity(0.25, zero_point, dtype),
        (INPUT_2D,),
        mode="fused",
    )

    integer_cast = re.search(
        rf"(?P<result>qdq_int_\d+)\s*:\s*{type_name}\s*"
        rf"\[\s*2\s*,\s*4\s*\]\s*\n"
        rf"\s*for i0,\s*i1 in dsl\.grid\(\s*2,\s*4,\s*"
        rf'name="(?P=result)_cast"\s*\):\s*\n'
        rf"\s*(?P=result)\[i0,\s*i1\]\s*=\s*"
        rf"qdq_min_\d+\[i0,\s*i1\]",
        source,
    )
    assert integer_cast is not None

    float_cast = re.search(
        r"(?P<result>qdq_float_\d+)\s*:\s*float32\s*"
        r"\[\s*2\s*,\s*4\s*\]\s*\n"
        r"\s*for i0,\s*i1 in dsl\.grid\(\s*2,\s*4,\s*"
        r'name="(?P=result)_cast"\s*\):\s*\n'
        r"\s*(?P=result)\[i0,\s*i1\]\s*=\s*"
        r"qdq_int_\d+\[i0,\s*i1\]",
        source,
    )
    assert float_cast is not None


def test_quantized_ranges_and_invalid_dtype():
    assert get_qrange(torch.qint8) == (-128, 127)
    assert get_qrange(torch.quint8) == (0, 255)

    with pytest.raises(NotImplementedError, match="Unsupported quantized dtype"):
        get_qrange(torch.qint32)


def test_invalid_qdq_lowering_mode_is_rejected():
    model = QDQLinear()
    gm = trace_model(model, (INPUT_2D,))

    with pytest.raises(ValueError, match="Unknown qdq_lowering_mode"):
        TorchBuilder(
            gm,
            (INPUT_2D,),
            qdq_lowering_mode="not-a-real-mode",
        )


def test_invalid_dtype_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unknown Allo dtype"):
        build_builder(
            QDQLinear(),
            (INPUT_2D,),
            mode="fused",
            op_dtypes={"linear": ("int8", "innt8", "int32")},
        )

    with pytest.raises(ValueError, match="exactly three entries"):
        build_builder(
            QDQLinear(),
            (INPUT_2D,),
            mode="fused",
            op_dtypes={"linear": ("int8", "int8")},
        )

    with pytest.raises(ValueError, match="Unknown Allo dtype"):
        build_builder(
            QDQIdentity(1.0, 0, torch.qint8),
            (INPUT_2D,),
            mode="fused",
            op_dtypes={"outputs": "innt8"},
        )


def test_delayed_qdq_materialization_is_reused_across_consumers():
    _, delayed_builder, delayed_source = build_builder(
        QDQReuse(),
        (INPUT_2D,),
        mode="delayed",
    )
    _, fused_builder, fused_source = build_builder(
        QDQReuse(),
        (INPUT_2D,),
        mode="fused",
    )

    assert delayed_builder.get_quant_info_map()
    assert fused_builder.get_quant_info_map()

    # The same Q/DQ value has two consumers. It must not emit two independent
    # divide/round/clamp/dequantize sequences.
    assert delayed_source.count("qdq_div") <= 2
    assert delayed_source.count("qdq_round") <= 2
    assert fused_source.count("qdq_div") <= 2
    assert fused_source.count("qdq_round") <= 2


def test_quantized_reindexing_preserves_integer_storage():
    gm, builder, source = build_builder(
        QDQReindex3D(),
        (INPUT_3D,),
        mode="fused",
    )

    reindex_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_method"
        and node.target
        in {
            "view",
            "reshape",
            "permute",
            "transpose",
            "contiguous",
        }
    ]
    assert {node.target for node in reindex_nodes} == {
        "view",
        "reshape",
        "permute",
        "transpose",
        "contiguous",
    }

    for node in reindex_nodes:
        assert builder.lookup_quant_info(node) is not None
        assert f"{node.name}_clamped" in builder.get_materialized_quant_map()

    quantize_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and "quantize_per_tensor" in str(node.target)
    ]
    assert len(quantize_nodes) == 2

    final_codes = builder.get_materialized_quant_map()[
        f"{quantize_nodes[-1].name}_clamped"
    ]
    assert final_codes in {
        node.name
        for node in reindex_nodes
        if node.target in {"transpose", "contiguous"}
    }

    assert source.count("dsl.view(") == 2
    assert source.count("dsl.transpose(") == 2
    assert source.count("roundeven(") == 1


def test_quantized_repeat_uses_integer_kernel_dtype():
    _, builder, source = build_builder(
        QDQRepeat3D(),
        (INPUT_3D[:1],),
        mode="fused",
        op_dtypes={"default": "float32"},
    )

    repeat_entries = [
        entry for entry in builder.composition if entry[0] == "repeat_batch3d"
    ]
    assert len(repeat_entries) == 1
    assert type_bits(repeat_entries[0][2][0]) == 8
    assert re.search(r"nn\.repeat_batch3d\[\s*int8\b", source)


def test_nonzero_zero_point_relu_uses_real_domain_fallback():
    _, builder, source = build_builder(
        QDQNonzeroZeroPointReLU3D(),
        (INPUT_3D,),
        mode="fused",
        op_dtypes={"relu": "int8"},
    )

    relu_entries = [entry for entry in builder.composition if entry[0] == "relu3d"]
    assert len(relu_entries) == 1
    assert type_bits(relu_entries[0][2][0]) == 32
    assert re.search(r"nn\.relu3d\[\s*float32\b", source)
    assert not re.search(r"nn\.relu3d\[\s*int8\b", source)


# ---------------------------------------------------------------------------
# TorchBuilder native integer contract
# ---------------------------------------------------------------------------


def test_torchbuilder_emits_native_int8_int8_int32_linear_source():
    _, builder, source = build_builder(
        QDQLinear(),
        (INPUT_2D,),
        mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
    )

    assert "def forward(" in source
    assert re.search(
        r"linear_weight\s*:\s*int8\s*\[\s*3\s*,\s*4\s*\]",
        source,
    )
    assert re.search(
        r"linear_bias\s*:\s*int32\s*\[\s*3\s*\]",
        source,
    )
    assert re.search(
        r"nn\.(?:q?linear2d)\s*\[\s*int8\s*,\s*int8\s*,\s*int32",
        source,
    )
    assert re.search(
        r"nn\.requantize2d\s*\[\s*int32\s*,\s*int64\s*,\s*int8",
        source,
    )
    assert "linear_zero_bias" not in source
    assert builder.param_dtypes["linear_bias"] == "int32"
    assert_native_route_text(source)

    linear_entries = [
        entry for entry in builder.composition if entry[0] in {"linear2d", "qlinear2d"}
    ]
    assert len(linear_entries) == 1

    _, _, instantiate = linear_entries[0]
    assert type_bits(instantiate[0]) == 8
    assert type_bits(instantiate[1]) == 8
    assert type_bits(instantiate[2]) == 32

    requantize_entries = [
        entry for entry in builder.composition if entry[0] == "requantize2d"
    ]
    assert len(requantize_entries) == 1
    assert [type_bits(dtype) for dtype in requantize_entries[0][2][:3]] == [32, 64, 8]

    weight_override = require_extra_global(
        builder,
        name_fragment="linear_weight",
        dtype=np.int8,
    )
    bias_override = require_extra_global(
        builder,
        name_fragment="linear_bias",
        dtype=np.int32,
    )

    np.testing.assert_array_equal(weight_override, WEIGHT.numpy().astype(np.int8))
    np.testing.assert_array_equal(bias_override, BIAS.numpy().astype(np.int32))


def test_torchbuilder_emits_rank3_native_integer_linear():
    _, builder, source = build_builder(
        QDQLinear3D(),
        (INPUT_3D,),
        mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
    )

    assert re.search(
        r"nn\.(?:q?linear3d)\s*\[\s*int8\s*,\s*int8\s*,\s*int32",
        source,
    )

    names = [name for name, _, _ in builder.composition]
    assert any(name in {"linear3d", "qlinear3d"} for name in names)


def test_torchbuilder_supports_native_integer_linear_without_bias():
    _, builder, source = build_builder(
        QDQLinear(bias=False),
        (INPUT_2D,),
        mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
    )

    assert "linear_weight" in source
    assert "linear_bias" not in source
    assert re.search(
        r"linear_zero_bias\s*:\s*int32\s*" r"\[\s*3\s*\]\s*=\s*g_linear_zero_bias",
        source,
    )
    assert any(name in {"linear2d", "qlinear2d"} for name, _, _ in builder.composition)


def test_generated_native_source_is_deterministic():
    _, first_builder, first_source = build_builder(
        QDQLinear(),
        (INPUT_2D,),
        mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
    )
    _, second_builder, second_source = build_builder(
        QDQLinear(),
        (INPUT_2D,),
        mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
    )

    assert first_source == second_source
    assert first_builder.composition == second_builder.composition
    assert first_builder.param_dtypes == second_builder.param_dtypes


def test_unequal_scale_residual_lowers_to_integer_scale_reconciliation():
    model = QDQResidual3D()
    _, builder, source = build_builder(
        model,
        (INPUT_3D, -INPUT_3D),
        mode="fused",
    )

    lowered = source.lower()
    assert builder.get_quant_info_map()

    assert any(
        token in lowered
        for token in (
            "qadd3d",
            "qadd2d",
            "requantize",
            "multiplier",
            "shift",
        )
    ), (
        "Unequal-scale residual addition still lowers as ordinary fake-quant "
        "floating arithmetic. Native residual lowering must reconcile both "
        "input scales in a widened integer domain."
    )


# ---------------------------------------------------------------------------
# Native Allo/MLIR and schedule behavior
# ---------------------------------------------------------------------------


def test_public_frontend_returns_native_schedule_with_integer_mlir():
    schedule = from_pytorch(
        QDQLinear(),
        (INPUT_2D,),
        qdq_lowering_mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
        target="mlir",
    )

    assert isinstance(schedule, Schedule)
    assert schedule.top_func_name == "forward"
    assert schedule.get_loops() is not None

    mlir = str(schedule.module)
    assert "module {" in mlir
    assert "func.func @forward" in mlir
    assert_native_route_text(mlir)

    # Verify the complete Linear → requantization path.
    assert "requantize2d" in mlir
    assert "memref<2x3xi32>" in mlir
    assert "memref<2x3xi8>" in mlir

    # Requantization must not use floating-point fake-quant operations.
    assert not re.search(
        r"math\.roundeven[^\n]*i32",
        mlir,
    )
    assert not re.search(
        r"arith\.sitofp[^\n]*memref",
        mlir,
    )

    linear_regions = find_linear_regions(mlir, "mlir")
    assert linear_regions, "No composed Linear function was found in native MLIR"

    integer_linear_regions = [
        region
        for region in linear_regions
        if contains_i8(region) and contains_i32(region)
    ]
    assert integer_linear_regions, (
        "The composed Linear MLIR does not preserve both i8 storage and i32 "
        "accumulation/output types."
    )

    region = "\n".join(integer_linear_regions)
    assert re.search(r"memref<[^>]*xi8>", region)
    assert re.search(r"memref<[^>]*xi32>", region)
    assert re.search(
        r"arith\.(?:mul|add)i", region
    ), "The native integer Linear region contains no integer multiply/add"


def test_public_qdq_boundary_has_integer_storage_and_float_io():
    schedule = from_pytorch(
        QDQIdentity(0.25, -3, torch.qint8),
        (INPUT_2D,),
        qdq_lowering_mode="fused",
        target="mlir",
    )

    mlir = str(schedule.module)
    functions = dict(find_function_regions(mlir, "mlir"))
    forward = functions["forward"]

    assert "memref<2x4xi8>" in forward
    assert re.search(r"->\s*memref<2x4xf32>", forward)
    assert re.search(r"math\.roundeven.*f32", forward)


def test_composed_linear_schedule_contains_pipeline_transformations():
    schedule = from_pytorch(
        QDQLinear(),
        (INPUT_2D,),
        qdq_lowering_mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
        target="mlir",
    )

    mlir = str(schedule.module)
    recorded = [
        (name, args, kwargs)
        for name, args, kwargs in schedule.primitive_sequences
        if name == "pipeline"
    ]

    recorded_text = "\n".join(
        f"{name} {args} {kwargs}" for name, args, kwargs in recorded
    )
    has_recorded_linear_pipeline = (
        len(recorded) >= 3 and "linear" in recorded_text.lower()
    )
    has_lowered_pipeline_attribute = bool(
        re.search(r"pipeline|initiation_interval|ii\s*=", mlir, re.IGNORECASE)
    )

    assert has_recorded_linear_pipeline or has_lowered_pipeline_attribute, (
        "The composed native Linear route did not preserve evidence of the "
        "registered Linear pipeline schedule."
    )


def test_scheduled_and_unscheduled_integer_linear_match_in_llvm():
    m, n, k = 4, 3, 8
    rng = np.random.default_rng(7)
    x = rng.integers(-16, 17, size=(m, k), dtype=np.int8)
    weight = rng.integers(-8, 9, size=(n, k), dtype=np.int8)
    bias = rng.integers(-100, 101, size=(n,), dtype=np.int32)
    expected = int8_int8_int32_linear_reference(x, weight, bias)

    base_schedule = allo.customize(
        allo_nn.linear2d,
        instantiate=[int8, int8, int32, m, n, k],
    )
    base_module = base_schedule.build(target="llvm")
    base_actual = base_module(x, weight, bias)

    scheduled = allo.customize(
        allo_nn.linear2d,
        instantiate=[int8, int8, int32, m, n, k],
    )
    scheduled = allo_nn.schedule_linear2d(scheduled)
    scheduled_module = scheduled.build(target="llvm")
    scheduled_actual = scheduled_module(x, weight, bias)

    np.testing.assert_array_equal(base_actual, expected)
    np.testing.assert_array_equal(scheduled_actual, expected)
    np.testing.assert_array_equal(scheduled_actual, base_actual)


# ---------------------------------------------------------------------------
# LLVM execution and ABI/error behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_value",
    [
        np.zeros((2, 4), dtype=np.float32),
        np.full((2, 4), 127.0, dtype=np.float32),
        np.full((2, 4), -128.0, dtype=np.float32),
        np.array(
            [[127.0, -128.0, 127.0, -128.0], [-128.0, 127.0, -128.0, 127.0]],
            dtype=np.float32,
        ),
        INPUT_2D.numpy(),
    ],
    ids=["zeros", "positive_limit", "negative_limit", "alternating_limits", "base"],
)
def test_native_frontend_llvm_is_bit_exact(input_value):
    model = QDQLinear()
    expected = torch_result(model, torch.from_numpy(input_value))

    module = from_pytorch(
        model,
        (INPUT_2D,),
        qdq_lowering_mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
        target="llvm",
    )
    actual = module(np.ascontiguousarray(input_value))

    np.testing.assert_array_equal(actual, expected)

    llvm_ir = str(module.module)
    assert "llvm.func" in llvm_ir
    assert contains_i8(llvm_ir)
    assert contains_i32(llvm_ir)
    assert_native_route_text(llvm_ir)


def test_native_rank3_frontend_llvm_is_bit_exact():
    model = QDQLinear3D()
    expected = torch_result(model, INPUT_3D)

    module = from_pytorch(
        model,
        (INPUT_3D,),
        qdq_lowering_mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
        target="llvm",
    )
    actual = module(INPUT_3D.numpy())

    np.testing.assert_array_equal(actual, expected)


def test_int32_accumulator_stress_exposes_narrow_accumulation():
    k = 8
    weight = torch.full((1, k), 127.0, dtype=torch.float32)
    bias = torch.zeros((1,), dtype=torch.float32)
    model = QDQLinear(
        in_features=k,
        out_features=1,
        weight=weight,
        bias_value=bias,
    )
    example = torch.full((1, k), 127.0, dtype=torch.float32)
    expected = torch_result(model, example)

    # Correct accumulation is 8 * 127 * 127 = 129,032, which does not fit in
    # int8 or int16. The final Q/DQ saturates it to +127.
    assert expected.item() == 127.0

    module = from_pytorch(
        model,
        (example,),
        qdq_lowering_mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
        target="llvm",
    )
    actual = module(example.numpy())

    np.testing.assert_array_equal(actual, expected)


def test_llvm_rejects_noncontiguous_input():
    module = from_pytorch(
        QDQLinear(),
        (INPUT_2D,),
        qdq_lowering_mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
        target="llvm",
    )

    noncontiguous = np.arange(8, dtype=np.float32).reshape(4, 2).T
    assert noncontiguous.shape == (2, 4)
    assert not noncontiguous.flags["C_CONTIGUOUS"]

    with pytest.raises(RuntimeError, match="not contiguous"):
        module(noncontiguous)


def test_llvm_rejects_wrong_argument_count():
    module = from_pytorch(
        QDQLinear(),
        (INPUT_2D,),
        qdq_lowering_mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
        target="llvm",
    )

    with pytest.raises(AssertionError, match="input arguments mismatch"):
        module()


def test_unequal_scale_residual_llvm_is_bit_exact():
    model = QDQResidual3D()
    second = torch.flip(INPUT_3D, dims=(-1,))
    expected = torch_result(model, INPUT_3D, second)

    module = from_pytorch(
        model,
        (INPUT_3D, second),
        qdq_lowering_mode="fused",
        target="llvm",
    )
    actual = module(INPUT_3D.numpy(), second.numpy())

    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# HLS emission, project generation, C simulation, and optional full gates
# ---------------------------------------------------------------------------


def test_nn_linear2d_vitis_csim_is_bit_exact(tmp_path):
    vitis_available_or_skip()

    x = INPUT_2D.numpy().astype(np.int8)
    weight = WEIGHT.numpy().astype(np.int8)
    bias = BIAS.numpy().astype(np.int32)
    expected = int8_int8_int32_linear_reference(x, weight, bias)
    actual = np.zeros((2, 3), dtype=np.int32)

    schedule = allo.customize(
        allo_nn.linear2d,
        instantiate=[int8, int8, int32, 2, 3, 4],
    )
    module = schedule.build(
        target="vitis_hls",
        mode="csim",
        project=str(tmp_path / "linear2d_csim.prj"),
    )
    module(x, weight, bias, actual)

    np.testing.assert_array_equal(actual, expected)


def test_qdq_boundary_vitis_csim_is_bit_exact(tmp_path):
    vitis_available_or_skip()

    model = QDQIdentity(1.0, 0, torch.qint8)
    project = tmp_path / "qdq_boundary_csim.prj"
    module = from_pytorch(
        model,
        (INPUT_2D,),
        qdq_lowering_mode="fused",
        target="vitis_hls",
        mode="csim",
        project=str(project),
    )

    expected = torch_result(model, INPUT_2D)
    actual = np.zeros_like(expected)
    module(INPUT_2D.numpy(), actual)

    np.testing.assert_array_equal(actual, expected)


def build_native_hls_project(project: Path, *, mode: str):
    return from_pytorch(
        QDQLinear(),
        (INPUT_2D,),
        qdq_lowering_mode="fused",
        op_dtypes=NATIVE_LINEAR_DTYPES,
        target="vitis_hls",
        mode=mode,
        project=str(project),
    )


def test_native_frontend_emits_integer_hls_for_same_qdq_model(tmp_path):
    project = tmp_path / "native_int8_codegen.prj"
    module = build_native_hls_project(project, mode="csim")

    code = module.hls_code
    assert code
    assert "forward" in code
    assert_native_route_text(code)
    assert_project_artifacts(project)

    linear_regions = find_linear_regions(code, "cpp")
    searchable = "\n".join(linear_regions) if linear_regions else code

    assert contains_i8(searchable), "Generated HLS contains no 8-bit integer type"
    assert contains_i32(searchable), "Generated HLS contains no 32-bit integer type"

    # The strict integer accumulation test must not be represented only as a
    # float multiply/add inside the Linear helper.
    if linear_regions:
        assert not re.search(r"\b(?:float|double)\b", searchable)


def test_native_frontend_vitis_csim_is_bit_exact(tmp_path):
    vitis_available_or_skip()

    project = tmp_path / "native_int8_csim.prj"
    module = build_native_hls_project(project, mode="csim")

    expected = torch_result(QDQLinear(), INPUT_2D)
    output = np.zeros_like(expected, dtype=np.float32)
    module(INPUT_2D.numpy(), output)

    np.testing.assert_array_equal(output, expected)
    assert_project_artifacts(project)


def test_native_frontend_vitis_synthesis_gate(tmp_path):
    if os.getenv("ALLO_RUN_VITIS_CSYN") != "1":
        pytest.skip(
            "Set ALLO_RUN_VITIS_CSYN=1 to run synthesis; this skip is not a synthesis pass"
        )
    vitis_available_or_skip()

    project = tmp_path / "native_int8_csyn.prj"
    module = build_native_hls_project(project, mode="csyn")
    module()
    assert_synthesis_report(project)


def test_native_frontend_vitis_hardware_emulation_gate(tmp_path):
    if os.getenv("ALLO_RUN_VITIS_HW_EMU") != "1":
        pytest.skip(
            "Set ALLO_RUN_VITIS_HW_EMU=1 to run hardware emulation; "
            "this skip is not an RTL-equivalence pass"
        )
    vitis_available_or_skip()
    assert os.getenv("XDEVICE"), "XDEVICE must be set for Vitis hardware emulation"

    project = tmp_path / "native_int8_hw_emu.prj"
    module = build_native_hls_project(project, mode="hw_emu")

    expected = torch_result(QDQLinear(), INPUT_2D)
    output = np.zeros_like(expected, dtype=np.float32)
    module(INPUT_2D.numpy(), output)

    np.testing.assert_array_equal(output, expected)


# ---------------------------------------------------------------------------
# Direct reusable-library kernel regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (1, 1, 1),
        (2, 3, 4),
        (4, 5, 8),
    ],
)
def test_nn_linear2d_int8_int32_llvm_is_exact(m, n, k):
    rng = np.random.default_rng(m * 100 + n * 10 + k)
    x = rng.integers(-32, 33, size=(m, k), dtype=np.int8)
    weight = rng.integers(-16, 17, size=(n, k), dtype=np.int8)
    bias = rng.integers(-500, 501, size=(n,), dtype=np.int32)
    expected = int8_int8_int32_linear_reference(x, weight, bias)

    schedule = allo.customize(
        allo_nn.linear2d,
        instantiate=[int8, int8, int32, m, n, k],
    )
    module = schedule.build(target="llvm")
    actual = module(x, weight, bias)

    np.testing.assert_array_equal(actual, expected)


def test_nn_linear3d_int8_int32_llvm_is_exact():
    batch, length, depth, output_features = 2, 3, 8, 5
    rng = np.random.default_rng(13)
    x = rng.integers(
        -16,
        17,
        size=(batch, length, depth),
        dtype=np.int8,
    )
    weight = rng.integers(
        -8,
        9,
        size=(output_features, depth),
        dtype=np.int8,
    )
    bias = rng.integers(
        -100,
        101,
        size=(output_features,),
        dtype=np.int32,
    )

    expected = (
        np.asarray(x, dtype=np.int64) @ np.asarray(weight, dtype=np.int64).T
        + np.asarray(bias, dtype=np.int64)
    ).astype(np.int32)

    schedule = allo.customize(
        allo_nn.linear3d,
        instantiate=[
            int8,
            int8,
            int32,
            batch,
            length,
            depth,
            output_features,
        ],
    )
    module = schedule.build(target="llvm")
    actual = module(x, weight, bias)

    np.testing.assert_array_equal(actual, expected)


def test_nn_linear2d_hls_preserves_integer_widths():
    schedule = allo.customize(
        allo_nn.linear2d,
        instantiate=[int8, int8, int32, 2, 3, 8],
    )
    module = schedule.build(target="vhls")
    code = module.hls_code

    assert contains_i8(code)
    assert contains_i32(code)
    assert not re.search(r"\b(?:float|double)\b", code)


def test_systolic_int8_int8_int32_overflow_stress():
    m, k, n = 4, 8, 4
    mt, nt = 2, 2

    a = np.full((m, k), 127, dtype=np.int8)
    b = np.full((k, n), 127, dtype=np.int8)
    expected = (np.asarray(a, dtype=np.int64) @ np.asarray(b, dtype=np.int64)).astype(
        np.int32
    )
    assert expected.max() > np.iinfo(np.int16).max

    output = np.zeros((m, n), dtype=np.int32)
    schedule = allo.customize(
        systolic_lib.systolic,
        instantiate=[int8, int8, int32, m, k, n, mt, nt],
    )
    module = schedule.build(target="llvm")
    module(a, b, output)

    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize(
    "symbol",
    [
        "linear3d",
        "qadd2d",
        "qadd3d",
        "requantize2d",
        "requantize3d",
    ],
)
def test_required_native_integer_library_symbols_exist(symbol):
    assert hasattr(allo_nn, symbol), (
        f"allo.library.nn.{symbol} is missing. The native INT8 route requires "
        f"one reusable implementation rather than reproducing its arithmetic "
        f"inside TorchBuilder or an alternate IR."
    )


def test_requantization_reference_covers_ties_saturation_and_zero_point():
    values = np.array(
        [
            -1000.0,
            -128.5,
            -127.5,
            -2.5,
            -1.5,
            -0.5,
            0.5,
            1.5,
            2.5,
            126.5,
            127.5,
            1000.0,
        ],
        dtype=np.float64,
    )
    actual = quantize_reference(values, scale=1.0, zero_point=0)
    expected = np.array(
        [
            -128,
            -128,
            -128,
            -2,
            -2,
            0,
            0,
            2,
            2,
            126,
            127,
            127,
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(actual, expected)

    shifted = quantize_reference(
        np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
        scale=1.0,
        zero_point=5,
    )
    np.testing.assert_array_equal(
        shifted,
        np.array([3, 3, 5, 5, 7, 7], dtype=np.int8),
    )


def test_unsigned_quantization_reference_does_not_wrap_above_127():
    actual = quantize_reference(
        np.array([-1000.0, 0.0, 127.5, 200.0, 1000.0]),
        scale=1.0,
        zero_point=128,
        qmin=0,
        qmax=255,
    )
    expected = np.array([0, 128, 255, 255, 255], dtype=np.uint8)

    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


def test_float_qdq_reference_matches_pytorch_edge_cases():
    values = torch.tensor(
        [
            -1000.0,
            -16.0625,
            -15.9375,
            -0.3125,
            -0.1875,
            -0.0625,
            0.0625,
            0.1875,
            0.3125,
            15.8125,
            15.9375,
            1000.0,
        ],
        dtype=torch.float32,
    )
    scale = 0.125
    zero_point = -2

    expected = (
        torch.quantize_per_tensor(values, scale, zero_point, torch.qint8)
        .dequantize()
        .numpy()
    )
    actual = qdq_reference(
        values.numpy(),
        scale=scale,
        zero_point=zero_point,
    )

    np.testing.assert_array_equal(actual.astype(np.float32), expected)


# ---------------------------------------------------------------------------
# Complete native INT8 residual MLP
# ---------------------------------------------------------------------------

RESIDUAL_MLP_BATCH = 1
RESIDUAL_MLP_SEQUENCE = 3
RESIDUAL_MLP_HIDDEN = 8
RESIDUAL_MLP_INTERMEDIATE = 12

RESIDUAL_INPUT_SCALE = 1.0
RESIDUAL_INPUT_ZERO_POINT = -3

RESIDUAL_HIDDEN_SCALE = 2.0
RESIDUAL_HIDDEN_ZERO_POINT = 0

RESIDUAL_BRANCH_SCALE = 4.0
RESIDUAL_BRANCH_ZERO_POINT = -7

RESIDUAL_OUTPUT_SCALE = 1.0
RESIDUAL_OUTPUT_ZERO_POINT = 3

RESIDUAL_MLP_INPUT = torch.tensor(
    [
        [
            [-7, -3, 0, 2, 4, 6, -5, 1],
            [3, -4, 7, -2, 5, -6, 1, 0],
            [-8, 2, -1, 6, -3, 4, 0, 5],
        ]
    ],
    dtype=torch.float32,
)

RESIDUAL_MLP_UP_WEIGHT = (
    (
        np.arange(
            RESIDUAL_MLP_INTERMEDIATE * RESIDUAL_MLP_HIDDEN,
            dtype=np.int64,
        ).reshape(RESIDUAL_MLP_INTERMEDIATE, RESIDUAL_MLP_HIDDEN)
        * 3
        + 1
    )
    % 5
    - 2
).astype(np.float32)

RESIDUAL_MLP_DOWN_WEIGHT = (
    (
        np.arange(
            RESIDUAL_MLP_HIDDEN * RESIDUAL_MLP_INTERMEDIATE,
            dtype=np.int64,
        ).reshape(RESIDUAL_MLP_HIDDEN, RESIDUAL_MLP_INTERMEDIATE)
        * 2
        + 2
    )
    % 5
    - 2
).astype(np.float32)

RESIDUAL_MLP_EXPECTED = np.array(
    [
        [
            [-103, 93, 48, 2, -44, -90, 91, 49],
            [-85, 76, 11, 18, -11, -94, 81, 4],
            [116, 58, -17, -82, -79, 124, 56, -11],
        ]
    ],
    dtype=np.float32,
)

RESIDUAL_MLP_OP_DTYPES = {
    "inputs": "float32",
    "outputs": "float32",
    "default": "float32",
    "linear": NATIVE_LINEAR_TRIPLET,
    "relu": "int8",
}


class QDQResidualMLP3D(torch.nn.Module):
    """Biasless transformer-shaped residual MLP.

    Bias is deliberately disabled because Qwen MLP projections are biasless and
    bias quantization is already covered by the single-Linear tests.
    """

    def __init__(self):
        super().__init__()

        self.up_proj = torch.nn.Linear(
            RESIDUAL_MLP_HIDDEN,
            RESIDUAL_MLP_INTERMEDIATE,
            bias=False,
        )
        self.activation = torch.nn.ReLU()
        self.down_proj = torch.nn.Linear(
            RESIDUAL_MLP_INTERMEDIATE,
            RESIDUAL_MLP_HIDDEN,
            bias=False,
        )

        with torch.no_grad():
            self.up_proj.weight.copy_(torch.from_numpy(RESIDUAL_MLP_UP_WEIGHT))
            self.down_proj.weight.copy_(torch.from_numpy(RESIDUAL_MLP_DOWN_WEIGHT))

    def forward(self, x):
        residual = torch.quantize_per_tensor(
            x,
            RESIDUAL_INPUT_SCALE,
            RESIDUAL_INPUT_ZERO_POINT,
            torch.qint8,
        ).dequantize()

        hidden = self.up_proj(residual)

        # INT32 accumulator → INT8 activation.
        hidden = torch.quantize_per_tensor(
            hidden,
            RESIDUAL_HIDDEN_SCALE,
            RESIDUAL_HIDDEN_ZERO_POINT,
            torch.qint8,
        ).dequantize()

        hidden = self.activation(hidden)

        # Preserve an explicit quantized boundary after activation.
        hidden = torch.quantize_per_tensor(
            hidden,
            RESIDUAL_HIDDEN_SCALE,
            RESIDUAL_HIDDEN_ZERO_POINT,
            torch.qint8,
        ).dequantize()

        branch = self.down_proj(hidden)

        # Use a scale different from the residual input scale.
        branch = torch.quantize_per_tensor(
            branch,
            RESIDUAL_BRANCH_SCALE,
            RESIDUAL_BRANCH_ZERO_POINT,
            torch.qint8,
        ).dequantize()

        output = residual + branch

        # This is the observable FP32 boundary. Fused lowering keeps the
        # intermediate Q/DQ regions in their integer domains.
        return torch.quantize_per_tensor(
            output,
            RESIDUAL_OUTPUT_SCALE,
            RESIDUAL_OUTPUT_ZERO_POINT,
            torch.qint8,
        ).dequantize()


def residual_mlp_reference(values):
    """Independent numerical reference for the complete residual MLP."""

    residual = qdq_reference(
        values,
        scale=RESIDUAL_INPUT_SCALE,
        zero_point=RESIDUAL_INPUT_ZERO_POINT,
    ).astype(np.float32)

    up_accumulator = (
        residual.astype(np.int64) @ RESIDUAL_MLP_UP_WEIGHT.astype(np.int64).T
    )

    hidden = qdq_reference(
        up_accumulator,
        scale=RESIDUAL_HIDDEN_SCALE,
        zero_point=RESIDUAL_HIDDEN_ZERO_POINT,
    ).astype(np.float32)

    hidden = np.maximum(hidden, 0.0)

    hidden = qdq_reference(
        hidden,
        scale=RESIDUAL_HIDDEN_SCALE,
        zero_point=RESIDUAL_HIDDEN_ZERO_POINT,
    ).astype(np.float32)

    down_accumulator = (
        hidden.astype(np.int64) @ RESIDUAL_MLP_DOWN_WEIGHT.astype(np.int64).T
    )

    branch = qdq_reference(
        down_accumulator,
        scale=RESIDUAL_BRANCH_SCALE,
        zero_point=RESIDUAL_BRANCH_ZERO_POINT,
    ).astype(np.float32)

    residual_sum = residual + branch

    output = qdq_reference(
        residual_sum,
        scale=RESIDUAL_OUTPUT_SCALE,
        zero_point=RESIDUAL_OUTPUT_ZERO_POINT,
    ).astype(np.float32)

    return output


def build_residual_mlp_module(
    target,
    mode="csim",
    project="residual_mlp.prj",
):
    return from_pytorch(
        QDQResidualMLP3D().eval(),
        (RESIDUAL_MLP_INPUT,),
        qdq_lowering_mode="fused",
        op_dtypes=RESIDUAL_MLP_OP_DTYPES,
        target=target,
        mode=mode,
        project=str(project),
    )


def test_residual_mlp_reference_matches_pytorch_and_fixed_golden():
    expected = residual_mlp_reference(RESIDUAL_MLP_INPUT.numpy())
    torch_output = torch_result(QDQResidualMLP3D(), RESIDUAL_MLP_INPUT)

    np.testing.assert_array_equal(expected, RESIDUAL_MLP_EXPECTED)
    np.testing.assert_array_equal(torch_output, RESIDUAL_MLP_EXPECTED)


def test_residual_mlp_torchbuilder_emits_complete_integer_graph():
    gm, builder, source = build_builder(
        QDQResidualMLP3D().eval(),
        (RESIDUAL_MLP_INPUT,),
        mode="fused",
        op_dtypes=RESIDUAL_MLP_OP_DTYPES,
    )

    graph_nodes = list(gm.graph.nodes)

    quantize_nodes = [
        node
        for node in graph_nodes
        if node.op == "call_function" and "quantize_per_tensor" in str(node.target)
    ]
    dequantize_nodes = [
        node
        for node in graph_nodes
        if node.op == "call_method" and node.target == "dequantize"
    ]

    assert len(quantize_nodes) == 5
    assert len(dequantize_nodes) == 5

    module_shapes = {
        node.target: tuple(node.meta["tensor_meta"].shape)
        for node in graph_nodes
        if node.op == "call_module"
    }

    assert module_shapes["up_proj"] == (1, 3, 12)
    assert module_shapes["activation"] == (1, 3, 12)
    assert module_shapes["down_proj"] == (1, 3, 8)

    activation_node = next(
        node
        for node in graph_nodes
        if node.op == "call_module" and node.target == "activation"
    )
    activation_qinfo = builder.lookup_quant_info(activation_node)

    assert activation_qinfo is not None
    assert activation_qinfo.scale == RESIDUAL_HIDDEN_SCALE
    assert activation_qinfo.zero_point == RESIDUAL_HIDDEN_ZERO_POINT

    post_relu_quantize = next(
        user
        for user in activation_node.users
        if user.op == "call_function" and "quantize_per_tensor" in str(user.target)
    )
    materialized = builder.get_materialized_quant_map()

    assert materialized[f"{activation_node.name}_clamped"] == activation_node.name
    assert materialized[f"{post_relu_quantize.name}_clamped"] == activation_node.name

    composition_names = [function_name for function_name, _, _ in builder.composition]

    linear_entries = [
        entry for entry in builder.composition if entry[0] in {"linear3d", "qlinear3d"}
    ]

    assert len(linear_entries) == 2
    assert composition_names.count("requantize3d") == 2
    assert composition_names.count("relu3d") == 1
    assert composition_names.count("qadd3d") == 1

    for _, _, instantiate in linear_entries:
        assert tuple(type_bits(dtype) for dtype in instantiate[:3]) == (8, 8, 32)

    requantize_entries = [
        entry for entry in builder.composition if entry[0] == "requantize3d"
    ]
    assert len(requantize_entries) == 2
    assert all(
        [type_bits(dtype) for dtype in instantiate[:3]] == [32, 64, 8]
        for _, _, instantiate in requantize_entries
    )

    relu_entries = [entry for entry in builder.composition if entry[0] == "relu3d"]
    assert len(relu_entries) == 1
    assert type_bits(relu_entries[0][2][0]) == 8

    assert tuple(linear_entries[0][2][3:]) == (
        1,
        3,
        8,
        12,
    )
    assert tuple(linear_entries[1][2][3:]) == (
        1,
        3,
        12,
        8,
    )

    assert (
        len(
            re.findall(
                r"nn\.(?:q)?linear3d" r"\[\s*int8\s*,\s*int8\s*,\s*int32",
                source,
            )
        )
        == 2
    )
    assert re.search(r"nn\.relu3d\[\s*int8\b", source)
    assert source.count("nn.requantize3d[") == 2
    assert "nn.qadd3d[" in source
    assert "requant_centered" not in source
    assert_native_route_text(source)

    up_weight = require_extra_global(
        builder,
        name_fragment="up_proj_weight",
        dtype=np.int8,
    )
    down_weight = require_extra_global(
        builder,
        name_fragment="down_proj_weight",
        dtype=np.int8,
    )
    # Fused QDQ lowering folds the input activation scale into each
    # downstream Linear weight: ((q - zp) * scale) @ W.T. The up projection
    # sees scale 1.0, while the down projection sees scale 2.0.
    np.testing.assert_array_equal(
        up_weight,
        RESIDUAL_MLP_UP_WEIGHT.astype(np.int8),
    )
    np.testing.assert_array_equal(
        down_weight,
        (RESIDUAL_MLP_DOWN_WEIGHT * RESIDUAL_HIDDEN_SCALE).astype(np.int8),
    )


def test_residual_mlp_native_mlir_contains_complete_integer_graph():
    schedule = build_residual_mlp_module(target="mlir")
    mlir = str(schedule.module)

    assert "i8" in mlir
    assert "i32" in mlir
    assert "f32" in mlir
    assert_native_route_text(mlir)

    for required_function in (
        "linear3d",
        "requantize3d",
        "relu3d",
        "qadd3d",
    ):
        assert required_function in mlir


def test_residual_mlp_llvm_is_bit_exact_across_inputs():
    module = build_residual_mlp_module(target="llvm")
    rng = np.random.default_rng(29)

    index_sum = np.indices(
        (
            RESIDUAL_MLP_BATCH,
            RESIDUAL_MLP_SEQUENCE,
            RESIDUAL_MLP_HIDDEN,
        )
    ).sum(axis=0)

    test_cases = {
        "base": RESIDUAL_MLP_INPUT.numpy(),
        "zeros": np.zeros(
            (
                RESIDUAL_MLP_BATCH,
                RESIDUAL_MLP_SEQUENCE,
                RESIDUAL_MLP_HIDDEN,
            ),
            dtype=np.float32,
        ),
        "positive_limit": np.full(
            (
                RESIDUAL_MLP_BATCH,
                RESIDUAL_MLP_SEQUENCE,
                RESIDUAL_MLP_HIDDEN,
            ),
            127.0,
            dtype=np.float32,
        ),
        "negative_limit": np.full(
            (
                RESIDUAL_MLP_BATCH,
                RESIDUAL_MLP_SEQUENCE,
                RESIDUAL_MLP_HIDDEN,
            ),
            -128.0,
            dtype=np.float32,
        ),
        "alternating_limits": np.where(
            index_sum % 2 == 0,
            -128.0,
            127.0,
        ).astype(np.float32),
        "seeded_random_with_saturation": rng.integers(
            -192,
            193,
            size=(
                RESIDUAL_MLP_BATCH,
                RESIDUAL_MLP_SEQUENCE,
                RESIDUAL_MLP_HIDDEN,
            ),
        ).astype(np.float32),
    }

    model = QDQResidualMLP3D().eval()

    for case_name, values in test_cases.items():
        values = np.ascontiguousarray(values, dtype=np.float32)
        expected = residual_mlp_reference(values)

        with torch.no_grad():
            torch_expected = model(torch.from_numpy(values)).cpu().numpy()

        np.testing.assert_array_equal(
            torch_expected,
            expected,
            err_msg=f"PyTorch/reference mismatch for {case_name}",
        )

        actual = module(values)

        np.testing.assert_array_equal(
            actual,
            expected,
            err_msg=f"LLVM mismatch for {case_name}",
        )


def test_residual_mlp_vitis_csim_is_bit_exact(tmp_path):
    vitis_available_or_skip()

    project = tmp_path / "residual_mlp_csim.prj"
    module = build_residual_mlp_module(
        target="vitis_hls",
        mode="csim",
        project=project,
    )

    input_values = np.ascontiguousarray(
        RESIDUAL_MLP_INPUT.numpy(),
        dtype=np.float32,
    )
    expected = residual_mlp_reference(input_values)
    actual = np.zeros_like(expected, dtype=np.float32)

    module(input_values, actual)

    np.testing.assert_array_equal(actual, expected)
    assert contains_i8(module.hls_code)
    assert contains_i32(module.hls_code)
    assert_project_artifacts(project)


def test_residual_mlp_vitis_synthesis_gate(tmp_path):
    if os.environ.get("ALLO_RUN_VITIS_CSYN") != "1":
        pytest.skip("Set ALLO_RUN_VITIS_CSYN=1 to run residual MLP synthesis")

    vitis_available_or_skip()

    project = tmp_path / "residual_mlp_csyn.prj"
    module = build_residual_mlp_module(
        target="vitis_hls",
        mode="csyn",
        project=project,
    )
    module()
    assert_synthesis_report(project)


def test_residual_mlp_vitis_hardware_emulation_gate(tmp_path):
    if os.environ.get("ALLO_RUN_VITIS_HW_EMU") != "1":
        pytest.skip(
            "Set ALLO_RUN_VITIS_HW_EMU=1 to run residual MLP hardware emulation"
        )

    vitis_available_or_skip()
    assert os.environ.get(
        "XDEVICE"
    ), "XDEVICE must be set for residual MLP hardware emulation"

    project = tmp_path / "residual_mlp_hw_emu.prj"
    module = build_residual_mlp_module(
        target="vitis_hls",
        mode="hw_emu",
        project=project,
    )

    input_values = np.ascontiguousarray(
        RESIDUAL_MLP_INPUT.numpy(),
        dtype=np.float32,
    )
    expected = residual_mlp_reference(input_values)
    actual = np.zeros_like(expected, dtype=np.float32)

    module(input_values, actual)

    np.testing.assert_array_equal(actual, expected)
    assert_project_artifacts(project)


# ---------------------------------------------------------------------------
# Larger composed workload: two residual FFN blocks
# ---------------------------------------------------------------------------

STACKED_RESIDUAL_MLP_EXPECTED = np.array(
    [
        [
            [124, 93, -131, 124, -131, 124, 91, -131],
            [124, 76, -131, 124, -131, 124, 81, -131],
            [64, -131, 111, -54, 124, 72, -131, 117],
        ]
    ],
    dtype=np.float32,
)

STACKED_RESIDUAL_MLP_OP_DTYPES = {
    "inputs": "float32",
    "outputs": "float32",
    "default": "float32",
    "linear": NATIVE_LINEAR_TRIPLET,
    "relu": "int8",
}


class QDQStackedResidualMLP3D(torch.nn.Module):
    """Two residual ReLU FFNs composed at the same rank-3 boundary.

    This deliberately stops short of claiming a full transformer block: it has
    no attention, normalization, RoPE, softmax, or SwiGLU. It is a larger
    supported composition test for repeated integer FFN and residual lowering.
    """

    def __init__(self):
        super().__init__()
        self.first = QDQResidualMLP3D()
        self.second = QDQResidualMLP3D()

    def forward(self, x):
        return self.second(self.first(x))


def stacked_residual_mlp_reference(values):
    first = residual_mlp_reference(values)
    return residual_mlp_reference(first)


def build_stacked_residual_mlp_module(
    target,
    mode="csim",
    project="stacked_residual_mlp.prj",
):
    return from_pytorch(
        QDQStackedResidualMLP3D().eval(),
        (RESIDUAL_MLP_INPUT,),
        qdq_lowering_mode="fused",
        op_dtypes=STACKED_RESIDUAL_MLP_OP_DTYPES,
        target=target,
        mode=mode,
        project=str(project),
    )


def test_stacked_residual_mlp_reference_matches_pytorch_and_fixed_golden():
    expected = stacked_residual_mlp_reference(RESIDUAL_MLP_INPUT.numpy())
    torch_output = torch_result(
        QDQStackedResidualMLP3D(),
        RESIDUAL_MLP_INPUT,
    )

    np.testing.assert_array_equal(expected, STACKED_RESIDUAL_MLP_EXPECTED)
    np.testing.assert_array_equal(torch_output, STACKED_RESIDUAL_MLP_EXPECTED)


def test_stacked_residual_mlp_torchbuilder_emits_all_integer_regions():
    gm, builder, source = build_builder(
        QDQStackedResidualMLP3D().eval(),
        (RESIDUAL_MLP_INPUT,),
        mode="fused",
        op_dtypes=STACKED_RESIDUAL_MLP_OP_DTYPES,
    )
    graph_nodes = list(gm.graph.nodes)

    quantize_nodes = [
        node
        for node in graph_nodes
        if node.op == "call_function" and "quantize_per_tensor" in str(node.target)
    ]
    dequantize_nodes = [
        node
        for node in graph_nodes
        if node.op == "call_method" and node.target == "dequantize"
    ]
    assert len(quantize_nodes) == 10
    assert len(dequantize_nodes) == 10

    module_shapes = {
        node.target: tuple(node.meta["tensor_meta"].shape)
        for node in graph_nodes
        if node.op == "call_module"
    }
    for prefix in ("first", "second"):
        assert module_shapes[f"{prefix}.up_proj"] == (1, 3, 12)
        assert module_shapes[f"{prefix}.activation"] == (1, 3, 12)
        assert module_shapes[f"{prefix}.down_proj"] == (1, 3, 8)

    composition_names = [name for name, _, _ in builder.composition]
    linear_entries = [
        entry for entry in builder.composition if entry[0] in {"linear3d", "qlinear3d"}
    ]
    assert len(linear_entries) == 4
    assert len({(entry[0], entry[1]) for entry in linear_entries}) == 4
    assert composition_names.count("requantize3d") == 5
    assert composition_names.count("relu3d") == 2
    assert composition_names.count("qadd3d") == 2

    expected_instantiations = [
        (1, 3, 8, 12),
        (1, 3, 12, 8),
        (1, 3, 8, 12),
        (1, 3, 12, 8),
    ]
    actual_instantiations = []
    for _, _, instantiate in linear_entries:
        assert tuple(type_bits(dtype) for dtype in instantiate[:3]) == (8, 8, 32)
        actual_instantiations.append(tuple(instantiate[3:]))
    assert actual_instantiations == expected_instantiations

    requantize_entries = [
        entry for entry in builder.composition if entry[0] == "requantize3d"
    ]
    assert len(requantize_entries) == 5
    assert all(
        [type_bits(dtype) for dtype in instantiate[:3]] == [32, 64, 8]
        for _, _, instantiate in requantize_entries
    )

    relu_entries = [entry for entry in builder.composition if entry[0] == "relu3d"]
    assert all(type_bits(entry[2][0]) == 8 for entry in relu_entries)

    assert (
        len(
            re.findall(
                r"nn\.(?:q)?linear3d" r"\[\s*int8\s*,\s*int8\s*,\s*int32",
                source,
            )
        )
        == 4
    )
    assert len(re.findall(r"nn\.relu3d\[\s*int8\b", source)) == 2
    assert source.count("nn.requantize3d[") == 5
    assert source.count("nn.qadd3d[") == 2
    assert (
        len(
            re.findall(
                r"requant_centered_\d+\s*:\s*int32\s*" r"\[\s*1\s*,\s*3\s*,\s*8\s*\]",
                source,
            )
        )
        == 1
    )
    assert_native_route_text(source)

    # Each down projection consumes activations with scale 2.0, so fused QDQ
    # lowering stores 2 * W. Each up projection consumes scale-1.0 input.
    expected_weights = {
        "first_up_proj_weight": RESIDUAL_MLP_UP_WEIGHT,
        "first_down_proj_weight": (RESIDUAL_MLP_DOWN_WEIGHT * RESIDUAL_HIDDEN_SCALE),
        "second_up_proj_weight": RESIDUAL_MLP_UP_WEIGHT,
        "second_down_proj_weight": (RESIDUAL_MLP_DOWN_WEIGHT * RESIDUAL_HIDDEN_SCALE),
    }
    for name_fragment, expected_weight in expected_weights.items():
        actual_weight = require_extra_global(
            builder,
            name_fragment=name_fragment,
            dtype=np.int8,
        )
        np.testing.assert_array_equal(
            actual_weight,
            expected_weight.astype(np.int8),
        )


def test_stacked_residual_mlp_native_mlir_contains_all_integer_regions():
    schedule = build_stacked_residual_mlp_module(target="mlir")
    mlir = str(schedule.module)

    assert "func.func @forward" in mlir
    assert "i8" in mlir
    assert "i32" in mlir
    assert "f32" in mlir
    assert_native_route_text(mlir)

    linear_regions = find_linear_regions(mlir, "mlir")
    integer_linear_regions = [
        region
        for region in linear_regions
        if contains_i8(region) and contains_i32(region)
    ]
    # The same two shape specializations may legally be reused by both blocks.
    assert len(integer_linear_regions) >= 2

    for required_function in (
        "linear3d",
        "requantize3d",
        "relu3d",
        "qadd3d",
    ):
        assert required_function in mlir


def test_stacked_residual_mlp_llvm_is_bit_exact_across_inputs():
    module = build_stacked_residual_mlp_module(target="llvm")
    rng = np.random.default_rng(71)
    test_cases = {
        "base": RESIDUAL_MLP_INPUT.numpy(),
        "zeros": np.zeros(
            (
                RESIDUAL_MLP_BATCH,
                RESIDUAL_MLP_SEQUENCE,
                RESIDUAL_MLP_HIDDEN,
            ),
            dtype=np.float32,
        ),
        "seeded_random_with_saturation": rng.integers(
            -192,
            193,
            size=(
                RESIDUAL_MLP_BATCH,
                RESIDUAL_MLP_SEQUENCE,
                RESIDUAL_MLP_HIDDEN,
            ),
        ).astype(np.float32),
    }
    model = QDQStackedResidualMLP3D().eval()

    for case_name, values in test_cases.items():
        values = np.ascontiguousarray(values, dtype=np.float32)
        expected = stacked_residual_mlp_reference(values)
        torch_expected = torch_result(model, torch.from_numpy(values))
        np.testing.assert_array_equal(
            torch_expected,
            expected,
            err_msg=f"PyTorch/reference mismatch for {case_name}",
        )

        actual = module(values)
        np.testing.assert_array_equal(
            actual,
            expected,
            err_msg=f"LLVM mismatch for {case_name}",
        )

    llvm_ir = str(module.module)
    assert contains_i8(llvm_ir)
    assert contains_i32(llvm_ir)
    assert_native_route_text(llvm_ir)


def test_stacked_residual_mlp_vitis_csim_is_bit_exact(tmp_path):
    vitis_available_or_skip()

    project = tmp_path / "stacked_residual_mlp_csim.prj"
    module = build_stacked_residual_mlp_module(
        target="vitis_hls",
        mode="csim",
        project=project,
    )
    input_values = np.ascontiguousarray(
        RESIDUAL_MLP_INPUT.numpy(),
        dtype=np.float32,
    )
    expected = stacked_residual_mlp_reference(input_values)
    actual = np.zeros_like(expected, dtype=np.float32)

    module(input_values, actual)

    np.testing.assert_array_equal(actual, expected)
    assert contains_i8(module.hls_code)
    assert contains_i32(module.hls_code)
    assert_project_artifacts(project)


def test_stacked_residual_mlp_vitis_synthesis_gate(tmp_path):
    if os.environ.get("ALLO_RUN_VITIS_CSYN") != "1":
        pytest.skip("Set ALLO_RUN_VITIS_CSYN=1 to run stacked residual MLP synthesis")
    vitis_available_or_skip()

    project = tmp_path / "stacked_residual_mlp_csyn.prj"
    module = build_stacked_residual_mlp_module(
        target="vitis_hls",
        mode="csyn",
        project=project,
    )
    module()
    assert_synthesis_report(project)


def test_stacked_residual_mlp_vitis_hardware_emulation_gate(tmp_path):
    if os.environ.get("ALLO_RUN_VITIS_HW_EMU") != "1":
        pytest.skip(
            "Set ALLO_RUN_VITIS_HW_EMU=1 to run stacked residual MLP "
            "hardware emulation"
        )
    vitis_available_or_skip()
    assert os.environ.get(
        "XDEVICE"
    ), "XDEVICE must be set for stacked residual MLP hardware emulation"

    project = tmp_path / "stacked_residual_mlp_hw_emu.prj"
    module = build_stacked_residual_mlp_module(
        target="vitis_hls",
        mode="hw_emu",
        project=project,
    )
    input_values = np.ascontiguousarray(
        RESIDUAL_MLP_INPUT.numpy(),
        dtype=np.float32,
    )
    expected = stacked_residual_mlp_reference(input_values)
    actual = np.zeros_like(expected, dtype=np.float32)

    module(input_values, actual)

    np.testing.assert_array_equal(actual, expected)
    assert_project_artifacts(project)
