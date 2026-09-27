# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-public-methods, too-many-instance-attributes, broad-exception-caught
# BEGIN NATIVE INT8 QUANTIZATION: ADDED lint allowances
# pylint: disable=too-many-arguments, too-many-locals
# END NATIVE INT8 QUANTIZATION: ADDED lint allowances

import operator
import inspect
import math

# BEGIN NATIVE INT8 QUANTIZATION: ADDED imports
from dataclasses import dataclass
import warnings

import numpy as np

# END NATIVE INT8 QUANTIZATION: ADDED imports

try:
    import torch
    from torch import fx
    from torch.nn import functional as F
    from torch.fx.graph_module import GraphModule
    from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata
    from .tracer import AlloTracer
except ImportError:
    pass
from .library import CoreAttention_lib, KVCache_lib, SliceClsToken_lib
from .. import dsl
from ..library import nn
from ..ir import types
from ..customize import customize
from ..ir.types import float32, AlloType

# BEGIN NATIVE INT8 QUANTIZATION: ADDED public metadata and qparam helpers


@dataclass(frozen=True)
class QuantInfo:
    scale: float
    zero_point: int
    dtype: object
    qmin: int
    qmax: int

    def __post_init__(self):
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Quantization scale must be finite and positive")
        if self.qmin >= self.qmax or not self.qmin <= self.zero_point <= self.qmax:
            raise ValueError("Invalid quantization range or zero point")


@dataclass(frozen=True)
class QuantizationConfig:
    activation_dtype: object = types.int8
    weight_dtype: object = types.int8
    accumulator_dtype: object = types.int32
    calibration_method: str = "symmetric_minmax"

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED weight quantization granularity
    weight_granularity: str = "per_tensor"
    # END NATIVE INT8 QUANTIZATION: ADDED weight quantization granularity

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED generic SmoothQuant controls
    # SmoothQuant runs offline before activation calibration. These floating-
    # point calculations are therefore not emitted inside native INT8 regions.
    smoothquant_alpha: float | None = None
    smoothquant_min_scale: float = 1.0e-5
    # END NATIVE INT8 QUANTIZATION: ADDED generic SmoothQuant controls

    def __post_init__(self):
        if self.calibration_method not in {"symmetric_minmax", "minmax"}:
            raise ValueError("Unknown calibration_method")

        # BEGIN NATIVE INT8 QUANTIZATION: ADDED granularity validation
        if self.weight_granularity not in {"per_tensor", "per_channel"}:
            raise ValueError(
                "weight_granularity must be 'per_tensor' or 'per_channel'"
            )
        # END NATIVE INT8 QUANTIZATION: ADDED granularity validation

        # BEGIN NATIVE INT8 QUANTIZATION: ADDED SmoothQuant validation
        if self.smoothquant_alpha is not None and not (
            0.0 <= float(self.smoothquant_alpha) <= 1.0
        ):
            raise ValueError("smoothquant_alpha must be in [0, 1]")
        if not (
            math.isfinite(float(self.smoothquant_min_scale))
            and float(self.smoothquant_min_scale) > 0.0
        ):
            raise ValueError(
                "smoothquant_min_scale must be finite and positive"
            )
        # END NATIVE INT8 QUANTIZATION: ADDED SmoothQuant validation

        get_qrange(self.activation_dtype)
        get_qrange(self.weight_dtype)
        accumulator = self.accumulator_dtype
        if isinstance(accumulator, str):
            accumulator = getattr(types, accumulator, None)
        if not (
            isinstance(accumulator, AlloType)
            and accumulator.__class__.__name__ in {"Int", "UInt"}
            and accumulator.bits >= 32
        ):
            raise ValueError(
                "accumulator_dtype must be an integer of at least 32 bits"
            )


def get_qrange(dtype):
    torch_module = globals().get("torch")
    if torch_module is not None and dtype is torch_module.qint8:
        return -128, 127
    if torch_module is not None and dtype is torch_module.quint8:
        return 0, 255
    if isinstance(dtype, str):
        dtype = getattr(types, dtype, dtype)
    if dtype is np.int8 or dtype == np.dtype(np.int8):
        return -128, 127
    if dtype is np.uint8 or dtype == np.dtype(np.uint8):
        return 0, 255
    if isinstance(dtype, AlloType) and dtype.__class__.__name__ in {"Int", "UInt"}:
        if dtype.__class__.__name__ == "UInt":
            return 0, (1 << dtype.bits) - 1
        return -(1 << (dtype.bits - 1)), (1 << (dtype.bits - 1)) - 1
    raise NotImplementedError(f"Unsupported quantized dtype: {dtype}")


def choose_qparams(min_val, max_val, qmin=-128, qmax=127, *, symmetric=True):
    min_val, max_val = float(min_val), float(max_val)
    if qmin >= qmax:
        raise ValueError("Quantized range must satisfy qmin < qmax")
    if not math.isfinite(min_val) or not math.isfinite(max_val):
        raise ValueError("Calibration range must be finite")
    if min_val > max_val:
        raise ValueError("Calibration range must satisfy min_val <= max_val")
    if min_val == max_val == 0.0:
        return 1.0, 0 if qmin <= 0 <= qmax else qmin
    min_val, max_val = min(min_val, 0.0), max(max_val, 0.0)
    if symmetric and qmin < 0 < qmax:
        scale = max(abs(min_val), abs(max_val)) / min(abs(qmin), abs(qmax))
        return max(scale, np.finfo(np.float32).tiny), 0
    scale = max(
        (max_val - min_val) / (qmax - qmin),
        np.finfo(np.float32).tiny,
    )
    zero_point = int(round(qmin - min_val / scale))
    return scale, min(qmax, max(qmin, zero_point))


def approximate_multiplier_shift(real_multiplier, multiplier_bits=31):
    real_multiplier = float(real_multiplier)
    if not 2 <= multiplier_bits <= 62:
        raise ValueError("multiplier_bits must be between 2 and 62")
    if not math.isfinite(real_multiplier) or real_multiplier < 0:
        raise ValueError("real_multiplier must be finite and nonnegative")
    if real_multiplier == 0:
        return 0, 0
    mantissa, exponent = math.frexp(real_multiplier)
    multiplier = int(round(mantissa * (1 << multiplier_bits)))
    if multiplier == 1 << multiplier_bits:
        multiplier >>= 1
        exponent += 1
    shift = multiplier_bits - exponent
    while shift > 0 and multiplier % 2 == 0:
        multiplier >>= 1
        shift -= 1
    return multiplier, shift


# END NATIVE INT8 QUANTIZATION: ADDED public metadata and qparam helpers

# BEGIN NATIVE INT8 QUANTIZATION: ADDED generic SmoothQuant preprocessing


def _normalize_calibration_inputs(
    example_inputs,
    calibration_inputs,
    concrete_args,
):
    """Return complete FX argument tuples for every calibration sample."""

    samples = (
        (tuple(example_inputs),)
        if calibration_inputs is None
        else tuple(calibration_inputs)
    )
    if not samples:
        raise ValueError("calibration_inputs must contain at least one sample")

    normalized = []
    for sample in samples:
        if not isinstance(sample, (tuple, list)):
            raise TypeError(
                "Each calibration sample must be a tuple or list"
            )
        if len(sample) != len(example_inputs):
            raise ValueError(
                "Each calibration sample must match the example_inputs arity"
            )

        # FX retains placeholders for arguments specialized to their defaults.
        normalized.append(
            tuple(sample) + tuple(concrete_args.values())
        )
    return tuple(normalized)


def _is_smoothquant_norm_candidate(module):
    """Recognize a last-axis affine normalization without model-name checks."""

    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    eps = getattr(module, "eps", None)

    return (
        "norm" in module.__class__.__name__.lower()
        and isinstance(weight, torch.Tensor)
        and weight.ndim == 1
        and (
            bias is None
            or (
                isinstance(bias, torch.Tensor)
                and bias.shape == weight.shape
            )
        )
        and isinstance(eps, (int, float))
        and math.isfinite(float(eps))
        and float(eps) >= 0.0
    )


def _discover_smoothquant_groups(gm):
    """Find safe affine-norm outputs whose only users are Linear modules."""

    modules = dict(gm.named_modules())
    norm_nodes = {}

    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue

        module = modules[node.target]
        if not _is_smoothquant_norm_candidate(module):
            continue

        meta = node.meta.get("tensor_meta")
        if not isinstance(meta, TensorMetadata) or not meta.shape:
            continue
        if int(meta.shape[-1]) != int(module.weight.numel()):
            continue

        norm_nodes.setdefault(node.target, []).append(node)

    # Parameter ownership prevents the offline fold from changing a tied
    # embedding/LM-head tensor or another consumer outside the discovered group.
    parameter_owners = {}
    try:
        named_modules = gm.named_modules(remove_duplicate=False)
    except TypeError:
        named_modules = gm.named_modules()

    for module_name, module in named_modules:
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            parameter_owners.setdefault(id(weight), set()).add(module_name)

    groups = {}
    skipped = {}
    linear_owner = {}

    for norm_target, nodes in norm_nodes.items():
        linear_targets = {}
        unsafe_user = False

        for node in nodes:
            if not node.users:
                unsafe_user = True
                break

            for user in node.users:
                if (
                    user.op != "call_module"
                    or user.target not in modules
                    or not isinstance(
                        modules[user.target],
                        torch.nn.Linear,
                    )
                    or not user.args
                    or user.args[0] is not node
                ):
                    unsafe_user = True
                    break

                linear_targets[user.target] = modules[user.target]

            if unsafe_user:
                break

        if unsafe_user or not linear_targets:
            continue

        hidden_size = int(modules[norm_target].weight.numel())
        if any(
            int(linear.in_features) != hidden_size
            for linear in linear_targets.values()
        ):
            continue

        target_names = set(linear_targets)

        external_aliases = {
            owner
            for linear in linear_targets.values()
            for owner in parameter_owners.get(
                id(linear.weight),
                set(),
            )
            if owner not in target_names
        }
        norm_aliases = (
            parameter_owners.get(
                id(modules[norm_target].weight),
                set(),
            )
            - {norm_target}
        )

        if external_aliases or norm_aliases:
            skipped[norm_target] = (
                "parameter storage is shared outside the group"
            )
            continue

        conflicting = {
            target: linear_owner[target]
            for target in target_names
            if (
                target in linear_owner
                and linear_owner[target] != norm_target
            )
        }
        if conflicting:
            skipped[norm_target] = (
                "a Linear consumer belongs to another norm group"
            )
            continue

        for target in target_names:
            linear_owner[target] = norm_target

        groups[norm_target] = (
            modules[norm_target],
            tuple(linear_targets.items()),
        )

    return groups, skipped


def _smoothquant_norm_weight_offset(
    norm,
    probe_args,
    scale,
):
    """Detect direct-weight versus one-plus-weight normalization gain."""

    weight = norm.weight
    bias = getattr(norm, "bias", None)

    weight_before = weight.detach().clone()
    bias_before = (
        bias.detach().clone()
        if bias is not None
        else None
    )

    # This unusual FP32 conversion is only for proving the offline fold. It is
    # not emitted into the generated native quantized implementation.
    scale = scale.to(
        device=weight.device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        reference = norm(*probe_args).detach().float()
        expected = reference / scale

        for offset in (0.0, 1.0):
            try:
                # Standard LayerNorm/RMSNorm uses weight directly. Some RMSNorm
                # implementations, including Gemma-style variants, use 1+weight.
                adjusted = (
                    (weight_before.float() + offset) / scale
                    - offset
                )
                weight.copy_(adjusted.to(weight.dtype))

                if bias is not None:
                    bias.copy_(
                        (bias_before.float() / scale).to(bias.dtype)
                    )

                actual = norm(*probe_args).detach().float()
            finally:
                weight.copy_(weight_before)
                if bias is not None:
                    bias.copy_(bias_before)

            if torch.allclose(
                actual,
                expected,
                rtol=2.0e-4,
                atol=2.0e-5,
            ):
                return offset

    raise NotImplementedError(
        "SmoothQuant cannot prove an affine fold for "
        f"{norm.__class__.__name__}"
    )


def apply_smoothquant(
    gm,
    calibration_inputs,
    *,
    alpha=0.5,
    min_scale=1.0e-5,
):
    """Apply offline SmoothQuant to safe FX norm-to-Linear groups."""

    alpha = float(alpha)
    min_scale = float(min_scale)

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("SmoothQuant alpha must be in [0, 1]")
    if not math.isfinite(min_scale) or min_scale <= 0.0:
        raise ValueError(
            "SmoothQuant min_scale must be finite and positive"
        )

    previous_alpha = getattr(
        gm,
        "_allo_smoothquant_alpha",
        None,
    )
    if previous_alpha is not None:
        # Compare all fold parameters so a repeated call cannot silently reuse
        # incompatible quantization metadata or compound a previous fold.
        previous_min_scale = getattr(
            gm,
            "_allo_smoothquant_min_scale",
            None,
        )
        if not (
            math.isclose(float(previous_alpha), alpha)
            and previous_min_scale is not None
            and math.isclose(
                float(previous_min_scale),
                min_scale,
            )
        ):
            raise ValueError(
                "SmoothQuant was already applied with different parameters"
            )

        return {
            name: scale.clone()
            for name, scale
            in gm._allo_smoothquant_scales.items()
        }

    calibration_inputs = tuple(calibration_inputs)
    if not calibration_inputs:
        raise ValueError(
            "SmoothQuant requires at least one calibration sample"
        )

    groups, skipped = _discover_smoothquant_groups(gm)
    if not groups:
        detail = (
            f"; skipped groups: {skipped}"
            if skipped
            else ""
        )
        raise ValueError(
            "No safe norm-to-Linear SmoothQuant groups were found"
            f"{detail}"
        )

    if skipped:
        warnings.warn(
            f"SmoothQuant skipped unsafe groups: {skipped}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Activation range metadata is collected per input channel. Only folded
    # parameters, not these FP32 calibration values, reach native INT8 lowering.
    activation_max = {
        name: torch.zeros(
            norm.weight.numel(),
            dtype=torch.float32,
        )
        for name, (norm, _) in groups.items()
    }
    probe_args = {}

    def capture_output(name):
        def hook(_module, args, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    "SmoothQuant normalization output must be a Tensor"
                )

            values = (
                output.detach()
                .float()
                .reshape(-1, output.shape[-1])
            )
            activation_max[name] = torch.maximum(
                activation_max[name],
                values.abs().amax(dim=0).cpu(),
            )

            if name not in probe_args:
                # Keep one real norm input to prove the fold before modifying
                # checkpoint tensors with potentially unusual norm semantics.
                probe_args[name] = tuple(
                    value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                    for value in args
                )

        return hook

    handles = [
        norm.register_forward_hook(capture_output(name))
        for name, (norm, _) in groups.items()
    ]

    was_training = gm.training
    try:
        gm.eval()
        with torch.no_grad():
            for sample in calibration_inputs:
                if not isinstance(sample, (tuple, list)):
                    raise TypeError(
                        "Each SmoothQuant calibration sample "
                        "must be a tuple or list"
                    )
                gm(*tuple(sample))
    finally:
        for handle in handles:
            handle.remove()
        gm.train(was_training)

    # Validate every group before changing any checkpoint tensor. Otherwise an
    # unsupported later norm could leave the model only partially smoothed.
    fold_plan = []

    for name, (norm, linears) in groups.items():
        if name not in probe_args:
            raise RuntimeError(
                f"Calibration did not execute norm group {name!r}"
            )

        device = linears[0][1].weight.device
        act_max = activation_max[name].to(
            device=device,
            dtype=torch.float32,
        )

        weight_max = torch.stack(
            [
                linear.weight.detach()
                .float()
                .abs()
                .amax(dim=0)
                for _, linear in linears
            ]
        ).amax(dim=0)
        weight_max = weight_max.clamp(min=min_scale)

        # SmoothQuant scale reconciliation:
        # s_j = max|x_j|^alpha / max|W[:,j]|^(1-alpha)
        scale = (
            act_max.pow(alpha)
            / weight_max.pow(1.0 - alpha)
        ).clamp(min=min_scale)

        if not torch.isfinite(scale).all():
            raise ValueError(
                f"SmoothQuant produced a non-finite scale for {name}"
            )

        offset = _smoothquant_norm_weight_offset(
            norm,
            probe_args[name],
            scale,
        )
        fold_plan.append(
            (name, norm, linears, scale, offset)
        )

    smooth_scales = {}
    group_metadata = {}

    with torch.no_grad():
        for name, norm, linears, scale, offset in fold_plan:
            norm_scale = scale.to(
                norm.weight.device,
                dtype=torch.float32,
            )

            adjusted = (
                (norm.weight.detach().float() + offset)
                / norm_scale
                - offset
            )
            norm.weight.copy_(
                adjusted.to(norm.weight.dtype)
            )

            if getattr(norm, "bias", None) is not None:
                norm.bias.div_(
                    norm_scale.to(norm.bias.dtype)
                )

            # Multiply each shared weight once. The existing Allo path still
            # performs INT8×INT8→INT32 accumulation and INT64 requantization.
            updated_weights = set()
            for _, linear in linears:
                if id(linear.weight) in updated_weights:
                    continue

                linear.weight.mul_(
                    scale.to(linear.weight.dtype).reshape(1, -1)
                )
                updated_weights.add(id(linear.weight))

            smooth_scales[name] = scale.detach().cpu()

            # Record exact topology and norm convention for reviewability and
            # to prevent an unnoticed repeated parameter fold.
            group_metadata[name] = {
                "linear_targets": tuple(
                    target for target, _ in linears
                ),
                "weight_offset": offset,
            }

    # Persist preprocessing metadata on the FX module. This is offline metadata;
    # it does not introduce a second graph representation or runtime operation.
    gm._allo_smoothquant_alpha = alpha
    gm._allo_smoothquant_min_scale = min_scale
    gm._allo_smoothquant_scales = smooth_scales
    gm._allo_smoothquant_groups = group_metadata
    gm._allo_smoothquant_skipped = skipped

    return {
        name: scale.clone()
        for name, scale in smooth_scales.items()
    }


# END NATIVE INT8 QUANTIZATION: ADDED generic SmoothQuant preprocessing

def from_pytorch(
    model,
    example_inputs,
    leaf_modules=None,
    verbose=False,
    enable_tensor=False,
    target="llvm",
    mode="csim",
    project="top.prj",
    op_dtypes=None,
    weights_as_args=False,
    # BEGIN NATIVE INT8 QUANTIZATION: ADDED public options
    quant_config=None,
    qdq_lowering_mode="early",
    calibration_inputs=None,
    # END NATIVE INT8 QUANTIZATION: ADDED public options
):
    sig = inspect.signature(model.forward)
    input_names = [
        p.name for i, p in enumerate(sig.parameters.values()) if i < len(example_inputs)
    ]
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    args = []
    args += example_inputs
    for item in concrete_args.values():
        args.append(item)

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED reusable calibration batches
    calibration_args = _normalize_calibration_inputs(
        example_inputs,
        calibration_inputs,
        concrete_args,
    )
    # END NATIVE INT8 QUANTIZATION: ADDED reusable calibration batches

    tracer = AlloTracer(model, concrete_args=concrete_args, leaf_modules=leaf_modules)
    graph = tracer.trace()
    name = (
        model.__class__.__name__
        if isinstance(model, torch.nn.Module)
        else model.__name__
    )
    gm = GraphModule(tracer.root, graph, name)
    ShapeProp(gm).propagate(*args)

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED frontend SmoothQuant dispatch
    # Perform the offline fold before TorchBuilder chooses activation qparams.
    smoothquant_alpha = getattr(
        quant_config,
        "smoothquant_alpha",
        None,
    )

    if smoothquant_alpha is not None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "SmoothQuant requires a torch.nn.Module model"
            )

        previous_alpha = getattr(
            model,
            "_allo_smoothquant_alpha",
            None,
        )

        if previous_alpha is None:
            scales = apply_smoothquant(
                gm,
                calibration_args,
                alpha=smoothquant_alpha,
                min_scale=quant_config.smoothquant_min_scale,
            )

            # GraphModule normally shares the original module objects. Loading
            # state back also covers FX versions that copy module state.
            if model is not gm:
                model.load_state_dict(
                    gm.state_dict(),
                    strict=False,
                )

            # Preserve offline scale/topology metadata on the public model.
            model._allo_smoothquant_alpha = float(
                smoothquant_alpha
            )
            model._allo_smoothquant_min_scale = float(
                quant_config.smoothquant_min_scale
            )
            model._allo_smoothquant_scales = scales
            model._allo_smoothquant_groups = (
                gm._allo_smoothquant_groups
            )
            model._allo_smoothquant_skipped = (
                gm._allo_smoothquant_skipped
            )

        elif not (
            math.isclose(
                float(previous_alpha),
                float(smoothquant_alpha),
            )
            and math.isclose(
                float(
                    getattr(
                        model,
                        "_allo_smoothquant_min_scale",
                        -1.0,
                    )
                ),
                float(quant_config.smoothquant_min_scale),
            )
        ):
            raise ValueError(
                "SmoothQuant was already applied with "
                "different parameters"
            )
    # END NATIVE INT8 QUANTIZATION: ADDED frontend SmoothQuant dispatch

    if verbose:
        print(str(gm.graph) + "\n")
    global_vars = {}
    for pymod in (types,):
        global_vars.update({item[0]: item[1] for item in inspect.getmembers(pymod)})
    global_vars.update({"dsl": dsl, "nn": nn})

    # Only add weights to global_vars if not passing as arguments
    if not weights_as_args:
        for name, param in gm.named_parameters():
            new_name = "g_" + name.replace(".", "_")
            global_vars.update({new_name: param.detach().numpy()})
        for name, buf in gm.named_buffers():
            new_name = "gb_" + name.replace(".", "_")
            global_vars.update({new_name: buf.detach().numpy()})

    builder = TorchBuilder(gm, example_inputs, leaf_modules, op_dtypes, weights_as_args, calibration_inputs=calibration_args,)
    # BEGIN NATIVE INT8 QUANTIZATION: ADDED opt-in builder configuration
    if quant_config is not None or qdq_lowering_mode != "early":
        builder.configure_quantization(quant_config, qdq_lowering_mode)
    # END NATIVE INT8 QUANTIZATION: ADDED opt-in builder configuration
    code = builder.build()
    if verbose:
        print(code)
    # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer constants
    if builder.quantization_enabled:
        global_vars["roundeven"] = dsl.roundeven
    global_vars.update(builder.extra_globals)
    # END NATIVE INT8 QUANTIZATION: ADDED integer constants
    # register any synthetic dtype symbols required by the builder
    if getattr(builder, "extra_types", None):
        global_vars.update(builder.extra_types)
    s = customize(code, global_vars=global_vars, enable_tensor=enable_tensor)
    # composition
    for func, idx, inst in builder.composition:
        s.compose(getattr(nn, func), id=idx, instantiate=inst)
    if verbose:
        print(s.module)
    if target == "mlir":
        return s
    mod = s.build(target=target, mode=mode, project=project)

    # If weights are passed as arguments, create a wrapper function
    if weights_as_args:
        # BEGIN NATIVE INT8 QUANTIZATION: CHANGED runtime arguments use the
        # exact integer arrays prepared by TorchBuilder instead of the original
        # floating-point checkpoint arrays.
        weight_data = []
        weight_names = []
        for name, param in builder.named_params.items():
            new_name = name.replace(".", "_")
            weight_names.append(new_name)
            weight_data.append(
                builder.runtime_param_data.get(new_name, param.detach().numpy())
            )
        for name, buf in builder.named_buffers.items():
            new_name = name.replace(".", "_")
            weight_names.append(new_name)
            weight_data.append(
                builder.runtime_param_data.get(new_name, buf.detach().numpy())
            )
        for name, (_, _, data) in builder.runtime_aux_params.items():
            weight_names.append(name)
            weight_data.append(data)
        # END NATIVE INT8 QUANTIZATION: CHANGED runtime argument preparation

        # Create a wrapper that accepts inputs and weights
        def wrapped_forward(*args):
            def flatten_inputs(values):
                flattened = []
                for value in values:
                    if isinstance(value, (list, tuple)):
                        flattened.extend(value)
                    else:
                        flattened.append(value)
                return flattened

            # If only inputs are provided, automatically add weights
            if len(args) == len(example_inputs):
                # User only passed inputs, automatically add weights
                return mod(*flatten_inputs(args), *weight_data)
            if len(args) == len(builder.input_args):
                return mod(*args, *weight_data)
            if len(args) > len(example_inputs):
                logical_inputs = flatten_inputs(args[: len(example_inputs)])
                if len(logical_inputs) == len(builder.input_args):
                    return mod(*logical_inputs, *args[len(example_inputs) :])
            # User passed both inputs and weights
            num_inputs = len(builder.input_args)
            inputs = args[:num_inputs]
            weights = args[num_inputs:]
            return mod(*inputs, *weights)

        # Attach weight data to the wrapper for convenience
        wrapped_forward.weight_data = weight_data
        wrapped_forward.weight_names = tuple(weight_names)
        wrapped_forward.num_inputs = len(builder.input_args)
        return wrapped_forward

    return mod


def get_var_name(node):
    return node.name if isinstance(node, fx.Node) else node


class TorchBuilder:
    def __init__(
        self,
        gm,
        example_inputs,
        leaf_modules=None,
        op_dtypes=None,
        weights_as_args=False,
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED direct builder options
        quant_config=None,
        qdq_lowering_mode="early",
        calibration_inputs=None,
        # END NATIVE INT8 QUANTIZATION: ADDED direct builder options
    ):
        self.gm = gm
        self.code = []
        self.input_names = []
        self.input_shapes = []
        self.example_inputs = example_inputs
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED multi-sample calibration state
        # These are complete FX call tuples. example_inputs still controls the
        # generated function signature and fixed tensor shapes.
        self.calibration_inputs = (
            (tuple(example_inputs),)
            if calibration_inputs is None
            else tuple(
                tuple(sample)
                for sample in calibration_inputs
            )
        )
        if not self.calibration_inputs:
            raise ValueError(
                "calibration_inputs must contain at least one sample"
            )
        # END NATIVE INT8 QUANTIZATION: ADDED multi-sample calibration state
        self.leaf_modules = leaf_modules
        self.input_args = []
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED tied-parameter alias metadata
        self.named_params = dict(gm.named_parameters())
        self.parameter_aliases = {}
        try:
            all_named_params = gm.named_parameters(remove_duplicate=False)
        except TypeError:
            all_named_params = (
                (name, value)
                for name, value in gm.state_dict(keep_vars=True).items()
                if isinstance(value, torch.nn.Parameter)
            )
        canonical_parameters = {
            id(param): name for name, param in self.named_params.items()
        }
        for name, param in all_named_params:
            canonical = canonical_parameters.get(id(param), name)
            self.parameter_aliases[name.replace(".", "_")] = canonical.replace(
                ".", "_"
            )
        self.named_buffers = dict(gm.named_buffers())
        # END NATIVE INT8 QUANTIZATION: ADDED tied-parameter alias metadata
        self.subfunctions = []
        self.output = []
        self.composition = []
        self.unique_id = {}
        # operator dtype preferences; values can be strings (e.g., "float16") or AlloType objects (e.g., types.int8)
        self.op_dtypes = op_dtypes or {}
        # mapping from parameter/buffer var name (underscored) to dtype name string used in code
        self.param_dtypes = {}
        # synthetic dtype symbols to inject into global_vars: name -> AlloType
        self.extra_types = {}
        # whether to pass weights as function arguments instead of global constants
        self.weights_as_args = weights_as_args
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED builder state
        self.runtime_param_data = {}
        self.runtime_aux_params = {}
        self.configure_quantization(quant_config, qdq_lowering_mode)
        # END NATIVE INT8 QUANTIZATION: ADDED builder state

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED TorchBuilder integer helpers

    def configure_quantization(self, quant_config=None, qdq_lowering_mode="early"):
        if quant_config is not None and not isinstance(
            quant_config, QuantizationConfig
        ):
            raise TypeError("quant_config must be QuantizationConfig or None")
        if qdq_lowering_mode not in {"early", "delayed", "fused"}:
            raise ValueError(f"Unknown qdq_lowering_mode: {qdq_lowering_mode}")
        self.quant_config = quant_config
        self.qdq_lowering_mode = qdq_lowering_mode
        self.delay_qdq_lowering = qdq_lowering_mode in {"delayed", "fused"}
        self.quantization_enabled = quant_config is not None or any(
            node.op == "call_function" and node.target is torch.quantize_per_tensor
            for node in self.gm.graph.nodes
        )
        self.tmp_id = {}
        self.extra_globals = {}
        self.synthetic_params = {}
        self.quant_info = {}
        self.materialized_quant_map = {}
        self.calibration_ranges = {}
        self.integer_values = set()
        if self.quantization_enabled:
            for value in self.op_dtypes.values():
                values = value if isinstance(value, (tuple, list)) else (value,)
                if isinstance(value, (tuple, list)) and len(value) != 3:
                    raise ValueError(
                        "Linear dtype specification must contain exactly three entries"
                    )
                for item in values:
                    if isinstance(item, str) and not hasattr(types, item):
                        raise ValueError(f"Unknown Allo dtype: {item!r}")
        if quant_config is not None:
            self.calibrate_quantization()
        return self

    def _new_tmp_name(self, base):
        index = self.tmp_id.get(base, 0)
        self.tmp_id[base] = index + 1
        return f"{base}_{index}"

    def _emit_scalar_constant(self, value, dtype_name, prefix):
        result = self._new_tmp_name(prefix)

        # BEGIN NATIVE INT8 QUANTIZATION: CHANGED materialize wide int64
        # constants from signed-i32-safe limbs because Allo initially infers
        # integer literals as i32 before applying the destination annotation.
        if dtype_name == "int64":
            integer_value = int(value)
            if not -(1 << 31) <= integer_value <= (1 << 31) - 1:
                if not -(1 << 63) <= integer_value <= (1 << 63) - 1:
                    raise OverflowError("int64 scalar constant is out of range")

                magnitude = abs(integer_value)
                limbs = []
                while magnitude:
                    limbs.append(magnitude & ((1 << 30) - 1))
                    magnitude >>= 30

                negative = integer_value < 0
                initial = -limbs[-1] if negative else limbs[-1]
                operation = "-" if negative else "+"

                self.code.append(f"{result}: int64 = {initial}")
                for limb in reversed(limbs[:-1]):
                    self.code.append(
                        f"{result} = ({result} << 30) {operation} {limb}"
                    )
                return result
        # END NATIVE INT8 QUANTIZATION: CHANGED wide int64 materialization

        self.code.append(f"{result}: {dtype_name} = {value}")
        return result

    def get_quant_info_map(self):
        return self.quant_info

    def get_materialized_quant_map(self):
        return self.materialized_quant_map

    def record_quant_info(self, node, info):
        self.quant_info[node] = info
        if isinstance(node, fx.Node):
            self.quant_info[node.name] = info

    def lookup_quant_info(self, node):
        try:
            info = self.quant_info.get(node)
        except TypeError:
            info = None
        return (
            self.quant_info.get(node.name)
            if info is None and isinstance(node, fx.Node)
            else info
        )

    def _dtype(self, dtype):
        torch_module = globals().get("torch")
        if torch_module is not None and dtype is torch_module.qint8:
            dtype = types.int8
        elif torch_module is not None and dtype is torch_module.quint8:
            dtype = types.uint8
        elif dtype is np.int8 or dtype == np.dtype(np.int8):
            dtype = types.int8
        elif dtype is np.uint8 or dtype == np.dtype(np.uint8):
            dtype = types.uint8
        elif isinstance(dtype, str):
            dtype = getattr(types, dtype, None)
        if not isinstance(dtype, AlloType):
            raise NotImplementedError(f"Unsupported quantized dtype: {dtype}")
        name = self._find_types_module_symbol(dtype)
        name = name if name is not None else self._literal_for_allotype(dtype)
        numpy_dtype = None
        if dtype.__class__.__name__ == "Int":
            numpy_dtype = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}.get(
                dtype.bits
            )
        elif dtype.__class__.__name__ == "UInt":
            numpy_dtype = {
                8: np.uint8,
                16: np.uint16,
                32: np.uint32,
                64: np.uint64,
            }.get(dtype.bits)
        return name, dtype, numpy_dtype

    @staticmethod
    def _is_integer(dtype, bits):
        return (
            isinstance(dtype, AlloType)
            and dtype.__class__.__name__ in {"Int", "UInt"}
            and dtype.bits == bits
        )

    def _resolve_fx_scalar(self, value):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return value.item() if hasattr(value, "item") else value
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return value.item()
        if isinstance(value, fx.Node) and value.op == "get_attr":
            resolved = self.gm
            for atom in value.target.split("."):
                resolved = getattr(resolved, atom)
            return self._resolve_fx_scalar(resolved)
        if isinstance(value, fx.Node) and value.meta.get("val") is not None:
            return self._resolve_fx_scalar(value.meta["val"])
        raise NotImplementedError(f"Unable to resolve quantization scalar: {value}")

    def calibrate_quantization(self):
        ranges = self.calibration_ranges

        class RangeInterpreter(fx.Interpreter):
            def run_node(self, n):
                result = super().run_node(n)
                if isinstance(result, torch.Tensor) and result.numel() > 0:
                    tensor = (
                        result.dequantize() if result.is_quantized else result
                    ).detach()
                    current = (
                        float(tensor.amin()),
                        float(tensor.amax()),
                    )
                    previous = ranges.get(n)

                    # NATIVE INT8 QUANTIZATION: CHANGED aggregate extrema
                    # across every representative calibration batch.
                    ranges[n] = (
                        current
                        if previous is None
                        else (
                            min(previous[0], current[0]),
                            max(previous[1], current[1]),
                        )
                    )

                    ranges[n.name] = ranges[n]
                return result

        with torch.no_grad():
            # NATIVE INT8 QUANTIZATION: CHANGED aggregate ranges across every
            # representative input batch instead of only example_inputs.
            for sample in self.calibration_inputs:
                RangeInterpreter(self.gm).run(*sample)
        return ranges

    def _activation_quant_info(self, node):
        qmin, qmax = get_qrange(self.quant_config.activation_dtype)
        value_range = self.calibration_ranges.get(
            node, self.calibration_ranges.get(get_var_name(node))
        )
        if value_range is None:
            raise RuntimeError(f"Missing calibration range for {get_var_name(node)}")
        scale, zero_point = choose_qparams(
            *value_range,
            qmin,
            qmax,
            symmetric=self.quant_config.calibration_method == "symmetric_minmax",
        )
        return QuantInfo(
            scale, zero_point, self.quant_config.activation_dtype, qmin, qmax
        )

    def _quant_info_from_node(self, node):
        scale = node.kwargs.get("scale", node.args[1] if len(node.args) > 1 else None)
        zero_point = node.kwargs.get(
            "zero_point", node.args[2] if len(node.args) > 2 else None
        )
        dtype = node.kwargs.get("dtype", node.args[3] if len(node.args) > 3 else None)
        qmin, qmax = get_qrange(dtype)
        return QuantInfo(
            float(self._resolve_fx_scalar(scale)),
            int(self._resolve_fx_scalar(zero_point)),
            dtype,
            qmin,
            qmax,
        )

    def _explicit_output_quant_info(self, node):
        for user in node.users:
            if user.op == "call_function" and user.target is torch.quantize_per_tensor:
                return self._quant_info_from_node(user)
        return None

    @staticmethod
    def _shape(node):
        meta = node.meta.get("tensor_meta")
        if isinstance(meta, TensorMetadata):
            return tuple(meta.shape)
        raise NotImplementedError(f"Missing tensor shape for {node.name}")

    @staticmethod
    def _is_output_boundary(node):
        return any(user.op == "output" for user in node.users)

    def _emit_cast(self, value, dtype_name, shape, prefix):
        result = self._new_tmp_name(prefix)
        dims = ", ".join(str(dim) for dim in shape)
        indices = ", ".join(f"i{axis}" for axis in range(len(shape)))
        self.code += [
            f"{result}: {dtype_name}[{dims}]",
            f'for {indices} in dsl.grid({dims}, name="{result}_cast"):',
            f"    {result}[{indices}] = {value}[{indices}]",
        ]
        return result

    def materialize_quant_info_clamped(self, node):
        key = node.name + "_clamped"
        if key in self.materialized_quant_map:
            return self.materialized_quant_map[key]
        info, shape = self.lookup_quant_info(node), self._shape(node)
        dims = ", ".join(str(dim) for dim in shape)
        indices = ", ".join(f"i{axis}" for axis in range(len(shape)))
        values = self._new_tmp_name("qdq_min")
        self.code += [
            f"{values}: int32[{dims}]",
            f'for {indices} in dsl.grid({dims}, name="{values}_quantize"):',
            (
                f"    {values}[{indices}] = min(max(roundeven("
                f"{node.name}[{indices}] / {repr(float(info.scale))}) + "
                f"{info.zero_point}, {info.qmin}), {info.qmax})"
            ),
        ]
        dtype_name, _, _ = self._dtype(info.dtype)
        result = self._emit_cast(values, dtype_name, shape, "qdq_int")
        self.materialized_quant_map[key] = result
        return result

    def materialize_quant_info(self, node):
        key = node.name + "_dequantized"
        if key in self.materialized_quant_map:
            return self.materialized_quant_map[key]
        info = self.lookup_quant_info(node)
        value = self._emit_cast(
            self.materialize_quant_info_clamped(node),
            "float32",
            self._shape(node),
            "qdq_float",
        )
        if info.zero_point != 0:
            centered = self._new_tmp_name("qdq_subz")
            self.code.append(f"{centered} = {value} - {float(info.zero_point)}")
            value = centered
        if info.scale != 1:
            scaled = self._new_tmp_name("qdq_mul")
            self.code.append(f"{scaled} = {value} * {repr(float(info.scale))}")
            value = scaled
        self.materialized_quant_map[key] = value
        return value

    def _prepare_quantized_passthrough(self, source, destination):
        info = self.lookup_quant_info(source)
        if info is None or not (
            self.delay_qdq_lowering
            or self.quant_config is not None
            or source in self.integer_values
        ):
            return source.name, None
        value = (
            source.name
            if source in self.integer_values
            else self.materialize_quant_info_clamped(source)
        )
        self.record_quant_info(destination, info)
        self.materialized_quant_map[destination.name + "_clamped"] = destination.name
        self.integer_values.add(destination)
        return value, info

    def _quantized_value_name(self, node):
        if node in self.integer_values:
            return node.name
        key = node.name + "_clamped"
        if key in self.materialized_quant_map:
            return self.materialized_quant_map[key]
        info = self.lookup_quant_info(node)
        if self.quant_config is None:
            return self.materialize_quant_info_clamped(node)
        shape = self._shape(node)
        kernel, name_id = f"quantize{len(shape)}d", self.get_unique_id(
            f"quantize{len(shape)}d"
        )
        dtype_name, dtype, _ = self._dtype(info.dtype)
        self.composition.append((kernel, name_id, [float32, dtype, *shape]))
        result = self._new_tmp_name(node.name + "_q")
        dims = ", ".join(str(dim) for dim in shape)
        self.code.append(
            f"{result} = nn.{kernel}[float32, {dtype_name}, {dims}, "
            f'"{name_id}"]({node.name}, {repr(float(info.scale))}, '
            f"{info.zero_point}, {info.qmin}, {info.qmax})"
        )
        self.materialized_quant_map[key] = result
        return result

    def _requantize_value(self, node, value, src_info, dst_info, result):
        shape = self._shape(node)
        if len(shape) not in {2, 3}:
            raise NotImplementedError("Requantization supports rank-2/rank-3")
        _, src_dtype, _ = self._dtype(src_info.dtype)
        if not self._is_integer(src_dtype, 32):
            dims = ", ".join(str(dim) for dim in shape)
            indices = ", ".join(f"i{axis}" for axis in range(len(shape)))
            centered = self._new_tmp_name("requant_centered")
            self.code += [
                f"{centered}: int32[{dims}]",
                f'for {indices} in dsl.grid({dims}, name="{centered}_cast"):',
                (
                    f"    {centered}[{indices}] = {value}[{indices}] "
                    f"- {src_info.zero_point}"
                ),
            ]
            value, input_zero_point = centered, 0
        else:
            input_zero_point = src_info.zero_point
        kernel, name_id = f"requantize{len(shape)}d", self.get_unique_id(
            f"requantize{len(shape)}d"
        )
        dst_name, dst_dtype, _ = self._dtype(dst_info.dtype)
        multiplier, shift = approximate_multiplier_shift(
            src_info.scale / dst_info.scale
        )
        multiplier = self._emit_scalar_constant(
            multiplier, "int64", "requant_multiplier"
        )
        self.composition.append(
            (kernel, name_id, [types.int32, types.int64, dst_dtype, *shape])
        )
        dims = ", ".join(str(dim) for dim in shape)
        return (
            f"{result} = nn.{kernel}[int32, int64, {dst_name}, {dims}, "
            f'"{name_id}"]({value}, {multiplier}, {shift}, '
            f"{input_zero_point}, {dst_info.zero_point}, "
            f"{dst_info.qmin}, {dst_info.qmax})"
        )

    def _dequantize_value(self, node, value, info, result):
        shape = self._shape(node)
        kernel, name_id = f"dequantize{len(shape)}d", self.get_unique_id(
            f"dequantize{len(shape)}d"
        )
        dtype_name, dtype, _ = self._dtype(info.dtype)
        self.composition.append((kernel, name_id, [dtype, float32, *shape]))
        dims = ", ".join(str(dim) for dim in shape)
        return (
            f"{result} = nn.{kernel}[{dtype_name}, float32, {dims}, "
            f'"{name_id}"]({value}, {repr(float(info.scale))}, '
            f"{info.zero_point})"
        )

    def _native_linear_dtypes(self, module_key):
        spec = self.op_dtypes.get(module_key, self.op_dtypes.get("linear"))
        if spec is None and self.quant_config is not None:
            spec = (
                self.quant_config.activation_dtype,
                self.quant_config.weight_dtype,
                self.quant_config.accumulator_dtype,
            )
        if spec is None:
            return None
        if not isinstance(spec, (tuple, list)) or len(spec) != 3:
            raise ValueError(
                "Linear dtype specification must contain exactly three entries"
            )
        return tuple(self._dtype(dtype) for dtype in spec)

    def _build_native_linear(self, node, has_bias, dtypes):
        input_node = node.args[0]
        input_info = self.lookup_quant_info(input_node)
        if input_info is None:
            return None

        (
            (x_name, x_type, _),
            (w_name, w_type, w_numpy),
            (acc_name, acc_type, acc_numpy),
        ) = dtypes

        if not (
            self._is_integer(x_type, 8)
            and self._is_integer(w_type, 8)
            and self._is_integer(acc_type, 32)
        ):
            return None

        target = node.target.replace(".", "_")
        requested_weight_name = target + "_weight"
        weight_name = self._module_parameter_symbol(node.target)
        weight = self._module_parameter(node.target).detach().numpy()
        wmin, wmax = get_qrange(w_type)

        # BEGIN NATIVE INT8 QUANTIZATION: CHANGED optional per-output-channel
        # weight quantization. A tied Linear alias, such as the tied LM head,
        # remains per-tensor so it can safely reuse the embedding table.
        per_channel_weights = (
            self.quant_config is not None
            and self.quant_config.weight_granularity == "per_channel"
            and weight_name == requested_weight_name
        )

        if self.quant_config is None:
            weight_scale = 1.0
            weight_source = weight * input_info.scale
            accumulator_scale = 1.0
        elif per_channel_weights:
            weight_scale = np.asarray(
                [
                    choose_qparams(
                        float(row.min()),
                        float(row.max()),
                        wmin,
                        wmax,
                    )[0]
                    for row in weight
                ],
                dtype=np.float64,
            )
            weight_source = weight
            accumulator_scale = input_info.scale * weight_scale
        else:
            weight_scale, _ = choose_qparams(
                float(weight.min()),
                float(weight.max()),
                wmin,
                wmax,
            )
            weight_source = weight
            accumulator_scale = input_info.scale * weight_scale

        weight_divisor = (
            weight_scale[:, None]
            if per_channel_weights
            else weight_scale
        )
        # END NATIVE INT8 QUANTIZATION: CHANGED weight scale selection

        weight_codes = np.clip(
            np.rint(weight_source / weight_divisor),
            wmin,
            wmax,
        ).astype(w_numpy)

        self._record_runtime_array(weight_name, weight_codes)
        self.param_dtypes[weight_name] = w_name

        amin, amax = get_qrange(acc_type)
        output_features, reduction = weight.shape

        bias_codes = -input_info.zero_point * np.asarray(
            weight_codes,
            dtype=np.int64,
        ).sum(axis=1)

        if has_bias:
            bias = self._module_parameter(
                node.target,
                "bias",
            ).detach().numpy()
            bias_codes += np.rint(
                bias / accumulator_scale
            ).astype(np.int64)

        bias_codes = np.clip(
            bias_codes,
            amin,
            amax,
        ).astype(acc_numpy)

        if has_bias:
            bias_name = target + "_bias"
            self._record_runtime_array(
                bias_name,
                bias_codes,
            )
            self.param_dtypes[bias_name] = acc_name
        else:
            bias_name = target + "_zero_bias"
            global_name = "g_" + bias_name

            if self.weights_as_args:
                self.runtime_aux_params[bias_name] = (
                    acc_name,
                    (output_features,),
                    bias_codes,
                )
            else:
                self.extra_globals[global_name] = bias_codes
                self.synthetic_params[bias_name] = (
                    acc_name,
                    (output_features,),
                    global_name,
                )

        shape = self._shape(node)

        if len(shape) == 2:
            kernel = "linear2d"
            dimensions = (
                shape[0],
                shape[1],
                reduction,
            )
        elif len(shape) == 3:
            kernel = "linear3d"
            dimensions = (
                shape[0],
                shape[1],
                reduction,
                shape[2],
            )
        else:
            raise NotImplementedError(
                "Integer Linear supports rank-2/rank-3"
            )

        name_id = self.get_unique_id("linear")
        self.composition.append(
            (
                kernel,
                name_id,
                [
                    x_type,
                    w_type,
                    acc_type,
                    *dimensions,
                ],
            )
        )

        template = ", ".join(
            [
                x_name,
                w_name,
                acc_name,
                *(str(value) for value in dimensions),
            ]
        )

        accumulator = (
            node.name + "_acc"
            if self.quant_config is not None
            else node.name
        )

        linear = (
            f'{accumulator} = nn.{kernel}[{template}, "{name_id}"]('
            f"{self._quantized_value_name(input_node)}, "
            f"{weight_name}, {bias_name})"
        )

        if self.quant_config is None:
            accumulator_info = QuantInfo(
                float(accumulator_scale),
                0,
                acc_type,
                amin,
                amax,
            )
            self.record_quant_info(node, accumulator_info)
            self.integer_values.add(node)
            self.materialized_quant_map[
                node.name + "_clamped"
            ] = node.name
            return linear

        self.code.append(linear)
        output_info = self._activation_quant_info(node)

        # BEGIN NATIVE INT8 QUANTIZATION: ADDED per-channel requantization
        if per_channel_weights:
            multiplier_shift = [
                approximate_multiplier_shift(
                    float(scale) / output_info.scale
                )
                for scale in accumulator_scale
            ]

            multipliers = np.asarray(
                [item[0] for item in multiplier_shift],
                dtype=np.int64,
            )
            shifts = np.asarray(
                [item[1] for item in multiplier_shift],
                dtype=np.int32,
            )

            multiplier_name = self._record_synthetic_array(
                node.name + "_requant_multipliers",
                "int64",
                multipliers,
            )
            shift_name = self._record_synthetic_array(
                node.name + "_requant_shifts",
                "int32",
                shifts,
            )

            requantize_kernel = (
                f"requantize_per_channel{len(shape)}d"
            )
            requantize_id = self.get_unique_id(
                requantize_kernel
            )

            self.composition.append(
                (
                    requantize_kernel,
                    requantize_id,
                    [
                        acc_type,
                        types.int64,
                        self._dtype(output_info.dtype)[1],
                        *shape,
                    ],
                )
            )

            out_name = self._dtype(output_info.dtype)[0]
            dims = ", ".join(str(dim) for dim in shape)

            result = (
                f"{node.name} = nn.{requantize_kernel}["
                f"{acc_name}, int64, {out_name}, {dims}, "
                f'"{requantize_id}"]('
                f"{accumulator}, {multiplier_name}, {shift_name}, "
                f"0, {output_info.zero_point}, "
                f"{output_info.qmin}, {output_info.qmax})"
            )

            self.record_quant_info(node, output_info)
            self.integer_values.add(node)
            self.materialized_quant_map[
                node.name + "_clamped"
            ] = node.name
            return result
        # END NATIVE INT8 QUANTIZATION: ADDED per-channel requantization

        accumulator_info = QuantInfo(
            float(accumulator_scale),
            0,
            acc_type,
            amin,
            amax,
        )

        requantized = self._requantize_value(
            node,
            accumulator,
            accumulator_info,
            output_info,
            node.name,
        )

        self.record_quant_info(node, output_info)
        self.integer_values.add(node)
        self.materialized_quant_map[
            node.name + "_clamped"
        ] = node.name
        return requantized

    def _build_quantized_add(self, node):
        lhs, rhs = node.args[:2]
        lhs_info, rhs_info = self.lookup_quant_info(lhs), self.lookup_quant_info(rhs)
        if lhs_info is None or rhs_info is None:
            return None
        if not (
            self.quant_config is not None
            or self.qdq_lowering_mode == "fused"
            or lhs in self.integer_values
            or rhs in self.integer_values
        ):
            return None
        output_info = self._explicit_output_quant_info(node)
        if output_info is None and self.quant_config is not None:
            output_info = self._activation_quant_info(node)
        if output_info is None:
            output_info = QuantInfo(
                max(lhs_info.scale, rhs_info.scale),
                0,
                lhs_info.dtype,
                lhs_info.qmin,
                lhs_info.qmax,
            )
        shape = self._shape(node)
        kernel, name_id = f"qadd{len(shape)}d", self.get_unique_id(f"qadd{len(shape)}d")
        lhs_name, lhs_type, _ = self._dtype(lhs_info.dtype)
        rhs_name, rhs_type, _ = self._dtype(rhs_info.dtype)
        out_name, out_type, _ = self._dtype(output_info.dtype)
        lm, ls = approximate_multiplier_shift(lhs_info.scale / output_info.scale)
        rm, rs = approximate_multiplier_shift(rhs_info.scale / output_info.scale)
        shift = max(ls, rs)
        lm <<= shift - ls
        rm <<= shift - rs
        lhs_value = self._quantized_value_name(lhs)
        rhs_value = self._quantized_value_name(rhs)
        lm = self._emit_scalar_constant(lm, "int64", "qadd_lhs_multiplier")
        rm = self._emit_scalar_constant(rm, "int64", "qadd_rhs_multiplier")
        self.composition.append(
            (kernel, name_id, [lhs_type, rhs_type, types.int64, out_type, *shape])
        )
        self.record_quant_info(node, output_info)
        self.integer_values.add(node)
        self.materialized_quant_map[node.name + "_clamped"] = node.name
        dims = ", ".join(str(dim) for dim in shape)
        return (
            f"{node.name} = nn.{kernel}[{lhs_name}, {rhs_name}, int64, "
            f'{out_name}, {dims}, "{name_id}"]('
            f"{lhs_value}, {rhs_value}, {lm}, {rm}, {shift}, "
            f"{lhs_info.zero_point}, {rhs_info.zero_point}, "
            f"{output_info.zero_point}, {output_info.qmin}, {output_info.qmax})"
        )

    def _build_quantized_relu(self, node):
        source, info = node.args[0], self.lookup_quant_info(node.args[0])
        if info is None or not (
            self.quant_config is not None
            or self.qdq_lowering_mode == "fused"
            or source in self.integer_values
        ):
            return None
        shape = self._shape(node)
        kernel, name_id = f"relu{len(shape)}d", self.get_unique_id(f"relu{len(shape)}d")
        if info.zero_point == 0:
            dtype_name, dtype, _ = self._dtype(info.dtype)
            value = self._quantized_value_name(source)
            self.record_quant_info(node, info)
            self.integer_values.add(node)
            self.materialized_quant_map[node.name + "_clamped"] = node.name
        else:
            dtype_name, dtype = "float32", float32
            if source in self.integer_values:
                value = self._new_tmp_name(source.name + "_float")
                self.code.append(
                    self._dequantize_value(source, source.name, info, value)
                )
            else:
                value = self.materialize_quant_info(source)
        dims = ", ".join(str(dim) for dim in shape)
        self.composition.append((kernel, name_id, [dtype, *shape]))
        return (
            f"{node.name} = nn.{kernel}[{dtype_name}, {dims}, " f'"{name_id}"]({value})'
        )

    def _quantize_module_weight(self, module_target, dtype=None):
        dtype = self.quant_config.weight_dtype if dtype is None else dtype
        dtype_name, dtype_obj, numpy_dtype = self._dtype(dtype)
        qmin, qmax = get_qrange(dtype_obj)
        weight = self._module_parameter(module_target).detach().numpy()
        scale, zero_point = choose_qparams(
            float(weight.min()), float(weight.max()), qmin, qmax, symmetric=True
        )
        codes = np.clip(np.rint(weight / scale) + zero_point, qmin, qmax).astype(
            numpy_dtype
        )
        name = self._module_parameter_symbol(module_target)
        self.param_dtypes[name] = dtype_name
        self._record_runtime_array(name, codes)
        return name, codes, QuantInfo(scale, zero_point, dtype_obj, qmin, qmax)

    def _record_integer_result(self, node, info):
        self.record_quant_info(node, info)
        self.integer_values.add(node)
        self.materialized_quant_map[node.name + "_clamped"] = node.name

    def _build_quantized_mul(self, node):
        lhs, rhs = node.args[:2]
        lhs_info = self.lookup_quant_info(lhs)
        rhs_info = self.lookup_quant_info(rhs)
        output_info = (
            self._activation_quant_info(node) if self.quant_config is not None else None
        )
        if output_info is None:
            return None

        if lhs_info is not None and rhs_info is not None:
            shape = self._shape(node)
            if len(shape) != 3:
                raise NotImplementedError("Integer tensor multiply supports rank-3")
            lhs_name, lhs_type, _ = self._dtype(lhs_info.dtype)
            rhs_name, rhs_type, _ = self._dtype(rhs_info.dtype)
            out_name, out_type, _ = self._dtype(output_info.dtype)
            multiplier, shift = approximate_multiplier_shift(
                lhs_info.scale * rhs_info.scale / output_info.scale
            )
            multiplier = self._emit_scalar_constant(
                multiplier, "int64", "qmul_multiplier"
            )
            name_id = self.get_unique_id("qmul3d")
            self.composition.append(
                (
                    "qmul3d",
                    name_id,
                    [
                        lhs_type,
                        rhs_type,
                        types.int64,
                        out_type,
                        *shape,
                    ],
                )
            )
            self._record_integer_result(node, output_info)
            B, L, D = shape
            return (
                f"{node.name} = nn.qmul3d[{lhs_name}, {rhs_name}, int64, "
                f'{out_name}, {B}, {L}, {D}, "{name_id}"]('
                f"{self._quantized_value_name(lhs)}, "
                f"{self._quantized_value_name(rhs)}, {multiplier}, {shift}, "
                f"{lhs_info.zero_point}, {rhs_info.zero_point}, "
                f"{output_info.zero_point}, {output_info.qmin}, {output_info.qmax})"
            )

        source, scalar, source_info = None, None, None
        if lhs_info is not None and isinstance(rhs, (int, float)):
            source, scalar, source_info = lhs, float(rhs), lhs_info
        elif rhs_info is not None and isinstance(lhs, (int, float)):
            source, scalar, source_info = rhs, float(lhs), rhs_info
        if source is None or scalar < 0:
            return None
        scaled_info = QuantInfo(
            source_info.scale * scalar,
            source_info.zero_point,
            source_info.dtype,
            source_info.qmin,
            source_info.qmax,
        )
        result = self._requantize_value(
            node,
            self._quantized_value_name(source),
            scaled_info,
            output_info,
            node.name,
        )
        self._record_integer_result(node, output_info)
        return result

    def _build_quantized_matmul(self, node):
        lhs, rhs = node.args[:2]
        lhs_info, rhs_info = self.lookup_quant_info(lhs), self.lookup_quant_info(rhs)
        if lhs_info is None or rhs_info is None or self.quant_config is None:
            return None
        lhs_shape, rhs_shape, out_shape = (
            self._shape(lhs),
            self._shape(rhs),
            self._shape(node),
        )
        if len(lhs_shape) != 3 or len(rhs_shape) != 3 or len(out_shape) != 3:
            raise NotImplementedError("Integer batched matmul supports rank-3")
        B, M, K = lhs_shape
        rhs_batch, rhs_k, N = rhs_shape
        if B != rhs_batch or K != rhs_k or out_shape != (B, M, N):
            raise ValueError("Incompatible rank-3 matmul shapes")
        output_info = self._activation_quant_info(node)
        lhs_name, lhs_type, _ = self._dtype(lhs_info.dtype)
        rhs_name, rhs_type, _ = self._dtype(rhs_info.dtype)
        out_name, out_type, _ = self._dtype(output_info.dtype)
        multiplier, shift = approximate_multiplier_shift(
            lhs_info.scale * rhs_info.scale / output_info.scale
        )
        multiplier = self._emit_scalar_constant(
            multiplier, "int64", "qmatmul_multiplier"
        )
        name_id = self.get_unique_id("qmatmul3d")
        self.composition.append(
            (
                "qmatmul3d",
                name_id,
                [
                    lhs_type,
                    rhs_type,
                    types.int32,
                    types.int64,
                    out_type,
                    B,
                    M,
                    K,
                    N,
                ],
            )
        )
        self._record_integer_result(node, output_info)
        return (
            f"{node.name} = nn.qmatmul3d[{lhs_name}, {rhs_name}, int32, int64, "
            f'{out_name}, {B}, {M}, {K}, {N}, "{name_id}"]('
            f"{self._quantized_value_name(lhs)}, {self._quantized_value_name(rhs)}, "
            f"{multiplier}, {shift}, {lhs_info.zero_point}, {rhs_info.zero_point}, "
            f"{output_info.zero_point}, {output_info.qmin}, {output_info.qmax})"
        )

    def _build_quantized_silu(self, node):
        source, source_info = node.args[0], self.lookup_quant_info(node.args[0])
        if source_info is None or self.quant_config is None:
            return None
        shape = self._shape(node)
        if len(shape) != 3:
            raise NotImplementedError("Integer SiLU supports rank-3")
        output_info = self._activation_quant_info(node)
        in_name, in_type, _ = self._dtype(source_info.dtype)
        out_name, out_type, out_numpy = self._dtype(output_info.dtype)
        codes = np.arange(source_info.qmin, source_info.qmax + 1, dtype=np.int32)
        real = (codes - source_info.zero_point) * source_info.scale
        silu = real / (1.0 + np.exp(-real))
        table = np.clip(
            np.rint(silu / output_info.scale) + output_info.zero_point,
            output_info.qmin,
            output_info.qmax,
        ).astype(out_numpy)
        name_id = self.get_unique_id("qsilu3d")
        table_name = self._record_synthetic_array(
            f"qsilu_table_{name_id}", out_name, table
        )
        self.composition.append(
            ("qsilu3d", name_id, [in_type, out_type, *shape])
        )
        self._record_integer_result(node, output_info)
        B, L, D = shape
        return (
            f"{node.name} = nn.qsilu3d[{in_name}, {out_name}, {B}, {L}, {D}, "
            f'"{name_id}"]({self._quantized_value_name(source)}, {table_name}, '
            f"{source_info.qmin})"
        )

    def _build_quantized_rmsnorm(self, node):
        source, source_info = node.args[0], self.lookup_quant_info(node.args[0])
        if source_info is None or self.quant_config is None:
            return None
        shape = self._shape(node)
        if len(shape) != 3:
            raise NotImplementedError("Integer RMSNorm supports rank-3")
        module = self.get_module(node.target)
        weight_name, _, weight_info = self._quantize_module_weight(node.target)
        output_info = self._activation_quant_info(node)
        in_name, in_type, _ = self._dtype(source_info.dtype)
        weight_dtype_name, weight_type, _ = self._dtype(weight_info.dtype)
        out_name, out_type, _ = self._dtype(output_info.dtype)
        B, L, D = shape
        factor = weight_info.scale * math.sqrt(D) / output_info.scale
        factor_multiplier = self._emit_scalar_constant(
            int(round(factor * (1 << 20))), "int64", "qrms_factor"
        )
        eps_codes = self._emit_scalar_constant(
            max(0, int(round(float(module.eps) * D / (source_info.scale**2)))),
            "int64",
            "qrms_eps",
        )
        name_id = self.get_unique_id("qrms_norm3d")
        self.composition.append(
            (
                "qrms_norm3d",
                name_id,
                [
                    in_type,
                    weight_type,
                    types.int64,
                    out_type,
                    B,
                    L,
                    D,
                ],
            )
        )
        self._record_integer_result(node, output_info)
        return (
            f"{node.name} = nn.qrms_norm3d[{in_name}, {weight_dtype_name}, int64, "
            f'{out_name}, {B}, {L}, {D}, "{name_id}"]('
            f"{self._quantized_value_name(source)}, {weight_name}, "
            f"{factor_multiplier}, {eps_codes}, {source_info.zero_point}, "
            f"{weight_info.zero_point}, {output_info.zero_point}, "
            f"{output_info.qmin}, {output_info.qmax})"
        )

    def _build_quantized_rope(self, node):
        source, source_info = node.args[0], self.lookup_quant_info(node.args[0])
        if source_info is None or self.quant_config is None:
            return None
        shape = self._shape(node)
        if len(shape) != 3:
            raise NotImplementedError("Integer RoPE supports rank-3")
        module = self.get_module(node.target)
        H, L, D = shape
        cos_name = node.target.replace(".", "_") + "_cos"
        sin_name = node.target.replace(".", "_") + "_sin"
        cos_codes = np.clip(
            np.rint(module.cos.detach().numpy() * 32767.0), -32768, 32767
        ).astype(np.int16)
        sin_codes = np.clip(
            np.rint(module.sin.detach().numpy() * 32767.0), -32768, 32767
        ).astype(np.int16)
        self.param_dtypes[cos_name] = "int16"
        self.param_dtypes[sin_name] = "int16"
        self._record_runtime_array(cos_name, cos_codes)
        self._record_runtime_array(sin_name, sin_codes)
        output_info = self._activation_quant_info(node)
        in_name, in_type, _ = self._dtype(source_info.dtype)
        out_name, out_type, _ = self._dtype(output_info.dtype)
        multiplier, shift = approximate_multiplier_shift(
            source_info.scale / (32767.0 * output_info.scale)
        )
        multiplier = self._emit_scalar_constant(
            multiplier, "int64", "qrope_multiplier"
        )
        name_id = self.get_unique_id("qrope3d")
        self.composition.append(
            (
                "qrope3d",
                name_id,
                [in_type, types.int16, types.int64, out_type, H, L, D],
            )
        )
        self._record_integer_result(node, output_info)
        return (
            f"{node.name} = nn.qrope3d[{in_name}, int16, int64, {out_name}, "
            f'{H}, {L}, {D}, "{name_id}"]('
            f"{self._quantized_value_name(source)}, {cos_name}, {sin_name}, "
            f"{multiplier}, {shift}, {source_info.zero_point}, "
            f"{output_info.zero_point}, {output_info.qmin}, {output_info.qmax})"
        )

    def _build_quantized_positioned_rope(self, node):
        source, source_info = node.args[0], self.lookup_quant_info(node.args[0])
        if source_info is None or self.quant_config is None:
            return None
        H, L, D = self._shape(node)
        module = self.get_module(node.target)
        S = int(module.cos.shape[0])
        target_name = node.target.replace(".", "_")
        cos_name, sin_name = target_name + "_cos", target_name + "_sin"
        cos_codes = np.clip(
            np.rint(module.cos.detach().numpy() * 32767.0), -32768, 32767
        ).astype(np.int16)
        sin_codes = np.clip(
            np.rint(module.sin.detach().numpy() * 32767.0), -32768, 32767
        ).astype(np.int16)
        self.param_dtypes[cos_name] = "int16"
        self.param_dtypes[sin_name] = "int16"
        self._record_runtime_array(cos_name, cos_codes)
        self._record_runtime_array(sin_name, sin_codes)
        output_info = self._activation_quant_info(node)
        in_name, in_type, _ = self._dtype(source_info.dtype)
        out_name, out_type, _ = self._dtype(output_info.dtype)
        multiplier, shift = approximate_multiplier_shift(
            source_info.scale / (32767.0 * output_info.scale)
        )
        multiplier = self._emit_scalar_constant(
            multiplier, "int64", "qpositioned_rope_multiplier"
        )
        name_id = self.get_unique_id("qpositioned_rope3d")
        self.composition.append(
            (
                "qpositioned_rope3d",
                name_id,
                [in_type, types.int16, types.int64, out_type, H, L, S, D],
            )
        )
        self._record_integer_result(node, output_info)
        position = get_var_name(node.args[1])
        return (
            f"{node.name} = nn.qpositioned_rope3d[{in_name}, int16, int64, "
            f'{out_name}, {H}, {L}, {S}, {D}, "{name_id}"]('
            f"{self._quantized_value_name(source)}, {cos_name}, {sin_name}, "
            f"{multiplier}, {shift}, {source_info.zero_point}, "
            f"{output_info.zero_point}, {output_info.qmin}, {output_info.qmax}, "
            f"{position})"
        )

    def _build_quantized_causal_softmax(self, node, offset_override=None):
        source, source_info = node.args[0], self.lookup_quant_info(node.args[0])
        if source_info is None or self.quant_config is None:
            return None
        H, L, S = self._shape(node)
        output_info = QuantInfo(1.0 / 255.0, 0, types.uint8, 0, 255)
        in_name, in_type, _ = self._dtype(source_info.dtype)
        out_name, out_type, _ = self._dtype(output_info.dtype)
        exp_table = np.rint(
            np.exp(-np.arange(256, dtype=np.float64) * source_info.scale)
            * (1 << 20)
        ).astype(np.int32)
        name_id = self.get_unique_id("qcausal_softmax3d")
        # BEGIN NATIVE INT8 QUANTIZATION: CHANGED use typed synthetic LUT storage
        table_name = self._record_synthetic_array(
            f"qsoftmax_exp_table_{name_id}", "int32", exp_table
        )
        # END NATIVE INT8 QUANTIZATION: CHANGED use typed synthetic LUT storage
        output_multiplier = self._emit_scalar_constant(
            int(round((1.0 / output_info.scale) * (1 << 20))),
            "int64",
            "qsoftmax_multiplier",
        )
        self.composition.append(
            ("qcausal_softmax3d", name_id, [in_type, out_type, H, L, S])
        )
        self._record_integer_result(node, output_info)
        offset = (
            offset_override
            if offset_override is not None
            else (get_var_name(node.args[1]) if len(node.args) > 1 else 0)
        )
        return (
            f"{node.name} = nn.qcausal_softmax3d[{in_name}, {out_name}, "
            f'{H}, {L}, {S}, "{name_id}"]('
            f"{self._quantized_value_name(source)}, {table_name}, "
            f"{output_multiplier}, 20, {offset}, {source_info.zero_point}, "
            f"{output_info.zero_point}, {output_info.qmin}, {output_info.qmax})"
        )

    def _build_quantized_embedding(self, node):
        if self.quant_config is None:
            return None
        input_shape, output_shape = self._shape(node.args[0]), self._shape(node)
        if len(input_shape) != 2 or len(output_shape) != 3:
            raise NotImplementedError("Integer Embedding supports [B, L] token IDs")
        module = self.get_module(node.target)
        weight_name, _, weight_info = self._quantize_module_weight(node.target)
        weight_dtype_name, weight_type, _ = self._dtype(weight_info.dtype)
        B, L = input_shape
        V, D = tuple(module.weight.shape)
        name_id = self.get_unique_id("qembedding2d")
        self.composition.append(
            ("qembedding2d", name_id, [weight_type, B, L, V, D])
        )
        self._record_integer_result(node, weight_info)
        return (
            f"{node.name} = nn.qembedding2d[{weight_dtype_name}, {B}, {L}, "
            f'{V}, {D}, "{name_id}"]({get_var_name(node.args[0])}, {weight_name})'
        )

    def _build_quantized_repeat_interleave(self, node):
        source, info = node.args[0], self.lookup_quant_info(node.args[0])
        if info is None:
            return None
        H, L, D = self._shape(source)
        out_h, out_l, out_d = self._shape(node)
        if out_l != L or out_d != D or out_h % H != 0:
            raise ValueError("Invalid grouped-query repeat shape")
        repeat_factor = out_h // H
        dtype_name, dtype, _ = self._dtype(info.dtype)
        name_id = self.get_unique_id("repeat_interleave3d")
        self.composition.append(
            ("repeat_interleave3d", name_id, [dtype, H, L, D, repeat_factor])
        )
        self._record_integer_result(node, info)
        return (
            f"{node.name} = nn.repeat_interleave3d[{dtype_name}, {H}, {L}, "
            f'{D}, {repeat_factor}, "{name_id}"]({self._quantized_value_name(source)})'
        )

    def _build_quantized_kv_cache_update(self, node):
        values, cache = node.args[:2]
        values_info, cache_info = self.lookup_quant_info(values), self.lookup_quant_info(
            cache
        )
        if values_info is None or cache_info is None:
            return None
        H, L, D = self._shape(values)
        cache_h, S, cache_d = self._shape(cache)
        if H != cache_h or D != cache_d:
            raise ValueError("KV-cache and update shapes are incompatible")
        value_name, value_type, _ = self._dtype(values_info.dtype)
        cache_name, cache_type, _ = self._dtype(cache_info.dtype)
        multiplier, shift = approximate_multiplier_shift(
            values_info.scale / cache_info.scale
        )
        multiplier = self._emit_scalar_constant(
            multiplier, "int64", "qkv_cache_multiplier"
        )
        name_id = self.get_unique_id("qkv_cache_update3d")
        self.composition.append(
            (
                "qkv_cache_update3d",
                name_id,
                [value_type, cache_type, types.int64, H, L, S, D],
            )
        )
        self._record_integer_result(node, cache_info)
        position = get_var_name(node.args[2])
        return (
            f"{node.name} = nn.qkv_cache_update3d[{value_name}, {cache_name}, "
            f'int64, {H}, {L}, {S}, {D}, "{name_id}"]('
            f"{self._quantized_value_name(values)}, {self._quantized_value_name(cache)}, "
            f"{multiplier}, {shift}, {values_info.zero_point}, "
            f"{cache_info.zero_point}, {cache_info.qmin}, {cache_info.qmax}, "
            f"{position})"
        )

    # END NATIVE INT8 QUANTIZATION: ADDED TorchBuilder integer helpers

    def _find_types_module_symbol(self, dtype_obj):
        # Try to find a public symbol name in allo.ir.types that references this object
        for name, val in inspect.getmembers(types):
            if isinstance(val, AlloType) and val is dtype_obj:
                return name
        return None

    def _literal_for_allotype(self, t: AlloType) -> str:
        cls = t.__class__.__name__
        # Handle known constructors by their signatures
        if cls in {"Fixed", "UFixed"}:
            return f"{cls}({t.bits}, {t.fracs})"
        if cls in {"Int", "UInt"}:
            return f"{cls}({t.bits})"
        if cls == "Float":
            if t.bits == 16:
                return "float16"
            if t.bits == 32:
                return "float32"
            if t.bits == 64:
                return "float64"
            raise NotImplementedError(f"Unsupported float bits: {t.bits}")
        if cls == "Index":
            return "Index()"
        # Fallback
        return f"{cls}({t.bits}, {t.fracs})"

    def _resolve_value_to_name_obj(self, value, kind_hint):
        # Accept string or AlloType; return (name, obj)
        if isinstance(value, str):
            try:
                obj = getattr(types, value)
                return value, obj
            except Exception:
                # invalid string -> fallback to kind hint
                return self._resolve_dtype_name(kind_hint), self._resolve_dtype_obj(
                    kind_hint
                )
        if isinstance(value, AlloType):
            # Use exact constructor literal like Fixed(16, 10)
            return self._literal_for_allotype(value), value
        # fallback
        return self._resolve_dtype_name(kind_hint), self._resolve_dtype_obj(kind_hint)

    def _get_linear_dtype_triplet(self, module_key):
        # Prefer module-specific list [TyX, TyW, TyO]
        spec = self.op_dtypes.get(module_key)
        if isinstance(spec, (list, tuple)) and len(spec) == 3:
            x_name, x_obj = self._resolve_value_to_name_obj(spec[0], "linear_input")
            w_name, w_obj = self._resolve_value_to_name_obj(spec[1], "linear_weight")
            o_name, o_obj = self._resolve_value_to_name_obj(spec[2], "linear")
            return (x_name, w_name, o_name, x_obj, w_obj, o_obj)
        # Fallback to global per-op keys
        x_name = self._resolve_dtype_name("linear_input")
        w_name = self._resolve_dtype_name("linear_weight")
        o_name = self._resolve_dtype_name("linear")
        x_obj = self._resolve_dtype_obj("linear_input")
        w_obj = self._resolve_dtype_obj("linear_weight")
        o_obj = self._resolve_dtype_obj("linear")
        return (x_name, w_name, o_name, x_obj, w_obj, o_obj)

    def _resolve_dtype_name(self, op_kind):
        # prefer exact op kind, then a global default, otherwise float32
        candidate = self.op_dtypes.get(op_kind, None)
        if candidate is None:
            candidate = self.op_dtypes.get("default", None)
        # If still None, default to float32
        if candidate is None:
            return "float32"
        # If user passed a string name, validate it against types module
        if isinstance(candidate, str):
            try:
                getattr(types, candidate)
                return candidate
            except Exception:
                return "float32"
        # If user passed an AlloType, emit constructor literal (e.g., Fixed(16, 10))
        if isinstance(candidate, AlloType):
            return self._literal_for_allotype(candidate)
        # Fallback
        return "float32"

    def _resolve_dtype_obj(self, op_kind):
        candidate = self.op_dtypes.get(op_kind, None)
        if candidate is None:
            candidate = self.op_dtypes.get("default", None)
        if isinstance(candidate, AlloType):
            return candidate
        if isinstance(candidate, str):
            try:
                return getattr(types, candidate)
            except Exception:
                return float32
        return float32

    def _record_param_dtype(self, var_name_underscored, op_kind):
        # only set if not already set
        if var_name_underscored not in self.param_dtypes:
            self.param_dtypes[var_name_underscored] = self._resolve_dtype_name(op_kind)

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED input and parameter helpers

    @staticmethod
    def _input_dtype_name(value, quant_config):
        if not isinstance(value, torch.Tensor):
            return "int32"
        if value.dtype in {torch.int8, torch.qint8}:
            return "int8"
        if value.dtype in {torch.uint8, torch.quint8}:
            return "uint8"
        if value.dtype in {torch.int16}:
            return "int16"
        if value.dtype in {torch.int32}:
            return "int32"
        if value.dtype in {torch.int64}:
            return "int64"
        if value.dtype in {torch.float16}:
            return "float16"
        if value.dtype in {torch.float64}:
            return "float64"
        if quant_config is not None:
            return "float32"
        return "float32"

    @staticmethod
    def _node_has_floating_tensor(node):
        meta = node.meta.get("tensor_meta")
        return isinstance(meta, TensorMetadata) and getattr(
            meta.dtype, "is_floating_point", False
        )

    def _module_parameter(self, module_target, name="weight"):
        module = self.get_module(module_target)
        parameter = getattr(module, name, None)
        if parameter is None:
            raise KeyError(f"Module {module_target!r} has no parameter {name!r}")
        return parameter

    def _module_parameter_symbol(self, module_target, name="weight"):
        requested = f"{module_target.replace('.', '_')}_{name}"
        return self.parameter_aliases.get(requested, requested)

    def _record_runtime_array(self, name, value):
        if self.weights_as_args:
            self.runtime_param_data[name] = value
        else:
            buffer_names = {
                item.replace(".", "_") for item in self.named_buffers
            }
            prefix = "gb_" if name in buffer_names else "g_"
            self.extra_globals[prefix + name] = value

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED synthetic array ABI helper
    def _record_synthetic_array(self, name, dtype_name, value):
        shape = tuple(value.shape)
        if self.weights_as_args:
            self.runtime_aux_params[name] = (dtype_name, shape, value)
        else:
            global_name = "g_" + name
            self.extra_globals[global_name] = value
            self.synthetic_params[name] = (dtype_name, shape, global_name)
        return name

    # END NATIVE INT8 QUANTIZATION: ADDED synthetic array ABI helper

    # END NATIVE INT8 QUANTIZATION: ADDED input and parameter helpers

    def build(self):
        for node in self.gm.graph.nodes:
            self(node)
        for i, x in enumerate(self.example_inputs):
            if isinstance(x, torch.Tensor):
                self.input_shapes.append(x.shape)
                self.input_args.append(self.input_names[i])
            # BEGIN NATIVE INT8 QUANTIZATION: ADDED flattened tuple/scalar ABI
            elif isinstance(x, (list, tuple)):
                input_name = self.input_names[i]
                for num, item in enumerate(x):
                    if isinstance(item, torch.Tensor):
                        self.input_shapes.append(item.shape)
                        self.input_args.append(f"{input_name}_{num}")
                    else:
                        raise NotImplementedError("Unsupported input type")
            elif isinstance(x, int):
                self.input_shapes.append(None)
                self.input_args.append(self.input_names[i])
            # END NATIVE INT8 QUANTIZATION: ADDED flattened tuple/scalar ABI
        # BEGIN NATIVE INT8 QUANTIZATION: CHANGED preserve each logical input's
        # actual storage type instead of assigning one global floating dtype.
        input_dtype_names = []
        for value in self.example_inputs:
            if isinstance(value, torch.Tensor):
                input_dtype_names.append(
                    self._input_dtype_name(value, self.quant_config)
                )
            elif isinstance(value, (list, tuple)):
                for item in value:
                    input_dtype_names.append(
                        self._input_dtype_name(item, self.quant_config)
                    )
            else:
                input_dtype_names.append("int32")
        args = [
            (
                f"{name}: {dtype_name}[{', '.join([str(s) for s in shape])}]"
                if shape
                else f"{name}: int32"
            )
            for name, shape, dtype_name in zip(
                self.input_args, self.input_shapes, input_dtype_names
            )
        ]
        # END NATIVE INT8 QUANTIZATION: CHANGED heterogeneous input dtypes

        # Add weight parameters to function signature if weights_as_args is True
        weight_args = []
        if self.weights_as_args:
            if self.named_params:
                for name, param in self.named_params.items():
                    new_name = name.replace(".", "_")
                    dtype_name = self.param_dtypes.get(
                        new_name, self._resolve_dtype_name("default")
                    )
                    weight_args.append(
                        f"{new_name}: {dtype_name}[{', '.join([str(s) for s in param.shape])}]"
                    )

            if self.named_buffers:
                for name, buf in self.named_buffers.items():
                    new_name = name.replace(".", "_")
                    dtype_name = self.param_dtypes.get(
                        new_name, self._resolve_dtype_name("default")
                    )
                    if buf.shape:
                        shape_str = ", ".join([str(s) for s in buf.shape])
                        weight_args.append(f"{new_name}: {dtype_name}[{shape_str}]")
                    else:
                        weight_args.append(f"{new_name}: {dtype_name}")

            # BEGIN NATIVE INT8 QUANTIZATION: ADDED synthetic runtime arrays
            for name, (dtype_name, shape, _) in self.runtime_aux_params.items():
                dims = ", ".join(str(dim) for dim in shape)
                weight_args.append(f"{name}: {dtype_name}[{dims}]")
            # END NATIVE INT8 QUANTIZATION: ADDED synthetic runtime arrays

        # Combine input args and weight args
        all_args = args + weight_args

        res = ""
        # top-level function
        res += f"def forward({', '.join(all_args)})".format()
        # outputs
        res += f" -> ({', '.join(self.output)}):\n"
        # subfunctions
        if self.subfunctions:
            res += "\n".join(self.subfunctions) + "\n"

        # Declare weights as local variables (either from arguments or global constants)
        if not self.weights_as_args:
            if self.named_params:
                for name, param in self.named_params.items():
                    new_name = name.replace(".", "_")
                    dtype_name = self.param_dtypes.get(
                        new_name, self._resolve_dtype_name("default")
                    )
                    res += f"    {new_name}: {dtype_name}[{', '.join([str(s) for s in param.shape])}] = g_{new_name}\n"
            if self.named_buffers:
                for name, buf in self.named_buffers.items():
                    new_name = name.replace(".", "_")
                    dtype_name = self.param_dtypes.get(
                        new_name, self._resolve_dtype_name("default")
                    )
                    if buf.shape:
                        shape_str = ", ".join([str(s) for s in buf.shape])
                        res += f"    {new_name}: {dtype_name}[{shape_str}] = gb_{new_name}\n"
                    else:
                        res += f"    {new_name}: {dtype_name} = gb_{new_name}\n"
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED synthetic biases
        if not self.weights_as_args:
            for name, (dtype_name, shape, global_name) in self.synthetic_params.items():
                dims = ", ".join(str(dim) for dim in shape)
                res += f"    {name}: {dtype_name}[{dims}] = {global_name}\n"
        # END NATIVE INT8 QUANTIZATION: ADDED synthetic biases
        # function body
        for line in self.code:
            res += f"    {line}\n"
        return res

    def __call__(self, node):
        method = getattr(self, "build_" + node.op)
        ret = method(node)
        if ret:
            self.code.append(ret)
        return ret

    def get_unique_id(self, name):
        if name not in self.unique_id:
            self.unique_id[name] = 0
            return 0
        self.unique_id[name] += 1
        return self.unique_id[name]

    def get_module(self, name):
        return dict(self.gm.named_modules())[name]

    def build_placeholder(self, node):
        self.input_names.append(node.name)
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED calibrated input
        if (
            self.quant_config is not None
            and "tensor_meta" in node.meta
            and self._node_has_floating_tensor(node)
        ):
            self.record_quant_info(node, self._activation_quant_info(node))
            self._quantized_value_name(node)
        # END NATIVE INT8 QUANTIZATION: ADDED calibrated input

    def build_getattr(self, node):
        pass

    def build_get_attr(self, node):
        # For cls_token
        pass

    def build_repeat(self, node):
        inp = get_var_name(node.args[0])
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer repeat input
        inp, repeat_info = self._prepare_quantized_passthrough(node.args[0], node)
        # END NATIVE INT8 QUANTIZATION: ADDED integer repeat input
        input_shape = tuple(node.args[0].meta["tensor_meta"].shape)
        output_shape = tuple(node.meta["tensor_meta"].shape)
        if len(input_shape) == 3:
            B, L, C = input_shape
            repeat_factor = output_shape[0] // B
            name_id = self.get_unique_id("repeat_batch3d")
            dtype_name = self._resolve_dtype_name("repeat_batch3d")
            dtype_obj = self._resolve_dtype_obj("repeat_batch3d")
            # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer repeat dtype
            if repeat_info is not None:
                dtype_name, dtype_obj, _ = self._dtype(repeat_info.dtype)
            # END NATIVE INT8 QUANTIZATION: ADDED integer repeat dtype
            self.composition.append(
                ("repeat_batch3d", name_id, [dtype_obj, B, L, C, repeat_factor])
            )
            return f'{node.name} = nn.repeat_batch3d[{dtype_name}, {B}, {L}, {C}, {repeat_factor}, "{name_id}"]({inp})'
        raise NotImplementedError("Unsupported shape for repeat")

    def build_call_module(self, node):
        module = self.get_module(node.target)
        op = {
            torch.nn.Linear: "linear",
            torch.nn.Dropout: "identity",
            torch.nn.ReLU: "relu",
            torch.nn.GELU: "gelu",
            # BEGIN NATIVE INT8 QUANTIZATION: ADDED transformer module dispatch
            torch.nn.SiLU: "silu",
            torch.nn.Embedding: "embedding",
            # END NATIVE INT8 QUANTIZATION: ADDED transformer module dispatch
            torch.nn.LayerNorm: "layernorm",
            torch.nn.Conv2d: "conv2d",
            torch.nn.MaxPool2d: "maxpool2d",
            torch.nn.AvgPool2d: "avgpool2d",
            torch.nn.BatchNorm2d: "batchnorm2d",
            torch.nn.BatchNorm1d: "batchnorm1d",
        }.get(type(module), None)
        if self.leaf_modules:
            for leaf_module in self.leaf_modules:
                if isinstance(module, leaf_module):
                    return getattr(self, f"build_{module.__class__.__name__}")(node)
        if op is None:
            raise NotImplementedError("Unsupported module")
        if op == "linear":
            bias = True if module.bias is not None else None
            res = getattr(self, "build_linear")(node, bias)
        else:
            res = getattr(self, f"build_{op}")(node)
        # append shape after the operation
        if "tensor_meta" in node.meta:
            res += f'  # shape: {str(tuple(node.meta["tensor_meta"].shape))}'
        return res

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED QDQ boundary builders

    def build_quantize_per_tensor(self, node):
        source = node.args[0]
        info = self._quant_info_from_node(node)
        source_info = self.lookup_quant_info(source)
        if source_info is not None and source in self.integer_values:
            same = (
                source_info.scale == info.scale
                and source_info.zero_point == info.zero_point
                and get_qrange(source_info.dtype) == get_qrange(info.dtype)
            )
            if same:
                result, codes = f"{node.name} = {source.name}", source.name
            else:
                result = self._requantize_value(
                    source, source.name, source_info, info, node.name
                )
                codes = node.name
            self.record_quant_info(node, info)
            self.integer_values.add(node)
            self.materialized_quant_map[node.name + "_clamped"] = codes
            return result
        self.record_quant_info(node, info)
        if self.delay_qdq_lowering:
            return f"{node.name} = {source.name}"

        shape = self._shape(node)
        qkernel, dkernel = f"quantize{len(shape)}d", f"dequantize{len(shape)}d"
        qid, did = self.get_unique_id(qkernel), self.get_unique_id(dkernel)
        dtype_name, dtype, _ = self._dtype(info.dtype)
        integer = self._new_tmp_name(node.name + "_integer")
        dims = ", ".join(str(dim) for dim in shape)
        self.composition += [
            (qkernel, qid, [float32, dtype, *shape]),
            (dkernel, did, [dtype, float32, *shape]),
        ]
        self.code.append(
            f"{integer} = nn.{qkernel}[float32, {dtype_name}, {dims}, "
            f'"{qid}"]({source.name}, {repr(float(info.scale))}, '
            f"{info.zero_point}, {info.qmin}, {info.qmax})"
        )
        return (
            f"{node.name} = nn.{dkernel}[{dtype_name}, float32, {dims}, "
            f'"{did}"]({integer}, {repr(float(info.scale))}, {info.zero_point})'
        )

    def build_dequantize(self, node):
        source, info = node.args[0], self.lookup_quant_info(node.args[0])
        if info is None:
            return f"{node.name} = {source.name}"
        self.record_quant_info(node, info)
        if source in self.integer_values:
            if self._is_output_boundary(node):
                return self._dequantize_value(source, source.name, info, node.name)
            self.integer_values.add(node)
            self.materialized_quant_map[node.name + "_clamped"] = (
                self.materialized_quant_map.get(source.name + "_clamped", source.name)
            )
            return f"{node.name} = {source.name}"
        if self._is_output_boundary(node) and self.delay_qdq_lowering:
            return f"{node.name} = {self.materialize_quant_info(source)}"
        return f"{node.name} = {source.name}"

    def build_int_repr(self, node):
        source, info = node.args[0], self.lookup_quant_info(node.args[0])
        if info is None:
            raise RuntimeError("int_repr requires quantization metadata")
        value = self._quantized_value_name(source)
        self.record_quant_info(node, info)
        self.integer_values.add(node)
        self.materialized_quant_map[node.name + "_clamped"] = value
        return f"{node.name} = {value}"

    # END NATIVE INT8 QUANTIZATION: ADDED QDQ boundary builders

    def build_call_function(self, node):
        opcls = {
            operator.add: "add",
            operator.sub: "sub",
            operator.mul: "mul",
            operator.truediv: "div",
            operator.getitem: "getitem",
            torch.matmul: "matmul",
            torch.ones: "ones",
            torch.zeros: "zeros",
            # BEGIN NATIVE INT8 QUANTIZATION: ADDED QDQ function dispatch
            torch.quantize_per_tensor: "quantize_per_tensor",
            torch.relu: "relu",
            # END NATIVE INT8 QUANTIZATION: ADDED QDQ function dispatch
            math.sqrt: "sqrt",
            F.softmax: "softmax",
            F.log_softmax: "log_softmax",
            F.linear: "linear",
            F.gelu: "gelu",
            # BEGIN NATIVE INT8 QUANTIZATION: ADDED transformer function dispatch
            F.silu: "silu",
            # END NATIVE INT8 QUANTIZATION: ADDED transformer function dispatch
            F.relu: "relu",
            F.dropout: "identity",
            torch.tril: "tril",
            torch.cat: "concat",
        }.get(node.target)
        # Only nodes with shape need to be built.
        if "tensor_meta" in node.meta:
            res = getattr(self, f"build_{opcls}")(node)
            # BEGIN NATIVE INT8 QUANTIZATION: CHANGED a builder may emit its
            # statements directly when ordering is significant (tuple-cache
            # aliases must exist before their quantization loops).
            if res is None:
                return None
            # END NATIVE INT8 QUANTIZATION: CHANGED direct-emission handling
            # append shape after the operation
            res += f'  # shape: {str(tuple(node.meta["tensor_meta"].shape))}'
            return res
        return None

    def build_call_method(self, node):
        if node.target == "contiguous":
            return self.build_identity(node)
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED QDQ method dispatch
        if node.target == "dequantize":
            return self.build_dequantize(node)
        if node.target == "int_repr":
            return self.build_int_repr(node)
        # END NATIVE INT8 QUANTIZATION: ADDED QDQ method dispatch
        # Only nodes with shape need to be built.
        return (
            getattr(self, f"build_{node.target}")(node)
            if "tensor_meta" in node.meta
            else None
        )

    def append_output(self, output):
        shape = str(list(output.shape))
        # Prefer user-specified outputs dtype, then global default, then tensor meta dtype
        dtype_name = self._resolve_dtype_name("outputs")
        if dtype_name is None:
            dtype_name = str(output.dtype)[6:]
        self.output.append(dtype_name + shape)

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED recursive output materialization

    def _flatten_output_names(self, value):
        if isinstance(value, fx.Node):
            if self.quant_config is not None and value in self.integer_values:
                name = value.name + "_dequantized"
                self.code.append(
                    self._dequantize_value(
                        value, value.name, self.lookup_quant_info(value), name
                    )
                )
                return [name]
            return [value.name]
        if isinstance(value, (list, tuple)):
            names = []
            for item in value:
                names.extend(self._flatten_output_names(item))
            return names
        if isinstance(value, dict):
            names = []
            for item in value.values():
                names.extend(self._flatten_output_names(item))
            return names
        return [str(value)]

    # END NATIVE INT8 QUANTIZATION: ADDED recursive output materialization

    def build_output(self, node):
        if isinstance(node.meta["tensor_meta"], TensorMetadata):
            self.append_output(node.meta["tensor_meta"])
        elif isinstance(node.meta["tensor_meta"], (list, tuple)):
            for output in node.meta["tensor_meta"]:
                if isinstance(output, TensorMetadata):
                    self.append_output(output)
                elif isinstance(output, (list, tuple)):
                    for item in output:
                        if isinstance(item, TensorMetadata):
                            self.append_output(item)
                elif isinstance(output, dict):
                    for item in output.values():
                        if isinstance(item, TensorMetadata):
                            self.append_output(item)
                        else:
                            raise NotImplementedError("Unsupported output type")
        elif isinstance(node.meta["tensor_meta"], dict):
            for output in node.meta["tensor_meta"].values():
                if isinstance(output, TensorMetadata):
                    self.append_output(output)
        # BEGIN NATIVE INT8 QUANTIZATION: CHANGED recursively unwrap and
        # dequantize every public output, including KV-cache tuples.
        return_names = self._flatten_output_names(node.args[0])
        return f"return ({', '.join(return_names)})"
        # END NATIVE INT8 QUANTIZATION: CHANGED recursive output handling

    def build_getitem(self, node):
        inp = get_var_name(node.args[0])
        index = node.args[1]
        result = f"{node.name} = {inp}_{index}"
        # BEGIN NATIVE INT8 QUANTIZATION: CHANGED tuple tensor inputs receive
        # their own calibration metadata before use by cached decoder kernels.
        if self.quant_config is not None and self._node_has_floating_tensor(node):
            self.code.append(result)
            info = self._activation_quant_info(node)
            self.record_quant_info(node, info)
            self._quantized_value_name(node)
            return None
        # END NATIVE INT8 QUANTIZATION: CHANGED tuple input materialization
        return result

    def build_add(self, node):
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer residual dispatch
        result = self._build_quantized_add(node)
        if result is not None:
            return result
        if self.delay_qdq_lowering:
            lhs_info = self.lookup_quant_info(node.args[0])
            rhs_info = self.lookup_quant_info(node.args[1])
            if lhs_info is not None or rhs_info is not None:
                lhs = (
                    self.materialize_quant_info(node.args[0])
                    if lhs_info is not None
                    else get_var_name(node.args[0])
                )
                rhs = (
                    self.materialize_quant_info(node.args[1])
                    if rhs_info is not None
                    else get_var_name(node.args[1])
                )
                return f"{node.name} = {lhs} + {rhs}"
        # END NATIVE INT8 QUANTIZATION: ADDED integer residual dispatch
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} + {rhs}"

    def build_sub(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} - {rhs}"

    def build_mul(self, node):
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer multiply dispatch
        result = self._build_quantized_mul(node)
        if result is not None:
            return result
        # END NATIVE INT8 QUANTIZATION: ADDED integer multiply dispatch
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} * {rhs}"

    def build_matmul(self, node):
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer matmul dispatch
        result = self._build_quantized_matmul(node)
        if result is not None:
            return result
        # END NATIVE INT8 QUANTIZATION: ADDED integer matmul dispatch
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        # BEGIN NATIVE INT8 QUANTIZATION: CHANGED preserve the FP32 path for
        # rank-3 torch.matmul by selecting Allo's batch-matmul intrinsic.
        if (
            len(self._shape(node.args[0])) == 3
            and len(self._shape(node.args[1])) == 3
        ):
            return f"{node.name} = dsl.bmm({lhs}, {rhs})"
        # END NATIVE INT8 QUANTIZATION: CHANGED rank-3 matmul selection
        return f"{node.name} = dsl.matmul({lhs}, {rhs})"

    def build_div(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} / {rhs}"

    def build_softmax(self, node):
        if node.kwargs.get("dim") != -1:
            raise NotImplementedError("Only support softmax on the last dimension")
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer noncausal softmax dispatch
        if self.lookup_quant_info(node.args[0]) is not None:
            result = self._build_quantized_causal_softmax(
                node, offset_override=self._shape(node)[-1]
            )
            if result is not None:
                return result
        # END NATIVE INT8 QUANTIZATION: ADDED integer noncausal softmax dispatch
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.softmax({inp})"

    def build_log_softmax(self, node):
        if node.kwargs.get("dim") != -1:
            raise NotImplementedError("Only support log_softmax on the last dimension")
        inp = get_var_name(node.args[0])

        shape = tuple(node.meta["tensor_meta"].shape)
        name_id = self.get_unique_id("log_softmax")
        dtype_name = self._resolve_dtype_name("log_softmax")
        dtype_obj = self._resolve_dtype_obj("log_softmax")

        if len(shape) == 2:
            n, d = shape
            self.composition.append(("log_softmax", name_id, [dtype_obj, n, d]))
            return f'{node.name} = nn.log_softmax[{dtype_name}, {n}, {d}, "{name_id}"]({inp})'
        raise NotImplementedError(f"Unsupported shape for log_softmax: {shape}")

    def build_relu(self, node):
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer ReLU dispatch
        result = self._build_quantized_relu(node)
        if result is not None:
            return result
        # END NATIVE INT8 QUANTIZATION: ADDED integer ReLU dispatch
        inp = get_var_name(node.args[0])
        shape = tuple(node.meta["tensor_meta"].shape)
        name_id = self.get_unique_id("relu")
        dtype_name = self._resolve_dtype_name("relu")
        dtype_obj = self._resolve_dtype_obj("relu")
        if len(shape) == 2:
            n, d = shape
            self.composition.append(("relu2d", name_id, [dtype_obj, n, d]))
            return (
                f'{node.name} = nn.relu2d[{dtype_name}, {n}, {d}, "{name_id}"]({inp})'
            )
        if len(shape) == 3:
            n, l, c = shape
            name_id = self.get_unique_id("relu3d")
            self.composition.append(("relu3d", name_id, [dtype_obj, n, l, c]))
            return f'{node.name} = nn.relu3d[{dtype_name}, {n}, {l}, {c}, "{name_id}"]({inp})'
        if len(shape) == 4:
            n, c, h, w = shape
            self.composition.append(("relu4d", name_id, [dtype_obj, n, c, h, w]))
            return f'{node.name} = nn.relu4d[{dtype_name}, {n}, {c}, {h}, {w}, "{name_id}"]({inp})'
        raise NotImplementedError("Unsupported shape for relu")

    def build_linear(self, node, bias):
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer Linear dispatch
        if isinstance(node.target, str):
            dtypes = self._native_linear_dtypes(node.target)
            if dtypes is not None:
                result = self._build_native_linear(node, bool(bias), dtypes)
                if result is not None:
                    return result
        # END NATIVE INT8 QUANTIZATION: ADDED integer Linear dispatch
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        # BEGIN NATIVE INT8 QUANTIZATION: CHANGED tied Linear weights reuse the
        # canonical runtime parameter instead of duplicating the table.
        weight = self._module_parameter_symbol(node.target)
        # END NATIVE INT8 QUANTIZATION: CHANGED tied Linear weight symbol
        if bias:
            bias = get_var_name(target_name + "_bias")
            shape = tuple(node.meta["tensor_meta"].shape)
            name_id = self.get_unique_id("linear")
            # resolve per-module dtype triplet or fall back to global op keys
            (
                dtype_X_name,
                dtype_W_name,
                dtype_O_name,
                dtype_X_obj,
                dtype_W_obj,
                dtype_O_obj,
            ) = self._get_linear_dtype_triplet(node.target)
            # record parameter dtypes
            # BEGIN NATIVE INT8 QUANTIZATION: CHANGED record the canonical
            # symbol and query the owning module for tied weights.
            self.param_dtypes[weight] = dtype_W_name
            module_weight_shape = self._module_parameter(node.target).shape
            # END NATIVE INT8 QUANTIZATION: CHANGED tied Linear metadata
            self.param_dtypes[f"{target_name}_bias"] = dtype_O_name
            if len(shape) == 2:
                n, d = shape
                _, m = module_weight_shape
                # instantiate TyX, TyW, TyO, M, N, K
                self.composition.append(
                    (
                        "linear2d",
                        name_id,
                        [dtype_X_obj, dtype_W_obj, dtype_O_obj, n, d, m],
                    )
                )
                return f'{node.name} = nn.linear2d[{dtype_X_name}, {dtype_W_name}, {dtype_O_name}, {n}, {d}, {m}, "{name_id}"]({inp}, {weight}, {bias})'
            if len(shape) == 3:
                bs, l, m = shape
                _, d = module_weight_shape
                # instantiate TyX, TyW, TyO, B, L, D, M
                self.composition.append(
                    (
                        "linear3d",
                        name_id,
                        [dtype_X_obj, dtype_W_obj, dtype_O_obj, bs, l, d, m],
                    )
                )
                return f'{node.name} = nn.linear3d[{dtype_X_name}, {dtype_W_name}, {dtype_O_name}, {bs}, {l}, {d}, {m}, "{name_id}"]({inp}, {weight}, {bias})'
            raise NotImplementedError("Unsupported shape for linear")
        return f"{node.name} = dsl.linear({inp}, {weight})"

    def build_gelu(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.gelu({inp})"

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED generic transformer builders

    def build_silu(self, node):
        result = self._build_quantized_silu(node)
        if result is not None:
            return result
        inp = get_var_name(node.args[0])
        shape = self._shape(node)
        if len(shape) != 3:
            raise NotImplementedError("SiLU supports rank-3 tensors")
        B, L, D = shape
        name_id = self.get_unique_id("silu3d")
        dtype_name = self._resolve_dtype_name("silu")
        dtype_obj = self._resolve_dtype_obj("silu")
        self.composition.append(("silu3d", name_id, [dtype_obj, B, L, D]))
        return f'{node.name} = nn.silu3d[{dtype_name}, {B}, {L}, {D}, "{name_id}"]({inp})'

    def build_embedding(self, node):
        result = self._build_quantized_embedding(node)
        if result is not None:
            return result
        module = self.get_module(node.target)
        B, L = self._shape(node.args[0])
        V, D = tuple(module.weight.shape)
        weight = self._module_parameter_symbol(node.target)
        dtype_name = self._resolve_dtype_name("embedding")
        dtype_obj = self._resolve_dtype_obj("embedding")
        self.param_dtypes[weight] = dtype_name
        name_id = self.get_unique_id("embedding2d")
        self.composition.append(
            ("embedding2d", name_id, [dtype_obj, B, L, V, D])
        )
        return (
            f'{node.name} = nn.embedding2d[{dtype_name}, {B}, {L}, {V}, {D}, '
            f'"{name_id}"]({get_var_name(node.args[0])}, {weight})'
        )

    def build_RMSNorm(self, node):
        result = self._build_quantized_rmsnorm(node)
        if result is not None:
            return result
        B, L, D = self._shape(node)
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        weight = target_name + "_weight"
        dtype_name = self._resolve_dtype_name("rms_norm")
        dtype_obj = self._resolve_dtype_obj("rms_norm")
        self.param_dtypes[weight] = dtype_name
        name_id = self.get_unique_id("rms_norm3d")
        self.composition.append(("rms_norm3d", name_id, [dtype_obj, B, L, D]))
        return (
            f'{node.name} = nn.rms_norm3d[{dtype_name}, {B}, {L}, {D}, '
            f'"{name_id}"]({get_var_name(node.args[0])}, {weight}, {module.eps})'
        )

    def build_RotaryEmbedding(self, node):
        result = self._build_quantized_rope(node)
        if result is not None:
            return result
        H, L, D = self._shape(node)
        target_name = node.target.replace(".", "_")
        cos_name, sin_name = target_name + "_cos", target_name + "_sin"
        dtype_name = self._resolve_dtype_name("rope")
        dtype_obj = self._resolve_dtype_obj("rope")
        self.param_dtypes[cos_name] = dtype_name
        self.param_dtypes[sin_name] = dtype_name
        name_id = self.get_unique_id("rope3d")
        self.composition.append(("rope3d", name_id, [dtype_obj, H, L, D]))
        return (
            f'{node.name} = nn.rope3d[{dtype_name}, {H}, {L}, {D}, "{name_id}"]('
            f"{get_var_name(node.args[0])}, {cos_name}, {sin_name})"
        )

    def build_PositionedRotaryEmbedding(self, node):
        result = self._build_quantized_positioned_rope(node)
        if result is not None:
            return result
        H, L, D = self._shape(node)
        module = self.get_module(node.target)
        S = int(module.cos.shape[0])
        target_name = node.target.replace(".", "_")
        cos_name, sin_name = target_name + "_cos", target_name + "_sin"
        dtype_name = self._resolve_dtype_name("rope")
        dtype_obj = self._resolve_dtype_obj("rope")
        self.param_dtypes[cos_name] = dtype_name
        self.param_dtypes[sin_name] = dtype_name
        name_id = self.get_unique_id("positioned_rope3d")
        self.composition.append(
            ("positioned_rope3d", name_id, [dtype_obj, H, L, S, D])
        )
        return (
            f'{node.name} = nn.positioned_rope3d[{dtype_name}, {H}, {L}, {S}, '
            f'{D}, "{name_id}"]({get_var_name(node.args[0])}, {cos_name}, '
            f"{sin_name}, {get_var_name(node.args[1])})"
        )

    def build_RepeatKV(self, node):
        result = self._build_quantized_repeat_interleave(node)
        if result is not None:
            return result
        H, L, D = self._shape(node.args[0])
        out_h, _, _ = self._shape(node)
        repeat_factor = out_h // H
        dtype_name = self._resolve_dtype_name("repeat_interleave3d")
        dtype_obj = self._resolve_dtype_obj("repeat_interleave3d")
        name_id = self.get_unique_id("repeat_interleave3d")
        self.composition.append(
            (
                "repeat_interleave3d",
                name_id,
                [dtype_obj, H, L, D, repeat_factor],
            )
        )
        return (
            f'{node.name} = nn.repeat_interleave3d[{dtype_name}, {H}, {L}, {D}, '
            f'{repeat_factor}, "{name_id}"]({get_var_name(node.args[0])})'
        )

    def build_CausalSoftmax(self, node):
        result = self._build_quantized_causal_softmax(node)
        if result is not None:
            return result
        H, L, S = self._shape(node)
        dtype_name = self._resolve_dtype_name("causal_softmax")
        dtype_obj = self._resolve_dtype_obj("causal_softmax")
        name_id = self.get_unique_id("causal_softmax3d")
        self.composition.append(
            ("causal_softmax3d", name_id, [dtype_obj, H, L, S])
        )
        offset = get_var_name(node.args[1]) if len(node.args) > 1 else 0
        return (
            f'{node.name} = nn.causal_softmax3d[{dtype_name}, {H}, {L}, {S}, '
            f'"{name_id}"]({get_var_name(node.args[0])}, {offset})'
        )

    def build_KVCacheUpdate(self, node):
        result = self._build_quantized_kv_cache_update(node)
        if result is not None:
            return result
        H, L, D = self._shape(node.args[0])
        cache_h, S, cache_d = self._shape(node.args[1])
        if H != cache_h or D != cache_d:
            raise ValueError("KV-cache and update shapes are incompatible")
        dtype_name = self._resolve_dtype_name("kv_cache")
        dtype_obj = self._resolve_dtype_obj("kv_cache")
        name_id = self.get_unique_id("kv_cache_update3d")
        self.composition.append(
            ("kv_cache_update3d", name_id, [dtype_obj, H, L, S, D])
        )
        return (
            f'{node.name} = nn.kv_cache_update3d[{dtype_name}, {H}, {L}, {S}, '
            f'{D}, "{name_id}"]({get_var_name(node.args[0])}, '
            f"{get_var_name(node.args[1])}, {get_var_name(node.args[2])})"
        )

    # END NATIVE INT8 QUANTIZATION: ADDED generic transformer builders

    def build_layernorm(self, node):
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        weight = get_var_name(target_name + "_weight")
        bias = get_var_name(target_name + "_bias")
        return f"{node.name} = dsl.layernorm({inp}, {weight}, {bias})"

    def build_view(self, node):
        inp = get_var_name(node.args[0])
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer view propagation
        inp, _ = self._prepare_quantized_passthrough(node.args[0], node)
        # END NATIVE INT8 QUANTIZATION: ADDED integer view propagation
        shape = tuple(node.meta["tensor_meta"].shape)
        return f"{node.name} = dsl.view({inp}, {shape})"

    def build_reshape(self, node):
        return self.build_view(node)

    def build_permute(self, node):
        inp = get_var_name(node.args[0])
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer permute propagation
        inp, _ = self._prepare_quantized_passthrough(node.args[0], node)
        # END NATIVE INT8 QUANTIZATION: ADDED integer permute propagation
        permutation = node.args[1:]
        return f"{node.name} = dsl.transpose({inp}, {permutation})"

    def build_transpose(self, node):
        # PyTorch only supports transposing two dimensions,
        # https://pytorch.org/docs/stable/generated/torch.transpose.html
        inp = get_var_name(node.args[0])
        shape_len = len(node.meta["tensor_meta"].shape)
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer transpose propagation
        inp, _ = self._prepare_quantized_passthrough(node.args[0], node)
        # END NATIVE INT8 QUANTIZATION: ADDED integer transpose propagation
        sorted_args = sorted(
            [
                node.args[1] if node.args[1] >= 0 else node.args[1] + shape_len,
                node.args[2] if node.args[2] >= 0 else node.args[2] + shape_len,
            ]
        )
        permutation = list(range(shape_len))
        permutation[sorted_args[0]] = sorted_args[1]
        permutation[sorted_args[1]] = sorted_args[0]
        return f"{node.name} = dsl.transpose({inp}, {tuple(permutation)})"

    def build_identity(self, node):
        inp = get_var_name(node.args[0])
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED identity metadata propagation
        info = self.lookup_quant_info(node.args[0])
        if info is not None:
            self.record_quant_info(node, info)
            if node.args[0] in self.integer_values:
                self.integer_values.add(node)
                self.materialized_quant_map[node.name + "_clamped"] = (
                    self.materialized_quant_map.get(
                        node.args[0].name + "_clamped", node.args[0].name
                    )
                )
        # END NATIVE INT8 QUANTIZATION: ADDED identity metadata propagation
        return f"{node.name} = {inp}"

    def build_ones(self, node):
        shape = tuple(node.meta["tensor_meta"].shape)
        dtype = node.meta["tensor_meta"].dtype
        if str(dtype).startswith("torch."):
            dtype = str(dtype)[6:]
        return f"{node.name} = dsl.ones({shape}, dtype={dtype})"

    def build_zeros(self, node):
        shape = tuple(node.meta["tensor_meta"].shape)
        dtype = node.meta["tensor_meta"].dtype
        if str(dtype).startswith("torch."):
            dtype = str(dtype)[6:]
        return f"{node.name} = dsl.zeros({shape}, dtype={dtype})"

    def build_tril(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.tril({inp})"

    def build_concat(self, node):
        shape_len = len(node.meta["tensor_meta"].shape)
        tensor_A = get_var_name(node.args[0][0])
        tensor_B = get_var_name(node.args[0][1])
        shape_a = tuple(node.args[0][0].meta["tensor_meta"].shape)
        shape_b = tuple(node.args[0][1].meta["tensor_meta"].shape)
        dim = node.kwargs["dim"] + (node.kwargs["dim"] < 0) * shape_len
        if dim != 1:
            return f"{node.name} = dsl.concat({tensor_A}, {tensor_B}, axis={dim})"

        # Not use dsl.concat to avoid memref.copy, which is not supported in HLS
        # Only for dim=1 (concat cls_token)
        # (B, N1, C) and (B, N2, C)
        B, N1, C = shape_a
        N2 = shape_b[1]
        name_id = self.get_unique_id("concat")
        dtype_name = self._resolve_dtype_name("repeat_batch3d")
        dtype_obj = self._resolve_dtype_obj("repeat_batch3d")
        self.composition.append(("concat", name_id, [dtype_obj, B, N1, N2, C]))
        return f'{node.name} = nn.concat[{dtype_name}, {B}, {N1}, {N2}, {C}, "{name_id}"]({tensor_A}, {tensor_B})'

    def build_CoreAttention(self, node):
        shape = tuple(self.example_inputs[1][0].shape)
        src = inspect.getsource(CoreAttention_lib(*shape))
        src = (
            src.replace("s_0", str(shape[0]))
            .replace("s_1", str(shape[1]))
            .replace("s_2", str(shape[2]))
            .replace("s_3", str(shape[3]))
        )

        if src not in self.subfunctions:
            self.subfunctions.append(src)
        return f"{node.name} = CoreAttention({', '.join([get_var_name(arg) for arg in node.args])})"

    def build_KVCache(self, node):
        shape = tuple(node.meta["tensor_meta"][0])
        src = inspect.getsource(KVCache_lib(*shape))
        src = (
            src.replace("s_0", str(shape[0]))
            .replace("s_1", str(shape[1]))
            .replace("s_2", str(shape[2]))
            .replace("s_3", str(shape[3]))
        )

        if src not in self.subfunctions:
            self.subfunctions.append(src)
        return f"{node.name} = KVCache({', '.join([get_var_name(arg) for arg in node.args])})"

    def build_SliceClsToken(self, node):
        shape = tuple(node.args[0].meta["tensor_meta"].shape)
        src = inspect.getsource(SliceClsToken_lib(*shape))
        src = (
            src.replace("s_0", str(shape[0]))
            .replace("s_1", str(shape[1]))
            .replace("s_2", str(shape[2]))
        )
        if src not in self.subfunctions:
            self.subfunctions.append(src)

        return f"{node.name} = SliceClsToken({get_var_name(node.args[0])})"

    def build_conv2d(self, node):
        # The current implementation only supports conv2d with bias, dialation=1, shape = 4
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        weight = get_var_name(target_name + "_weight")
        input_shape = tuple(node.args[0].meta["tensor_meta"].shape)

        has_bias = hasattr(module, "bias") and module.bias is not None
        bias = get_var_name(target_name + "_bias") if has_bias else None
        padding = module.padding
        stride = module.stride
        dilation = module.dilation

        out_shape = tuple(node.meta["tensor_meta"].shape)
        weight_shape = tuple(self.named_params[f"{str(node.target)}.weight"].shape)

        if len(input_shape) == 4:
            B, Cin, H, W = input_shape  # (B, Cin, H, W)
            B, Cout, Oh, Ow = out_shape  # (B, Cout, Oh, Ow)
            _, Cin, Kh, Kw = weight_shape  # (Cout, Cin/groups, Kh, Kw)

            name_id = self.get_unique_id("conv2d")
            dtype_name = self._resolve_dtype_name("conv2d")
            dtype_obj = self._resolve_dtype_obj("conv2d")
            # record parameter dtypes
            self._record_param_dtype(f"{target_name}_weight", "conv2d")
            if has_bias:
                self._record_param_dtype(f"{target_name}_bias", "conv2d")

            self.composition.append(
                (
                    "conv2d",
                    name_id,
                    [
                        dtype_obj,
                        B,
                        Cin,
                        Cout,
                        H,
                        W,
                        Kh,
                        Kw,
                        Oh,
                        Ow,
                        stride[0],
                        stride[1],
                        padding[0],
                        padding[1],
                    ],
                )
            )
            if dilation != (1, 1):
                raise NotImplementedError(
                    f"Unsupported conv2d with dilation: {dilation}"
                )

            if has_bias:
                return f'{node.name} = nn.conv2d[{dtype_name}, {B}, {Cin}, {Cout}, {H}, {W}, {Kh}, {Kw}, {Oh}, {Ow}, {stride[0]}, {stride[1]}, {padding[0]}, {padding[1]}, "{name_id}"]({inp}, {weight}, {bias})'
            raise NotImplementedError("Unsupported conv2d without bias")
        raise NotImplementedError(f"Unsupported shape for conv: {input_shape}")

    def build_maxpool2d(self, node):
        module = self.get_module(node.target)
        inp = get_var_name(node.args[0])
        input_shape = tuple(node.args[0].meta["tensor_meta"].shape)

        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding

        out_shape = tuple(node.meta["tensor_meta"].shape)

        if len(input_shape) == 4:
            B, C, H, W = input_shape
            B, C, Oh, Ow = out_shape
            K = kernel_size
            name_id = self.get_unique_id("maxpool2d")
            dtype_name = self._resolve_dtype_name("maxpool2d")
            dtype_obj = self._resolve_dtype_obj("maxpool2d")

            self.composition.append(
                (
                    "maxpool2d",
                    name_id,
                    [dtype_obj, B, C, H, W, K, Oh, Ow, stride, padding],
                )
            )

            return f'{node.name} = nn.maxpool2d[{dtype_name}, {B}, {C}, {H}, {W}, {K}, {Oh}, {Ow}, {stride}, {padding}, "{name_id}"]({inp})'
        raise NotImplementedError(f"Unsupported shape for maxpool2d: {input_shape}")

    def build_avgpool2d(self, node):
        module = self.get_module(node.target)
        inp = get_var_name(node.args[0])
        input_shape = tuple(node.args[0].meta["tensor_meta"].shape)

        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding

        out_shape = tuple(node.meta["tensor_meta"].shape)

        if len(input_shape) == 4:
            B, C, H, W = input_shape
            B, C, Oh, Ow = out_shape
            K = kernel_size
            name_id = self.get_unique_id("avgpool2d")
            dtype_name = self._resolve_dtype_name("avgpool2d")
            dtype_obj = self._resolve_dtype_obj("avgpool2d")

            self.composition.append(
                (
                    "avgpool2d",
                    name_id,
                    [dtype_obj, B, C, H, W, K, Oh, Ow, stride, padding],
                )
            )

            return f'{node.name} = nn.avgpool2d[{dtype_name}, {B}, {C}, {H}, {W}, {K}, {Oh}, {Ow}, {stride}, {padding}, "{name_id}"]({inp})'
        raise NotImplementedError(f"Unsupported shape for avgpool2d: {input_shape}")

    def build_batchnorm1d(self, node):
        module = self.get_module(node.target)
        inp = get_var_name(node.args[0])
        input_shape = tuple(node.args[0].meta["tensor_meta"].shape)
        target_name = node.target.replace(".", "_")

        gamma = get_var_name(target_name + "_weight")
        beta = get_var_name(target_name + "_bias")
        eps = module.eps
        running_mean = get_var_name(target_name + "_running_mean")
        running_var = get_var_name(target_name + "_running_var")

        # Input: (N,C)
        if len(input_shape) == 2:
            B, C = input_shape
            name_id = self.get_unique_id("batchnorm1d_2d")
            dtype_name = self._resolve_dtype_name("batchnorm1d_2d")
            dtype_obj = self._resolve_dtype_obj("batchnorm1d_2d")

            self._record_param_dtype(f"{target_name}_weight", "batchnorm1d_2d")
            self._record_param_dtype(f"{target_name}_bias", "batchnorm1d_2d")
            self._record_param_dtype(f"{target_name}_running_mean", "batchnorm1d_2d")
            self._record_param_dtype(f"{target_name}_running_var", "batchnorm1d_2d")

            self.composition.append(("batchnorm1d_2d", name_id, [dtype_obj, B, C]))
            return f'{node.name} = nn.batchnorm1d_2d[{dtype_name}, {B}, {C}, "{name_id}"]({inp}, {gamma}, {beta}, {eps}, {running_mean}, {running_var})'
        # Input: (N,C,L)
        if len(input_shape) == 3:
            B, C, L = input_shape
            name_id = self.get_unique_id("batchnorm1d_3d")
            dtype_name = self._resolve_dtype_name("batchnorm1d_3d")
            dtype_obj = self._resolve_dtype_obj("batchnorm1d_3d")

            self.composition.append(("batchnorm1d_3d", name_id, [dtype_obj, B, C, L]))
            return (
                f'{node.name} = nn.batchnorm1d_3d[{dtype_name}, {B}, {C}, {L}, "{name_id}"]'
                f"({inp}, {gamma}, {beta}, {eps}, {running_mean}, {running_var})"
            )
        raise NotImplementedError(f"Unsupported shape for batchnorm1d: {input_shape}")

    def build_batchnorm2d(self, node):
        module = self.get_module(node.target)
        inp = get_var_name(node.args[0])
        input_shape = tuple(node.args[0].meta["tensor_meta"].shape)
        target_name = node.target.replace(".", "_")

        gamma = get_var_name(target_name + "_weight")
        beta = get_var_name(target_name + "_bias")
        eps = module.eps

        running_mean = get_var_name(target_name + "_running_mean")
        running_var = get_var_name(target_name + "_running_var")

        if len(input_shape) == 4:
            B, C, H, W = input_shape

            name_id = self.get_unique_id("batchnorm2d")
            dtype_name = self._resolve_dtype_name("batchnorm2d")
            dtype_obj = self._resolve_dtype_obj("batchnorm2d")
            # record parameter/buffer dtypes
            self._record_param_dtype(f"{target_name}_weight", "batchnorm2d")
            self._record_param_dtype(f"{target_name}_bias", "batchnorm2d")
            self._record_param_dtype(f"{target_name}_running_mean", "batchnorm2d")
            self._record_param_dtype(f"{target_name}_running_var", "batchnorm2d")

            self.composition.append(("batchnorm2d", name_id, [dtype_obj, B, C, H, W]))

            return f'{node.name} = nn.batchnorm2d[{dtype_name}, {B}, {C}, {H}, {W}, "{name_id}"]({inp}, {gamma}, {beta}, {eps}, {running_mean}, {running_var})'
        raise NotImplementedError(f"Unsupported shape for batchnorm2d: {input_shape}")
