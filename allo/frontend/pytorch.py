# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-public-methods, too-many-instance-attributes, broad-exception-caught

import operator
import inspect
import math

try:
    import torch
    from torch import fx
    from torch.nn import functional as F
    from torch.fx.graph_module import GraphModule
    from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata
    from .tracer import AlloTracer
except ImportError:
    pass
from .library import CoreAttention_lib, KVCache_lib
from .. import dsl
from ..library import nn
from ..ir import types
from ..customize import customize
from ..ir.types import float32, AlloType


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

    tracer = AlloTracer(model, concrete_args=concrete_args, leaf_modules=leaf_modules)
    graph = tracer.trace()
    name = (
        model.__class__.__name__
        if isinstance(model, torch.nn.Module)
        else model.__name__
    )
    gm = GraphModule(tracer.root, graph, name)
    ShapeProp(gm).propagate(*args)
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

    builder = TorchBuilder(gm, example_inputs, leaf_modules, op_dtypes, weights_as_args)
    code = builder.build()
    if verbose:
        print(code)
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
        # Store weight data for easy access
        weight_data = []
        for name, param in gm.named_parameters():
            weight_data.append(param.detach().numpy())
        for name, buf in gm.named_buffers():
            weight_data.append(buf.detach().numpy())

        # Create a wrapper that accepts inputs and weights
        def wrapped_forward(*args):
            # If only inputs are provided, automatically add weights
            if len(args) == len(example_inputs):
                # User only passed inputs, automatically add weights
                return mod(*args, *weight_data)
            # User passed both inputs and weights
            num_inputs = len(example_inputs)
            inputs = args[:num_inputs]
            weights = args[num_inputs:]
            return mod(*inputs, *weights)

        # Attach weight data to the wrapper for convenience
        wrapped_forward.weight_data = weight_data
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
    ):
        self.gm = gm
        self.code = []
        self.input_names = []
        self.input_shapes = []
        self.example_inputs = example_inputs
        self.leaf_modules = leaf_modules
        self.input_args = []
        self.named_params = dict(gm.named_parameters())
        self.named_buffers = dict(gm.named_buffers())
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

    def build(self):
        for node in self.gm.graph.nodes:
            self(node)
        for i, x in enumerate(self.example_inputs):
            if isinstance(x, torch.Tensor):
                self.input_shapes.append(x.shape)
                self.input_args.append(self.input_names[i])
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
        input_dtype_name = self._resolve_dtype_name("inputs")
        args = [
            (
                f"{name}: {input_dtype_name}[{', '.join([str(s) for s in shape])}]"
                if shape
                else f"{name}: int32"
            )
            for name, shape in zip(self.input_args, self.input_shapes)
        ]

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

    def build_getattr(self, node):
        pass

    def build_call_module(self, node):
        module = self.get_module(node.target)
        op = {
            torch.nn.Linear: "linear",
            torch.nn.Dropout: "identity",
            torch.nn.ReLU: "relu",
            torch.nn.GELU: "gelu",
            torch.nn.LayerNorm: "layernorm",
            torch.nn.Conv2d: "conv2d",
            torch.nn.MaxPool2d: "maxpool2d",
            torch.nn.AvgPool2d: "avgpool2d",
            torch.nn.BatchNorm2d: "batchnorm2d",
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
            math.sqrt: "sqrt",
            F.softmax: "softmax",
            F.linear: "linear",
            F.gelu: "gelu",
            F.relu: "relu",
            F.dropout: "identity",
            torch.tril: "tril",
            torch.cat: "concat",
        }.get(node.target)
        # Only nodes with shape need to be built.
        if "tensor_meta" in node.meta:
            res = getattr(self, f"build_{opcls}")(node)
            # append shape after the operation
            res += f'  # shape: {str(tuple(node.meta["tensor_meta"].shape))}'
            return res
        return None

    def build_call_method(self, node):
        if node.target == "contiguous":
            return self.build_identity(node)
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
        # Unwrap all outputs and return them
        name = get_var_name(node.args[0])
        if isinstance(name, dict):
            name = list(name.values())
        return_name = (
            str(name)
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
        )
        if return_name.endswith(","):
            return_name = return_name[:-1]
        return f"return ({return_name})"

    def build_getitem(self, node):
        inp = get_var_name(node.args[0])
        index = node.args[1]
        return f"{node.name} = {inp}_{index}"

    def build_add(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} + {rhs}"

    def build_sub(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} - {rhs}"

    def build_mul(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} * {rhs}"

    def build_matmul(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = dsl.matmul({lhs}, {rhs})"

    def build_div(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} / {rhs}"

    def build_softmax(self, node):
        if node.kwargs.get("dim") != -1:
            raise NotImplementedError("Only support softmax on the last dimension")
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.softmax({inp})"

    def build_relu(self, node):
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
        if len(shape) == 4:
            n, c, h, w = shape
            self.composition.append(("relu4d", name_id, [dtype_obj, n, c, h, w]))
            return f'{node.name} = nn.relu4d[{dtype_name}, {n}, {c}, {h}, {w}, "{name_id}"]({inp})'
        raise NotImplementedError("Unsupported shape for relu")

    def build_linear(self, node, bias):
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        weight = get_var_name(target_name + "_weight")
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
            self.param_dtypes[f"{target_name}_weight"] = dtype_W_name
            self.param_dtypes[f"{target_name}_bias"] = dtype_O_name
            if len(shape) == 2:
                n, d = shape
                _, m = self.named_params[f"{str(node.target)}.weight"].shape
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
                _, d = self.named_params[f"{str(node.target)}.weight"].shape
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

    def build_layernorm(self, node):
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        weight = get_var_name(target_name + "_weight")
        bias = get_var_name(target_name + "_bias")
        return f"{node.name} = dsl.layernorm({inp}, {weight}, {bias})"

    def build_view(self, node):
        inp = get_var_name(node.args[0])
        shape = tuple(node.meta["tensor_meta"].shape)
        return f"{node.name} = dsl.view({inp}, {shape})"

    def build_reshape(self, node):
        return self.build_view(node)

    def build_permute(self, node):
        inp = get_var_name(node.args[0])
        permutation = node.args[1:]
        return f"{node.name} = dsl.transpose({inp}, {permutation})"

    def build_transpose(self, node):
        # PyTorch only supports transposing two dimensions,
        # https://pytorch.org/docs/stable/generated/torch.transpose.html
        inp = get_var_name(node.args[0])
        shape_len = len(node.meta["tensor_meta"].shape)
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
        dim = node.kwargs["dim"] + (node.kwargs["dim"] < 0) * shape_len
        return f"{node.name} = dsl.concat({tensor_A}, {tensor_B}, axis={dim})"

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
