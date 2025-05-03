# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-public-methods, too-many-instance-attributes

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
from ..ir.types import float32


def from_pytorch(
    model,
    example_inputs,
    leaf_modules=None,
    verbose=False,
    enable_tensor=False,
    target="llvm",
    mode="csim",
    project="top.prj",
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
    for name, param in gm.named_parameters():
        new_name = "g_" + name.replace(".", "_")
        global_vars.update({new_name: param.detach().numpy()})

    builder = TorchBuilder(gm, example_inputs, leaf_modules)
    code = builder.build()
    if verbose:
        print(code)
    s = customize(code, global_vars=global_vars, enable_tensor=enable_tensor)
    # composition
    for func, idx, inst in builder.composition:
        s.compose(getattr(nn, func), id=idx, instantiate=inst)
    if verbose:
        print(s.module)
    if target == "mlir":
        return s
    mod = s.build(target=target, mode=mode, project=project)
    return mod


def get_var_name(node):
    return node.name if isinstance(node, fx.Node) else node


class TorchBuilder:
    def __init__(self, gm, example_inputs, leaf_modules=None):
        self.gm = gm
        self.code = []
        self.input_names = []
        self.input_shapes = []
        self.example_inputs = example_inputs
        self.leaf_modules = leaf_modules
        self.input_args = []
        self.named_params = dict(gm.named_parameters())
        self.subfunctions = []
        self.output = []
        self.composition = []
        self.unique_id = {}

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
        args = [
            (
                f"{name}: float32[{', '.join([str(s) for s in shape])}]"
                if shape
                else f"{name}: int32"
            )
            for name, shape in zip(self.input_args, self.input_shapes)
        ]
        res = ""
        # top-level function
        res += f"def forward({', '.join(args)})".format()
        # outputs
        res += f" -> ({', '.join(self.output)}):\n"
        # subfunctions
        if self.subfunctions:
            res += "\n".join(self.subfunctions) + "\n"
        if self.named_params:
            for name, param in self.named_params.items():
                new_name = name.replace(".", "_")
                res += f"    {new_name}: float32[{', '.join([str(s) for s in param.shape])}] = g_{new_name}\n"
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
            torch.nn.GELU: "gelu",
            torch.nn.LayerNorm: "layernorm",
            torch.nn.Conv2d: "conv2d",
            torch.nn.MaxPool2d: "maxpool",
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
        dtype = str(output.dtype)[6:]
        self.output.append(dtype + shape)

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
        if len(shape) == 2:
            n, d = shape
            self.composition.append(("relu2d", name_id, [float32, n, d]))
            return f'{node.name} = nn.relu2d[float32, {n}, {d}, "{name_id}"]({inp})'
        if len(shape) == 4:
            n, c, h, w = shape
            self.composition.append(("relu4d", name_id, [float32, n, c, h, w]))
            return f'{node.name} = nn.relu4d[float32, {n}, {c}, {h}, {w}, "{name_id}"]({inp})'
        raise NotImplementedError("Unsupported shape for relu")

    def build_linear(self, node, bias):
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        weight = get_var_name(target_name + "_weight")
        if bias:
            bias = get_var_name(target_name + "_bias")
            shape = tuple(node.meta["tensor_meta"].shape)
            name_id = self.get_unique_id("linear")
            if len(shape) == 2:
                n, d = shape
                _, m = self.named_params[f"{str(node.target)}.weight"].shape
                # n*m x (m*d)^T + (n*1) = n*d
                self.composition.append(("linear2d", name_id, [float32, n, d, m]))
                return f'{node.name} = nn.linear2d[float32, {n}, {d}, {m}, "{name_id}"]({inp}, {weight}, {bias})'
            if len(shape) == 3:
                bs, l, m = shape
                _, d = self.named_params[f"{str(node.target)}.weight"].shape
                self.composition.append(
                    (
                        "linear3d",
                        name_id,
                        [float32, bs, l, d, m],
                    )
                )
                return f'{node.name} = nn.linear3d[float32, {bs}, {l}, {d}, {m}, "{name_id}"]({inp}, {weight}, {bias})'
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
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        filter = get_var_name(target_name + "_weight")
        return f"{node.name} = dsl.conv2d({inp}, {filter})"

    def build_maxpool(self, node):
        inp = get_var_name(node.args[0])
        kernel_size = self.gm.get_submodule(node.target).kernel_size
        return f"{node.name} = dsl.maxpool({inp}, {kernel_size})"
