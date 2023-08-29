# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
import inspect

try:
    import torch
    from torch import fx
    from torch.nn import functional as F
except ImportError:
    pass

from .. import dsl
from ..ir import types
from ..customize import customize


def from_pytorch(model, example_inputs, verbose=False):
    gm = fx.symbolic_trace(model)
    if verbose:
        print(gm.graph)
    global_vars = {}
    for pymod in (types,):
        global_vars.update({item[0]: item[1] for item in inspect.getmembers(pymod)})
    global_vars.update({"dsl": dsl})
    for name, param in gm.named_parameters():
        new_name = "g_" + name.replace(".", "_")
        global_vars.update({new_name: param.detach().numpy()})

    builder = TorchBuilder(gm, example_inputs)
    code = builder.build()
    s = customize(code, verbose=verbose, global_vars=global_vars)
    if verbose:
        print(s.module)
    return s.build()


def get_var_name(node):
    return node.name if isinstance(node, fx.Node) else node


class TorchBuilder:
    def __init__(self, gm, example_inputs):
        self.gm = gm
        self.code = []
        self.input_args = []
        self.input_shapes = [x.shape for x in example_inputs]
        self.named_params = gm.named_parameters()

    def build(self):
        for node in self.gm.graph.nodes:
            self(node)
        args = [
            f"{name}: float32[{', '.join([str(s) for s in shape])}]"
            for name, shape in zip(self.input_args, self.input_shapes)
        ]
        # inputs
        res = f"def forward({', '.join(args)})".format()
        # outputs
        # FIXME: Update return type (can use shape propagation)
        res += f" -> float32[{', '.join([str(s) for s in self.input_shapes[0]])}]:\n"
        # global parameters
        if self.named_params:
            for name, param in self.named_params:
                new_name = name.replace(".", "_")
                res += f"  {new_name}: float32[{', '.join([str(s) for s in param.shape])}] = g_{new_name}\n"
        # function body
        for line in self.code:
            res += f"  {line}\n"
        return res

    def __call__(self, node):
        method = getattr(self, "build_" + node.op)
        ret = method(node)
        if ret:
            self.code.append(ret)
        return ret

    def get_module(self, name):
        return dict(self.gm.named_modules())[name]

    def build_placeholder(self, node):
        self.input_args.append(node.name)

    def build_getattr(self, node):
        pass

    def build_call_module(self, node):
        if isinstance(self.get_module(node.target), torch.nn.Linear):
            op = "linear"
        return getattr(self, f"build_{op}")(node)

    def build_call_function(self, node):
        opcls = {
            operator.add: "add",
            operator.sub: "sub",
            operator.mul: "mul",
            F.relu: "relu",
        }.get(node.target)
        return getattr(self, f"build_{opcls}")(node)

    def build_call_method(self, node):
        pass

    def build_output(self, node):
        return f"return {node.args[0]}"

    def build_add(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} + {rhs}"

    def build_relu(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.relu({inp})"

    def build_linear(self, node):
        inp = get_var_name(node.args[0])
        weight = get_var_name(node.target + "_weight")
        bias = get_var_name(node.target + "_bias")
        return f"{node.name} = dsl.linear({inp}, {weight}, {bias})"
