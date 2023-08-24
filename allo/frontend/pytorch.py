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
    builder = TorchBuilder(gm, example_inputs)
    code = builder.build()
    global_vars = {}
    for pymod in [types]:
        global_vars.update({item[0]: item[1] for item in inspect.getmembers(pymod)})
    global_vars.update({"dsl": dsl})
    s = customize(code, verbose=verbose, global_vars=global_vars)
    if verbose:
        print(s.module)
    return s.build()


def get_var_name(node):
    if isinstance(node, fx.Node):
        return node.name
    else:
        return node


class TorchBuilder:
    def __init__(self, gm, example_inputs):
        self.gm = gm
        self.code = []
        self.input_args = []
        self.input_shapes = [x.shape for x in example_inputs]

    def build(self):
        for node in self.gm.graph.nodes:
            self(node)
        args = [
            f"{name}: float32[{','.join(map(str, shape))}]"
            for name, shape in zip(self.input_args, self.input_shapes)
        ]
        # inputs
        res = "def forward({})".format(", ".join(args))
        # outputs
        # FIXME: Update return type (can use shape propagation)
        res += " -> float32[{}]:\n".format(",".join(map(str, self.input_shapes[0])))
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

    def build_placeholder(self, node):
        self.input_args.append(node.name)

    def build_getattr(self, node):
        pass

    def build_call_module(self, node):
        pass

    def build_call_function(self, node):
        opcls = {
            operator.add: "add",
            operator.sub: "sub",
            operator.mul: "mul",
            F.relu: "relu",
        }.get(node.target)
        return getattr(TorchBuilder, "build_" + opcls)(self, node)

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
