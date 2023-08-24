# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    import torch
    from torch import fx
    from torch.nn import functional as F
    import operator
except ImportError:
    pass


def from_pytorch(model):
    gm = fx.symbolic_trace(model)
    print(gm.graph)
    builder = TorchBuilder(gm)
    code = builder.build()
    print(code)


def get_var_name(node):
    if isinstance(node, fx.Node):
        return node.name
    else:
        return node


class TorchBuilder:
    def __init__(self, gm):
        self.gm = gm
        self.code = []
        self.input_args = []

    def build(self):
        for node in self.gm.graph.nodes:
            self(node)
        res = "def forward({}):\n".format(", ".join(self.input_args))
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
        return f"return {node.name}"

    def build_add(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} + {rhs}"

    def build_relu(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = allo.relu({inp})"
