# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    import torch
    from torch import fx
except ImportError:
    pass


def from_pytorch(model):
    gm = fx.symbolic_trace(model)
    print(gm.graph)


class TorchBuilder:
    def __call__(self, node):
        method = getattr(self, "build_" + node.op)
        return method(node)

    def build_placeholder():
        pass

    def build_getattr():
        pass

    def build_call_module():
        pass

    def build_call_function():
        pass

    def build_call_method():
        pass

    def build_output():
        pass
