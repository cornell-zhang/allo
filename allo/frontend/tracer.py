# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools

try:
    import torch
    from torch.fx import Tracer, Proxy
except ImportError:
    pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/utils/fx.py
def _gen_constructor_wrapper(target):
    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = None

        def check_has_proxy(v):
            if isinstance(v, Proxy):
                nonlocal proxy
                proxy = v

        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        if proxy is not None:
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        return target(*args, **kwargs)

    return wrapper, target


# https://github.com/pytorch/pytorch/issues/51803
class AlloTracer(Tracer):
    _TORCH_METHODS_TO_PATCH = [
        "arange",
        "zeros",
        "ones",
        "full",
        "full_like",
        "eye",
        "empty",
        "tensor",
        "clamp",
        "finfo",
    ]

    def __init__(self, model, concrete_args):
        super().__init__()
        self.patched_torch_methods = {
            target: _gen_constructor_wrapper(getattr(torch, target))
            for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()
        self.model = model
        self.concrete_args = concrete_args

    def trace(self):
        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)
        try:
            graph = super().trace(self.model, self.concrete_args)
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)
        return graph
