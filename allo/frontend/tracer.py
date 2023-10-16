# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

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

    def __init__(self, model, concrete_args, leaf_modules=None):
        super().__init__()
        self.patched_torch_methods = {
            target: _gen_constructor_wrapper(getattr(torch, target))
            for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()
        self.model = model
        self.concrete_args = concrete_args
        self.leaf_modules = leaf_modules

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.leaf_modules and isinstance(m, self.leaf_modules):
            return True
        return m.__module__.startswith("torch.nn") and not isinstance(
            m, torch.nn.Sequential
        )

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
