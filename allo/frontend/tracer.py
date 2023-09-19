# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# https://github.com/huggingface/transformers/blob/main/src/transformers/utils/fx.py

import torch
from torch.fx import Tracer
from transformers.utils.fx import _gen_constructor_wrapper


# https://github.com/pytorch/pytorch/issues/51803
class CustomedTracer(Tracer):
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
