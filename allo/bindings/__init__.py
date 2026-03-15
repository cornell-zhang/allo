# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=import-self

from typing import TYPE_CHECKING
import importlib

__all__ = [
    "ir",
    "utils",
    "arith",
    "math",
    "scf",
    "cf",
    "ub",
    "func",
    "affine",
    "tensor",
    "memref",
    "linalg",
    "transform",
]

_LAZY_SUBMODULES = frozenset(
    {
        "utils",
        "arith",
        "math",
        "scf",
        "cf",
        "ub",
        "func",
        "affine",
        "tensor",
        "memref",
        "linalg",
        "transform",
    }
)

if TYPE_CHECKING:
    from . import (
        ir,
        utils,
        arith,
        math,
        scf,
        cf,
        ub,
        func,
        affine,
        tensor,
        memref,
        linalg,
        transform,
    )
else:
    ir = importlib.import_module(".ir", __name__)

    def __getattr__(name: str):
        if name == "ir":
            return ir
        if name in _LAZY_SUBMODULES:
            mod = importlib.import_module(f".{name}", __name__)
            globals()[name] = mod
            return mod
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__():
        return sorted(set(globals()) | set(__all__))
