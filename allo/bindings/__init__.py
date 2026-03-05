# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import (
        ir,
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
        allo,
        transform,
    )
else:
    _PUBLIC_SUBMODULES = (
        "ir",
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
        "allo",
        "transform",
    )
    _liballo_module = None

    def _get_liballo_module():
        global _liballo_module  # pylint: disable=global-statement
        if _liballo_module is None:
            from . import _liballo as loaded_module

            _liballo_module = loaded_module
        return _liballo_module

    def __getattr__(name: str) -> Any:
        if name not in _PUBLIC_SUBMODULES:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
        liballo = _get_liballo_module()
        if name == "transform":
            liballo._initialize_transform_bindings()
        value = getattr(liballo, name)
        globals()[name] = value
        return value

    def __dir__():
        return sorted(_PUBLIC_SUBMODULES)
