# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from .._mlir.ir import Context
from .._mlir.dialects import allo as allo_d
from ..customize import customize


def unify_kernels(func1: Callable, func2: Callable, loop_num: int):
    mlir_ctx = Context()
    allo_d.register_dialect(mlir_ctx)
    s1 = customize(func1, context=mlir_ctx)
    s2 = customize(func2, context=mlir_ctx)
    unified_module = allo_d.unify_kernels(s1.module, s2.module, loop_num)
    return unified_module
