# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

from collections.abc import Callable

from .._mlir.ir import (
    Context,
)
from .._mlir.dialects import (
    allo as allo_d,
)
from ..customize import customize


def unify(func1: Callable, func2: Callable, loop_num: int):
    """
    Unify two kernels by extracting common parts (e.g. for loop) and using conditional branch.
    The new module take one more input than func1 and func2 (they should have same input arguments), and the extra input is an array of int8.
    The unified kernel uses an outer loop to wrap the input kernels, where the number of iteration in the outer loop is loop_num.
    In each iteration, the kernel will execute the branch according to the element in the input array (e.g. 0 for func1, 1 for func2).
    ----------------
    Parameters:
    func1: Callable
        A python function to be unified
    func2: Callable
        A python function to be unified
    loop_num: int
        The number of iteration of the outter loop in the unified kernel
    ----------------
    Returns:
    MLIRModule
        The unified MLIRModule, can be used to create LLVMModule and HLSModule in allo
    """
    mlir_ctx = Context()
    allo_d.register_dialect(mlir_ctx)
    s1 = customize(func1, context=mlir_ctx)
    s2 = customize(func2, context=mlir_ctx)
    unified_module = allo_d.unify_kernels(s1.module, s2.module, loop_num)
    return unified_module
