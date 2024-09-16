# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unexpected-keyword-arg, no-value-for-parameter

from types import FunctionType as PyFunctionType
from hcl_mlir.ir import (
    StringAttr,
    InsertionPoint,
    FlatSymbolRefAttr,
    Location,
    UnitAttr,
)
from hcl_mlir.dialects import func as func_d
import numpy as np

from .customize import customize, _get_global_vars


def kernel(mapping=None):
    def top():
        # Just for locating insertion point
        pass

    def decorator(func):
        # construct a common module
        s_top = customize(top)
        global_vars = _get_global_vars(func)
        new_global_vars = global_vars.copy()
        for var in global_vars.values():
            # import functions from other files
            if isinstance(var, PyFunctionType):
                new_global_vars.update(_get_global_vars(var))
        # call different PE kernels
        with s_top.module.context, Location.unknown():
            for dim in np.ndindex(*mapping):
                global_vars = new_global_vars.copy()
                global_vars.update({"df.pi": dim[0], "df.pj": dim[1]})
                s = customize(
                    func, global_vars=global_vars, context=s_top.module.context
                )
                new_func_name = func.__name__ + f"_{dim[0]}_{dim[1]}"
                s.top_func.attributes["sym_name"] = StringAttr.get(new_func_name)
                s.top_func.operation.clone(InsertionPoint(s_top.top_func))
            top_func = func_d.FuncOp(
                name="top", type=s.top_func.type, ip=InsertionPoint(s_top.top_func)
            )
            top_func.add_entry_block()
            top_func.attributes["itypes"] = s.top_func.attributes["itypes"]
            top_func.attributes["otypes"] = s.top_func.attributes["otypes"]
            func_d.ReturnOp([], ip=InsertionPoint(top_func.entry_block))
            for dim in np.ndindex(*mapping):
                new_func_name = func.__name__ + f"_{dim[0]}_{dim[1]}"
                func_d.CallOp(
                    [],
                    FlatSymbolRefAttr.get(new_func_name),
                    top_func.arguments,
                    ip=InsertionPoint.at_block_terminator(top_func.entry_block),
                )
            top_func.attributes["dataflow"] = UnitAttr.get()
        s_top.top_func.operation.erase()
        s_top.top_func = top_func
        print(s_top.module)
        exe = s_top.build()
        return exe

    return decorator


def get_pid():
    raise NotImplementedError("This function should be called in a kernel function.")
