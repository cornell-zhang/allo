# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unexpected-keyword-arg, no-value-for-parameter

import functools
from types import FunctionType as PyFunctionType
from hcl_mlir.ir import (
    StringAttr,
    InsertionPoint,
    FlatSymbolRefAttr,
    Location,
    UnitAttr,
    MemRefType,
    FunctionType,
)
from hcl_mlir.dialects import func as func_d, memref as memref_d
import numpy as np

from .customize import customize, _get_global_vars


def move_stream_to_interface(func):
    stream_ops = []
    stream_types = []
    for op in func.entry_block.operations:
        if (
            isinstance(op, memref_d.AllocOp)
            and "stream" in MemRefType(op.result.type).memory_space.value
        ):
            stream_ops.append(op)
            stream_types.append(MemRefType(op.result.type))
    if len(stream_ops) == 0:
        return func
    in_types = func.attributes["function_type"].value.inputs
    out_types = func.attributes["function_type"].value.results
    in_types += stream_types
    func_type = FunctionType.get(in_types, out_types)
    func_op = func_d.FuncOp(
        name=func.attributes["sym_name"].value,
        type=func_type,
        ip=InsertionPoint(func),
    )
    func_op.add_entry_block()
    func_op.attributes["itypes"] = func.attributes["itypes"]
    func_op.attributes["otypes"] = func.attributes["otypes"]
    # copy function operations
    cnt_stream = 0
    for op in func.entry_block.operations:
        if op in stream_ops:
            op.result.replace_all_uses_with(
                func_op.arguments[len(func_op.entry_block.arguments) - 1 + cnt_stream]
            )
            cnt_stream += 1
            continue
        op.operation.clone(InsertionPoint(func_op.entry_block))
    for i, arg in enumerate(func.arguments):
        arg.replace_all_uses_with(func_op.arguments[i])
    func.operation.erase()
    return func_op, stream_types


def kernel(mapping=None):
    def top():
        # Just for locating insertion point
        pass

    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # *args and **kwargs are the actual arguments that are passed into the kernel function
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
                assert len(mapping) <= 2, "Only support 1D/2D mapping now."
                for dim in np.ndindex(*mapping):
                    global_vars = new_global_vars.copy()
                    if len(dim) == 1:
                        global_vars.update({"df.pi": dim[0]})
                        new_func_name = func.__name__ + f"_{dim[0]}"
                    else:
                        global_vars.update({"df.pi": dim[0], "df.pj": dim[1]})
                        new_func_name = func.__name__ + f"_{dim[0]}_{dim[1]}"
                    s = customize(
                        func, global_vars=global_vars, context=s_top.module.context
                    )
                    s.top_func, _ = move_stream_to_interface(s.top_func)
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
                    new_func_name = func.__name__ + f"_{'_'.join(map(str, dim))}"
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
            return exe(*args, **kwargs)

        return wrapper

    return actual_decorator


def get_pid():
    raise NotImplementedError("This function should be called in a kernel function.")


def pipe():
    raise NotImplementedError("This function should be called in a kernel function.")


def build(funcs):
    def top():
        # Just for locating insertion point
        pass

    # construct a common module
    s_top = customize(top)
    with s_top.module.context, Location.unknown():
        input_types = []
        func_info = {}
        for func in funcs:
            s = customize(func.__wrapped__, context=s_top.module.context)
            input_types += s.top_func.attributes["function_type"].value.inputs
            s.top_func, stream_types = move_stream_to_interface(s.top_func)
            s.top_func.operation.clone(InsertionPoint(s_top.top_func))
            func_info[s.top_func.attributes["sym_name"].value] = [stream_types[0]]
        func_type = FunctionType.get(input_types, [])
        top_func = func_d.FuncOp(
            name="top", type=func_type, ip=InsertionPoint(s_top.top_func)
        )
        top_func.attributes["itypes"] = s.top_func.attributes["itypes"]
        top_func.attributes["otypes"] = s.top_func.attributes["otypes"]
        top_func.add_entry_block()
        func_d.ReturnOp([], ip=InsertionPoint(top_func.entry_block))
        # create global stream ops
        for func_name, stream_types in func_info.items():
            new_op = memref_d.AllocOp(
                stream_types[0],
                [],
                [],
                ip=InsertionPoint.at_block_terminator(top_func.entry_block),
            )
            func_info[func_name] = [new_op.result]
            break
        for i, (func_name, stream_op) in enumerate(func_info.items()):
            func_d.CallOp(
                [],
                FlatSymbolRefAttr.get(func_name),
                [top_func.arguments[i]] + [func_info["producer"][0]],
                ip=InsertionPoint.at_block_terminator(top_func.entry_block),
            )
        s_top.top_func.attributes["dataflow"] = UnitAttr.get()
        s_top.top_func.operation.erase()
        s_top.top_func = top_func
    print(s_top.module)
