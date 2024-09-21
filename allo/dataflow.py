# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unexpected-keyword-arg, no-value-for-parameter

import functools
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

from .customize import customize
from .ir.utils import get_global_vars


def get_pid():
    raise NotImplementedError("This function should be called in a kernel function.")


def pipe():
    raise NotImplementedError("This function should be called in a kernel function.")


def move_stream_to_interface(func):
    stream_ops = []
    stream_types = []
    stream_info = []
    for op in func.entry_block.operations:
        if (
            isinstance(op, memref_d.AllocOp)
            and MemRefType(op.result.type).memory_space is not None
            and "stream" in MemRefType(op.result.type).memory_space.value
        ):
            stream_type = MemRefType(op.result.type)
            stream_ops.append(op)
            stream_types.append(stream_type)
            src = op.attributes["src"].value
            dst = op.attributes["dst"].value
            stream_info.append((src, dst, stream_type))
    if len(stream_ops) == 0:
        return func, stream_info
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
    ip = func_d.ReturnOp([], ip=InsertionPoint(func_op.entry_block))
    # copy function operations
    cnt_stream = 0
    for op in func.entry_block.operations:
        if isinstance(op, func_d.ReturnOp):
            break
        if op in stream_ops:
            op.result.replace_all_uses_with(
                func_op.arguments[len(in_types) - len(stream_types) + cnt_stream]
            )
            cnt_stream += 1
            continue
        op.operation.move_before(ip)
    for i, arg in enumerate(func.arguments):
        arg.replace_all_uses_with(func_op.arguments[i])
    func.operation.erase()
    return func_op, stream_info


def _build_top(s_top, input_types, stream_info):
    """
    s_top: schedule of top-level function
    input_types: top-level function input types
    stream_info: {(src, dst): stream_types} (TODO: support more than one stream)
    """
    func_type = FunctionType.get(input_types, [])
    top_func = func_d.FuncOp(
        name="top", type=func_type, ip=InsertionPoint(s_top.top_func)
    )
    top_func.add_entry_block()
    func_d.ReturnOp([], ip=InsertionPoint(top_func.entry_block))
    # create global stream ops
    stream_dict = {}
    for func_name, (stream_lst, start_idx, size) in stream_info.items():
        arg_lst = []
        for src, dst, stream_type in stream_lst:
            if (src, dst) not in stream_dict:
                new_op = memref_d.AllocOp(
                    stream_type,
                    [],
                    [],
                    ip=InsertionPoint.at_block_terminator(top_func.entry_block),
                )
                new_op.attributes["src"] = StringAttr.get(src)
                new_op.attributes["dst"] = StringAttr.get(dst)
                stream_dict[(src, dst)] = new_op
            else:
                new_op = stream_dict[(src, dst)]
            arg_lst.append(new_op.result)
        func_d.CallOp(
            [],
            FlatSymbolRefAttr.get(func_name),
            list(top_func.arguments[start_idx : start_idx + size]) + arg_lst,
            ip=InsertionPoint.at_block_terminator(top_func.entry_block),
        )
    top_func.attributes["dataflow"] = UnitAttr.get()
    s_top.top_func.operation.erase()
    s_top.top_func = top_func
    hls_mod = s_top.build(
        target="vitis_hls",
        mode="csim",
        project="top.prj",
    )
    return hls_mod


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
            global_vars = get_global_vars(func)
            # call same PE kernel with different location IDs
            with s_top.module.context, Location.unknown():
                assert len(mapping) <= 2, "Only support 1D/2D mapping now."
                all_stream_info = {}
                for dim in np.ndindex(*mapping):
                    if len(dim) == 1:
                        global_vars.update({"df.p0": dim[0]})
                        new_func_name = func.__name__ + f"_{dim[0]}"
                    else:
                        global_vars.update({"df.p0": dim[0], "df.p1": dim[1]})
                        new_func_name = func.__name__ + f"_{dim[0]}_{dim[1]}"
                    s = customize(
                        func, global_vars=global_vars, context=s_top.module.context
                    )
                    input_types = s.top_func.attributes["function_type"].value.inputs
                    s.top_func, stream_info = move_stream_to_interface(s.top_func)
                    all_stream_info[new_func_name] = (stream_info, 0, len(input_types))
                    s.top_func.attributes["sym_name"] = StringAttr.get(new_func_name)
                    s.top_func.operation.clone(InsertionPoint(s_top.top_func))
                hls_mod = _build_top(s_top, input_types, all_stream_info)
            return hls_mod(*args, **kwargs)

        return wrapper

    return actual_decorator


def build(funcs):
    def top():
        # Just for locating insertion point
        pass

    # construct a common module
    s_top = customize(top)
    with s_top.module.context, Location.unknown():
        input_types = []
        all_stream_info = {}
        for func in funcs:
            global_vars = get_global_vars(func)
            s = customize(
                func.__wrapped__, global_vars=global_vars, context=s_top.module.context
            )
            start_idx = len(input_types)
            size = len(s.top_func.attributes["function_type"].value.inputs)
            input_types += s.top_func.attributes["function_type"].value.inputs
            s.top_func, stream_info = move_stream_to_interface(s.top_func)
            all_stream_info[func.__name__] = (stream_info, start_idx, size)
            s.top_func.operation.clone(InsertionPoint(s_top.top_func))
        hls_mod = _build_top(s_top, input_types, all_stream_info)
    return hls_mod
