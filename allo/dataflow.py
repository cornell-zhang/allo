# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unexpected-keyword-arg, no-value-for-parameter

import functools

from ._mlir.ir import (
    InsertionPoint,
    FlatSymbolRefAttr,
    Location,
    UnitAttr,
    StringAttr,
    FunctionType,
)
from ._mlir.dialects import func as func_d, allo as allo_d
from ._mlir.passmanager import PassManager as mlir_pass_manager
from .customize import customize
from .ir.utils import get_global_vars, get_all_funcs_except_top
from .backend.aie import AIEModule
from .ir.types import Stream


def get_pid():
    raise NotImplementedError("This function should be called in a kernel function.")


def pipe(dtype, shape=(), depth=2):
    return Stream(dtype, shape, depth)


class Array:
    def __init__(self, element, shape):
        self.element = element
        self.shape = shape


def array(element, shape):
    return Array(element, shape)


def move_stream_to_interface(s):
    stream_info = {}
    funcs = get_all_funcs_except_top(s)

    for func in funcs:
        func_name = func.attributes["sym_name"].value
        stream_ops = []
        stream_types = []
        stream_info[func_name] = []
        in_types = func.attributes["function_type"].value.inputs
        out_types = func.attributes["function_type"].value.results
        s_type_str = "_" * len(in_types)
        for op in func.entry_block.operations:
            if isinstance(op, allo_d.StreamConstructOp):
                stream_ops.append(op)
                stream_types.append(op.result.type)
                stream_name = op.attributes["name"].value
                for use in op.result.uses:
                    # get use's parent operation
                    if isinstance(use.owner, allo_d.StreamGetOp):
                        direction = "in"
                    elif isinstance(use.owner, allo_d.StreamPutOp):
                        direction = "out"
                    else:
                        raise ValueError("Stream is not used correctly.")
                stream_info[func_name].append((stream_name, direction))
                s_type_str += direction[0]
        # create new func to update arguments
        in_types += stream_types
        with s.module.context, Location.unknown():
            func_type = FunctionType.get(in_types, out_types)
            new_func = func_d.FuncOp(
                name=func.attributes["sym_name"].value,
                type=func_type,
                ip=InsertionPoint(func),
            )
            new_func.add_entry_block()
            return_op = func_d.ReturnOp([], ip=InsertionPoint(new_func.entry_block))
            # tag stream types
            new_func.attributes["stypes"] = StringAttr.get(s_type_str)
            # move operations from old func to new func
            cnt_stream = 0
            for op in func.entry_block.operations:
                if isinstance(op, func_d.ReturnOp):
                    break
                if op in stream_ops:
                    op.result.replace_all_uses_with(
                        new_func.arguments[len(in_types) - len(stream_ops) + cnt_stream]
                    )
                    cnt_stream += 1
                    op.operation.erase()
                    continue
                op.operation.move_before(return_op)
            # update original arguments
            for i, arg in enumerate(func.arguments):
                arg.replace_all_uses_with(new_func.arguments[i])
            func.operation.erase()
    return stream_info


def remove_unused_func_ops(s, func_names):
    for i in range(len(func_names) - 1, -1, -1):
        func_op = s.module.body.operations[i]
        blocks = func_op.body.blocks
        if (
            len(blocks) == 1
            and len(blocks[0].operations) == 1
            and blocks[0].operations[0].name == "func.return"
        ):
            func_op.erase()


def _build_top(s, stream_info):
    """
    s: top-level schedule
    stream_info: {func_name: [(stream_names, direction)]}
    """
    # remove unused kernel
    passes = ["canonicalize"]
    pipeline = f'builtin.module(func.func({",".join(passes)}))'
    try:
        with s.module.context:
            mlir_pass_manager.parse(pipeline).run(s.module.operation)
    except Exception as e:
        print("Error: failed to run MLIR lower pipeline, printing module...")
        print(s.module)
        raise e
    remove_unused_func_ops(s, stream_info.keys())

    # create argument mapping
    funcs = get_all_funcs_except_top(s)
    input_types = []
    arg_mapping = {}
    used_args = {}  # {arg_name: arg_idx in top_func}
    for func in funcs:
        func_name = func.attributes["sym_name"].value
        arg_mapping[func_name] = []
        for i, arg in enumerate(func.arguments):
            if "!allo.stream" not in str(arg.type):
                arg_name = s.func_args[func_name][i]
                if arg_name not in used_args:
                    used_args[arg_name] = len(input_types)
                    input_types.append(arg.type)
                arg_mapping[func_name].append(used_args[arg_name])
    # update top function
    top_func = None
    for func in s.module.body.operations:
        if (
            isinstance(func, func_d.FuncOp)
            and func.attributes["sym_name"].value == s.top_func_name
        ):
            top_func = func
            break
    with s.module.context, Location.unknown():
        # create new func
        func_type = FunctionType.get(input_types, [])
        new_top = func_d.FuncOp(name="top", type=func_type, ip=InsertionPoint(top_func))
        new_top.add_entry_block()
        return_op = func_d.ReturnOp([], ip=InsertionPoint(new_top.entry_block))
        for op in top_func.entry_block.operations:
            if isinstance(op, func_d.ReturnOp):
                break
            op.operation.move_before(return_op)
        top_func.operation.erase()
        # get all global streams
        stream_map = {}
        for op in new_top.entry_block.operations:
            if isinstance(op, allo_d.StreamConstructOp):
                stream_name = op.attributes["name"].value
                stream_map[stream_name] = op
        # add call functions
        for i, func in enumerate(funcs):
            func_name = func.attributes["sym_name"].value
            arg_lst = [new_top.arguments[idx] for idx in arg_mapping[func_name]]
            stream_lst = [
                stream_map[stream_name] for stream_name, _ in stream_info[func_name]
            ]
            call_op = func_d.CallOp(
                [],
                FlatSymbolRefAttr.get(func_name),
                arg_lst + stream_lst,
                ip=InsertionPoint.at_block_terminator(new_top.entry_block),
            )
            if i == len(stream_info) - 1:
                call_op.attributes["last"] = UnitAttr.get()
        new_top.attributes["dataflow"] = UnitAttr.get()
    s.top_func = new_top
    return s


def kernel(mapping=None):

    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # *args and **kwargs are the actual arguments that are passed into the kernel function
            func.mapping = mapping
            hls_mod = build(funcs=[func])
            return hls_mod(*args, **kwargs)

        wrapper.mapping = mapping
        return wrapper

    return actual_decorator


def region():

    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # *args and **kwargs are the actual arguments that are passed into the kernel function
            hls_mod = build(funcs=[func])
            return hls_mod(*args, **kwargs)

        return wrapper

    return actual_decorator


def build(func, target="vitis_hls", mode="csim", project="top.prj"):
    global_vars = get_global_vars(func)
    s = customize(func, global_vars=global_vars)
    if target == "aie":
        mapping = func.mapping
        print(s.module)
        mod = AIEModule(s.module, s.top_func_name, project, mapping)
        mod.build()
        return mod
    # FPGA backend
    stream_info = move_stream_to_interface(s)
    s = _build_top(s, stream_info)
    print(s.module)
    hls_mod = s.build(
        target=target,
        mode=mode,
        project=project,
    )
    return hls_mod
