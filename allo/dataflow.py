# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unexpected-keyword-arg, no-value-for-parameter

import gc
import functools
import numpy as np

from ._mlir.ir import (
    StringAttr,
    InsertionPoint,
    FlatSymbolRefAttr,
    Location,
    UnitAttr,
    MemRefType,
    FunctionType,
)
from ._mlir.dialects import func as func_d, memref as memref_d
from ._mlir.passmanager import PassManager as mlir_pass_manager
from .customize import customize
from .ir.utils import get_global_vars
from .backend.aie import AIEModule


def get_pid():
    raise NotImplementedError("This function should be called in a kernel function.")


def pipe():
    raise NotImplementedError("This function should be called in a kernel function.")


def move_stream_to_interface(func):
    stream_ops = []
    stream_types = []
    stream_info = []
    used_stream_ops = []
    used_stream_types = []
    used_stream_info = []

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

    # Check if the stream op is used in the function, remove if not
    for op, stype, sinfo in zip(stream_ops, stream_types, stream_info):
        if len(list(op.result.uses)) != 0:
            used_stream_ops.append(op)
            used_stream_types.append(stype)
            used_stream_info.append(sinfo)
        else:
            op.operation.erase()

    if len(used_stream_ops) == 0:
        return func, used_stream_info

    in_types = func.attributes["function_type"].value.inputs
    out_types = func.attributes["function_type"].value.results

    in_types += used_stream_types
    func_type = FunctionType.get(in_types, out_types)

    func_op = func_d.FuncOp(
        name=func.attributes["sym_name"].value,
        type=func_type,
        ip=InsertionPoint(func),
    )
    func_op.add_entry_block()
    ip = func_d.ReturnOp([], ip=InsertionPoint(func_op.entry_block))

    cnt_stream = 0
    total_args = len(func_op.arguments)

    for op in func.entry_block.operations:
        if isinstance(op, func_d.ReturnOp):
            break
        if op in used_stream_ops:
            op.result.replace_all_uses_with(
                func_op.arguments[total_args - len(used_stream_types) + cnt_stream]
            )
            cnt_stream += 1
            op.operation.erase()
            continue
        op.operation.move_before(ip)
    for i, arg in enumerate(func.arguments):
        arg.replace_all_uses_with(func_op.arguments[i])

    func.operation.erase()
    return func_op, used_stream_info


def tag_stream_type(func, func_name, stream_info, start_index):
    if "stypes" in func.attributes:
        stypes = func.attributes["stypes"].value.split()
    else:
        stypes = ["_"] * start_index
    for src, dst, _ in stream_info:
        if func_name == src:
            stypes.append("o")
        elif func_name == dst:
            stypes.append("i")
        else:
            stypes.append("g")
    func.attributes["stypes"] = StringAttr.get("".join(stypes))


def remove_unused_func_ops(s_top, func_names):
    for i in range(len(func_names) - 1, -1, -1):
        func_op = s_top.module.body.operations[i]
        blocks = func_op.body.blocks
        if (
            len(blocks) == 1
            and len(blocks[0].operations) == 1
            and blocks[0].operations[0].name == "func.return"
        ):
            func_op.erase()
            del func_names[i]


def _build_top(s_top, input_types, stream_info, func_names):
    """
    s_top: schedule of top-level function
    input_types: top-level function input types
    stream_info: {(src, dst): stream_types} (TODO: support more than one stream)
    func_names: all kernel function names (top not included)
    """
    s_top.top_func.operation.erase()
    # remove unused kernel
    passes = ["canonicalize"]
    pipeline = f'builtin.module(func.func({",".join(passes)}))'
    try:
        with s_top.module.context:
            mlir_pass_manager.parse(pipeline).run(s_top.module.operation)
    except Exception as e:
        print("Error: failed to run MLIR lower pipeline, printing module...")
        print(s_top.module)
        raise e
    remove_unused_func_ops(s_top, func_names)
    ip = InsertionPoint(s_top.module.body)
    func_type = FunctionType.get(input_types, [])
    top_func = func_d.FuncOp(name="top", type=func_type, ip=ip)
    top_func.add_entry_block()
    func_d.ReturnOp([], ip=InsertionPoint(top_func.entry_block))
    # create global stream ops
    stream_dict = {}
    for i, func_name in enumerate(func_names):
        stream_lst, indices = stream_info[func_name]
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
        arguments = []
        for idx in indices:
            arguments.append(top_func.arguments[idx])
        call_op = func_d.CallOp(
            [],
            FlatSymbolRefAttr.get(func_name),
            arguments + arg_lst,
            ip=InsertionPoint.at_block_terminator(top_func.entry_block),
        )
        # tag the last callOp for tapa invoke
        if i == len(func_names) - 1:
            call_op.attributes["last"] = UnitAttr.get()
    top_func.attributes["dataflow"] = UnitAttr.get()
    s_top.top_func = top_func
    return s_top


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


def build(funcs, target="vitis_hls", mode="csim", project="top.prj"):
    if target == "aie":
        assert not isinstance(funcs, list), "Only support one function for AIE target."
        func = funcs
        global_vars = get_global_vars(func)
        s = customize(func, global_vars=global_vars)
        print(s.module)
        mod = AIEModule(s.module, s.top_func_name, project)
        mod.build()
        return mod

    def top():
        # Just for locating insertion point
        pass

    # construct a common module
    s_top = customize(top)
    input_types = []
    all_stream_info = {}
    # collect all funcOp
    func_names = []
    # mapping from arg name to arg index
    top_func_arg_mapping = {}
    if not isinstance(funcs, list):
        funcs = [funcs]
    with s_top.module.context, Location.unknown():
        for func in funcs:
            global_vars = get_global_vars(func)
            mapping = func.mapping
            assert len(mapping) <= 2, "Only support 1D/2D mapping now."
            for dim in np.ndindex(*mapping):
                # A randomly crashed bug
                # https://github.com/cornell-zhang/allo/issues/196
                gc.collect()
                if len(dim) == 1:
                    global_vars.update({"df.p0": dim[0]})
                    new_func_name = func.__name__ + f"_{dim[0]}"
                else:
                    global_vars.update({"df.p0": dim[0], "df.p1": dim[1]})
                    new_func_name = func.__name__ + f"_{dim[0]}_{dim[1]}"
                s = customize(
                    func.__wrapped__ if hasattr(func, "__wrapped__") else func,
                    global_vars=global_vars,
                    context=s_top.module.context,
                )
                indices = []
                for idx, arg in enumerate(s.func_args[func.__name__]):
                    if not arg in top_func_arg_mapping:
                        top_func_arg_mapping[arg] = len(top_func_arg_mapping)
                        input_types.append(
                            s.top_func.attributes["function_type"].value.inputs[idx]
                        )
                    indices.append(top_func_arg_mapping[arg])
                s.top_func, stream_info = move_stream_to_interface(s.top_func)
                tag_stream_type(s.top_func, new_func_name, stream_info, len(indices))
                all_stream_info[new_func_name] = (stream_info, indices)
                s.top_func.attributes["sym_name"] = StringAttr.get(new_func_name)
                s.top_func.operation.clone(InsertionPoint(s_top.top_func))
                func_names.append(new_func_name)
        s_top = _build_top(s_top, input_types, all_stream_info, func_names)
        hls_mod = s_top.build(
            target=target,
            mode=mode,
            project=project,
        )
    return hls_mod
