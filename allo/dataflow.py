# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unexpected-keyword-arg, no-value-for-parameter, global-variable-not-assigned, global-statement, broad-exception-caught, too-many-arguments

import functools
from ._mlir.ir import (
    InsertionPoint,
    FlatSymbolRefAttr,
    Location,
    UnitAttr,
    StringAttr,
    FunctionType,
    MemRefType,
)
from ._mlir.dialects import func as func_d, allo as allo_d
from ._mlir.passmanager import PassManager as mlir_pass_manager
from .customize import customize as _customize
from .ir.utils import get_global_vars, get_all_df_kernels
from .backend.ai_engine import AIEModule

from .backend.simulator import LLVMOMPModule
from .ir.types import Stream
from .passes import df_pipeline
from .backend.experimental import AIE_MLIRModule


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
    funcs = get_all_df_kernels(s)
    new_func_args = s.func_args.copy()

    for func in funcs:
        func_name = func.attributes["sym_name"].value
        stream_ops = []
        stream_types = []
        stream_signed = ""
        stream_info[func_name] = []
        in_types = func.attributes["function_type"].value.inputs
        out_types = func.attributes["function_type"].value.results
        s_type_str = "_" * len(in_types)
        new_args = new_func_args[func_name].copy()
        for op in func.entry_block.operations:
            if isinstance(op, allo_d.StreamConstructOp):
                stream_ops.append(op)
                stream_types.append(op.result.type)
                stream_name = op.attributes["name"].value
                stream_signed += "u" if "unsigned" in op.attributes else "_"
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
                new_args.append(stream_name)
        # create new func to update arguments
        in_types += stream_types
        new_func_args[func_name] = new_args
        with s.module.context, Location.unknown():
            func_type = FunctionType.get(in_types, out_types)
            new_func = func_d.FuncOp(
                name=func.attributes["sym_name"].value,
                type=func_type,
                ip=InsertionPoint(func),
            )
            new_func.add_entry_block()
            final_op = func_d.ReturnOp([], ip=InsertionPoint(new_func.entry_block))
            # copy old attributes
            if "itypes" in func.attributes:
                new_func.attributes["itypes"] = StringAttr.get(
                    func.attributes["itypes"].value + stream_signed
                )
            if "otypes" in func.attributes:
                new_func.attributes["otypes"] = func.attributes["otypes"]
            # tag stream types
            new_func.attributes["stypes"] = StringAttr.get(s_type_str)
            if "df.kernel" in func.attributes:
                new_func.attributes["df.kernel"] = UnitAttr.get()
            # move operations from old func to new func
            cnt_stream = 0
            for op in func.entry_block.operations:
                if op in stream_ops:
                    op.result.replace_all_uses_with(
                        new_func.arguments[len(in_types) - len(stream_ops) + cnt_stream]
                    )
                    cnt_stream += 1
                    op.operation.erase()
                    continue
                op.operation.move_before(final_op)
            final_op.erase()
            # update original arguments
            for i, arg in enumerate(func.arguments):
                arg.replace_all_uses_with(new_func.arguments[i])
            func.operation.erase()
    s.func_args = new_func_args
    return stream_info


def remove_unused_func_ops(s, func_names):
    for func_op in s.module.body.operations:
        if not (
            isinstance(func_op, func_d.FuncOp)
            and func_op.attributes["sym_name"].value in func_names
        ):
            continue
        blocks = func_op.body.blocks
        if (
            len(blocks) == 1
            and len(blocks[0].operations) == 1
            and blocks[0].operations[0].name == "func.return"
        ):
            s.func_args.pop(func_op.attributes["sym_name"].value)
            func_op.erase()


def _build_top(s, stream_info, target="vitis_hls"):
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
    funcs = get_all_df_kernels(s)
    input_types = []
    input_signed = ""
    arg_mapping = {}
    used_args = {}  # {arg_name: arg_idx in top_func}
    for func in funcs:
        func_name = func.attributes["sym_name"].value
        arg_mapping[func_name] = []
        for i, arg in enumerate(func.arguments):
            if "!allo.stream" not in str(arg.type):
                arg_name = s.func_args[func_name][i].name
                if arg_name not in used_args:
                    used_args[arg_name] = len(input_types)
                    dtensor = s.func_args[func_name][i]
                    input_types.append((dtensor.shape, dtensor.dtype))
                    if "itypes" in func.attributes:
                        input_signed += func.attributes["itypes"].value[i]
                    s.func_args[s.top_func_name].append(s.func_args[func_name][i])
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
    assert top_func is not None, "Top function not found"
    with s.module.context, Location.unknown():
        # create new func
        func_type = FunctionType.get(
            [MemRefType.get(shape, dtype.build()) for shape, dtype in input_types], []
        )
        new_top = func_d.FuncOp(
            name=s.top_func_name, type=func_type, ip=InsertionPoint(top_func)
        )
        new_top.attributes["itypes"] = StringAttr.get(input_signed)
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
            if target != "aie":
                call_op = func_d.CallOp(
                    [],
                    FlatSymbolRefAttr.get(func_name),
                    arg_lst + stream_lst,
                    ip=InsertionPoint.at_block_terminator(new_top.entry_block),
                )
                if i == len(funcs) - 1:
                    call_op.attributes["last"] = UnitAttr.get()
        new_top.attributes["dataflow"] = UnitAttr.get()
    s.top_func = new_top
    return s


# record kernel mapping in the current region context
_current_region_context = None


def kernel(mapping=None):

    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # *args and **kwargs are the actual arguments that are passed into the kernel function
            func.mapping = mapping
            hls_mod = build(funcs=[func])
            return hls_mod(*args, **kwargs)

        wrapper.mapping = mapping
        global _current_region_context
        if _current_region_context is not None:
            _current_region_context[func.__name__] = mapping
        return wrapper

    return actual_decorator


def region():

    def actual_decorator(func):
        # TODO: ideally this context information should be recorded in the builder
        global _current_region_context
        _current_region_context = {}
        # The function call is only to collect the kernel mapping info, use try except to avoid any exception
        try:
            func()
        except Exception as _:
            pass
        func.mappings = _current_region_context
        _current_region_context = None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # *args and **kwargs are the actual arguments that are passed into the kernel function
            hls_mod = build(funcs=[func])
            return hls_mod(*args, **kwargs)

        wrapper.mappings = func.mappings
        return wrapper

    return actual_decorator


def df_primitive_default(s):
    df_pipeline(s.module, rewind=True)


def customize(func, opt_default=True, enable_tensor=False):
    global_vars = get_global_vars(func)
    s = _customize(func, global_vars=global_vars, enable_tensor=enable_tensor)
    stream_info = move_stream_to_interface(s)
    s = _build_top(s, stream_info, enable_tensor)

    if opt_default:
        df_primitive_default(s)

    return s


def build(
    func,
    target="vitis_hls",
    mode="csim",
    project="top.prj",
    configs=None,
    wrap_io=True,
    opt_default=True,
    enable_tensor=False,
    profile=False,
    warmup=20,
    num_iters=100,
):
    assert (
        not profile or target == "aie-mlir"
    ), "Profiling is only supported for AIE target"
    if target == "aie":
        global_vars = get_global_vars(func)
        s = _customize(func, global_vars=global_vars, enable_tensor=False)
        stream_info = move_stream_to_interface(s)
        s = _build_top(s, stream_info, target=target)
        mod = AIEModule(
            s.module,
            s.top_func_name,
            s.func_args,
            project,
            stream_info,
        )
        mod.build()
        return mod

    if target == "aie-mlir":
        global_vars = get_global_vars(func)
        s = _customize(func, global_vars=global_vars, enable_tensor=False)
        stream_info = move_stream_to_interface(s)
        s = _build_top(s, stream_info, target="aie")
        aie_mod = AIE_MLIRModule(
            s.module, s.top_func_name, s.func_args, project, stream_info, s.ext_libs
        )
        aie_mod.build(
            profile=profile,
            warmup=warmup,
            num_iters=num_iters,
        )
        return aie_mod

    if target == "simulator":
        s = customize(func, opt_default)
        return LLVMOMPModule(s.module, s.top_func_name)
    # FPGA backend
    s = customize(func, opt_default, enable_tensor=enable_tensor)
    hls_mod = s.build(
        target=target,
        mode=mode,
        project=project,
        configs=configs,
        wrap_io=wrap_io,
    )
    return hls_mod
