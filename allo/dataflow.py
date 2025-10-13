# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unexpected-keyword-arg, no-value-for-parameter, global-variable-not-assigned, global-statement, broad-exception-caught

import functools
import itertools
import os
from typing import Union
from ._mlir.ir import (
    InsertionPoint,
    FlatSymbolRefAttr,
    Location,
    UnitAttr,
    StringAttr,
    FunctionType,
    MemRefType,
    Type,
)
from ._mlir.dialects import func as func_d, allo as allo_d
from ._mlir.passmanager import PassManager as mlir_pass_manager
from .customize import customize as _customize, Schedule
from .utils import parse_kernel_name, construct_kernel_name
from .ir.utils import get_global_vars, get_all_df_kernels
from .backend.simulator import LLVMOMPModule
from .ir.types import Stream
from .passes import df_pipeline
from .backend import AIE_MLIRModule


def gather(pipes: list):
    """
    Collect all pipe objects from the given list (explicit list or slice) in their original order.
    """
    raise NotImplementedError("This function should be called in a kernel function.")


def scatter(buffer, pipes: list):
    """
    Distribute data to all pipe objects in the given list (explicit list or slice)
    in their original order.
    """
    raise NotImplementedError("This function should be called in a kernel function.")


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


# pylint: disable=eval-used, bad-builtin, too-many-branches, too-many-nested-blocks
def move_stream_to_interface(
    s: Schedule,
    with_stream_type: bool = False,
    with_extra_info: bool = False,
    unroll=True,
):
    stream_info = {}
    if with_extra_info:
        extra_stream_info = {}
    funcs = get_all_df_kernels(s)
    new_func_args = s.func_args.copy()
    if with_stream_type:
        stream_types_dict: dict[str, Type] = {}
    for func in funcs:
        func_name = func.attributes["sym_name"].value
        stream_ops = []
        stream_types = []
        stream_signed = ""
        stream_info[func_name] = []
        if with_extra_info:
            extra_stream_info[func_name] = {}
        in_types = func.attributes["function_type"].value.inputs
        out_types = func.attributes["function_type"].value.results
        s_type_str = "_" * len(in_types)
        new_args = new_func_args[func_name].copy()
        skip_new_args_flag = False
        prefix, ids = parse_kernel_name(func_name)
        if not unroll:
            assert s.func_instances is not None and prefix in s.func_instances
            assert ids in s.func_instances[prefix].keys()
            for ids_, predicate_tag in s.func_instances[prefix].items():
                func_name_ = construct_kernel_name(prefix, ids_)
                if (
                    func_name_ != func_name
                    and predicate_tag == s.func_instances[prefix][ids]
                ):
                    stream_info[func_name_] = []
                    if with_extra_info:
                        extra_stream_info[func_name_] = {}
                    new_func_args[func_name_] = new_func_args[func_name].copy()
        for op in func.entry_block.operations:
            if isinstance(op, allo_d.StreamConstructOp):
                stream_ops.append(op)
                stream_types.append(op.result.type)
                stream_signed += "u" if "unsigned" in op.attributes else "_"
                for use in op.result.uses:
                    # get use's parent operation
                    if isinstance(use.owner, allo_d.StreamGetOp):
                        direction = "in"
                    elif isinstance(use.owner, allo_d.StreamPutOp):
                        direction = "out"
                    else:
                        raise ValueError("Stream is not used correctly.")
                stream_name = op.attributes["name"].value
                if with_stream_type and stream_name not in stream_types_dict:
                    stream_types_dict[stream_name] = op.result.type
                stream_info[func_name].append((stream_name, direction))
                s_type_str += direction[0]
                new_args.append(stream_name)
                if not unroll and "symbolic_slice" in op.attributes:
                    symbolic_name = op.attributes["symbolic_slice"].value
                    arg_idx = len(new_args) - 1
                    for ids_, predicate_tag in s.func_instances[prefix].items():
                        func_name_ = construct_kernel_name(prefix, ids_)
                        if func_name_ == func_name:
                            skip_new_args_flag = True
                        if predicate_tag == s.func_instances[prefix][ids]:
                            pid_map = {
                                f"p{idx}": value for idx, value in enumerate(ids_)
                            }
                            loops = []
                            if "iterators" in op.attributes:
                                for name_attr in op.attributes["iterators"]:
                                    rargs = []
                                    for val in name_attr.attr:
                                        rargs.append(val.value)
                                    loops.append((name_attr.name, range(*rargs)))

                            def eval_stream(
                                symbol_map_,
                                pid_map_,
                                symbolic_name_,
                                org_stream_name,
                                op_,
                                arg_idx_,
                                init_iter_map=None,
                            ):
                                iter_map = {}
                                if init_iter_map is not None:
                                    iter_map.update(init_iter_map)
                                for k, v in symbol_map_.items():
                                    if k not in pid_map_:
                                        iter_map[k] = v
                                slice_ = eval(symbolic_name_, symbol_map_)
                                if isinstance(slice_, int):
                                    slice_ = tuple([slice_])
                                parts = org_stream_name.rsplit("_", len(slice_))[
                                    : -len(slice_)
                                ]
                                stream_name_ = f"{"_".join(map(str, parts))}_{"_".join(map(str, slice_))}"
                                if (
                                    with_stream_type
                                    and stream_name_ not in stream_types_dict
                                ):
                                    stream_types_dict[stream_name_] = op_.result.type
                                if len(new_func_args[func_name_]) == arg_idx_:
                                    new_func_args[func_name_].append([])
                                stream_info[func_name_].append(
                                    (stream_name_, direction)
                                )
                                if with_extra_info:
                                    extra_stream_info[func_name_][
                                        stream_name_
                                    ] = iter_map
                                new_func_args[func_name_][-1].append(stream_name_)

                            if len(loops) == 0:
                                if "stream_list" in op.attributes:
                                    stream_list = op.attributes["stream_list"]
                                    stream_symbolic_slice_list = op.attributes[
                                        "stream_symbolic_slice_list"
                                    ]
                                    loop_name = op.attributes["loop_name"].value
                                    for idx, (name_, symbolic_name_) in enumerate(
                                        zip(stream_list, stream_symbolic_slice_list)
                                    ):
                                        eval_stream(
                                            pid_map,
                                            pid_map_=pid_map,
                                            symbolic_name_=symbolic_name_.value,
                                            org_stream_name=name_.value,
                                            op_=op,
                                            arg_idx_=arg_idx,
                                            init_iter_map={loop_name: idx},
                                        )
                                else:
                                    eval_stream(
                                        pid_map,
                                        pid_map_=pid_map,
                                        symbolic_name_=symbolic_name,
                                        org_stream_name=stream_name,
                                        op_=op,
                                        arg_idx_=arg_idx,
                                    )
                            else:
                                for combo in itertools.product(
                                    *[rng for _, rng in loops]
                                ):
                                    iter_symbol_map = pid_map.copy()
                                    for (name, _), val in zip(loops, combo):
                                        iter_symbol_map[name] = val
                                    eval_stream(
                                        iter_symbol_map,
                                        pid_map_=pid_map,
                                        symbolic_name_=symbolic_name,
                                        org_stream_name=stream_name,
                                        op_=op,
                                        arg_idx_=arg_idx,
                                    )

        # create new func to update arguments
        in_types += stream_types
        # skip_new_args_flag: updated with symbolic info, do not use `new_args`
        if not skip_new_args_flag:
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
            if "tag" in func.attributes:
                new_func.attributes["tag"] = StringAttr.get(
                    func.attributes["tag"].value
                )
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
    if with_stream_type:
        if with_extra_info:
            return stream_info, stream_types_dict, extra_stream_info
        return stream_info, stream_types_dict
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


def _build_top(s, stream_info, target="vitis_hls", get_parameter_list: bool = False):
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
    if get_parameter_list:
        return used_args, s
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


def customize(func, enable_tensor=False, opt_default=False):
    global_vars = get_global_vars(func)
    s = _customize(func, global_vars=global_vars, enable_tensor=enable_tensor)
    stream_info = move_stream_to_interface(s)
    s = _build_top(s, stream_info, enable_tensor)

    if opt_default:
        df_primitive_default(s)

    return s


# pylint: disable=too-many-arguments
def build(
    func,
    target="vitis_hls",
    mode="csim",
    project="top.prj",
    configs=None,
    wrap_io=True,
    enable_tensor=False,
    mapping_primitives: list[tuple[str, list]] = None,
    profile=False,
    warmup=20,
    num_iters=100,
    trace: list[tuple[str, tuple[int, ...]]] = None,
    trace_size: int = 4096,
    device_type: Union[str, None] = None,
):
    assert not profile or target == "aie", "Profiling is only supported for AIE target"
    assert (
        trace is None or target == "aie"
    ), "Trace profiling is only supported for AIE target"

    if target == "aie":
        global_vars = get_global_vars(func)
        # [NOTE]: set unroll = False to improve compilation efficiency
        s: Schedule = _customize(
            func, global_vars=global_vars, enable_tensor=False, unroll=False
        )
        stream_info, stream_types_dict, extra_stream_info = move_stream_to_interface(
            s, with_stream_type=True, with_extra_info=True, unroll=False
        )
        parameter_list, s = _build_top(
            s, stream_info, target=target, get_parameter_list=True
        )
        aie_mod = AIE_MLIRModule(
            s.module,
            s.top_func_name,
            parameter_list,
            s.func_args,
            project,
            stream_info,
            stream_types_dict,
            s.ext_libs,
            s.func_instances,
            extra_stream_info=extra_stream_info,
        )
        if device_type is None:
            if os.getenv("NPU2") == "1":
                device_type = "npu2"
            else:
                device_type = "npu1_4col"
        aie_mod.build(
            device_type=device_type,
            mapping_primitives=mapping_primitives,
            profile=profile,
            warmup=warmup,
            num_iters=num_iters,
            trace=trace,
            trace_size=trace_size,
        )
        return aie_mod

    if target == "simulator":
        s = customize(func)
        return LLVMOMPModule(s.module, s.top_func_name)
    # FPGA backend
    s = customize(func, enable_tensor=enable_tensor)
    hls_mod = s.build(
        target=target,
        mode=mode,
        project=project,
        configs=configs,
        wrap_io=wrap_io,
    )
    return hls_mod
