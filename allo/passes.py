# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unexpected-keyword-arg, no-value-for-parameter, too-many-nested-blocks

import os
import re
import math
import numpy as np
from tabulate import tabulate

from hcl_mlir.ir import (
    Location,
    MemRefType,
    UnrankedMemRefType,
    FunctionType,
    TypeAttr,
    FlatSymbolRefAttr,
    ArrayAttr,
    Attribute,
    AffineMapAttr,
    InsertionPoint,
    Module,
    IntegerAttr,
    IntegerType,
    IndexType,
)
from hcl_mlir.dialects import (
    hcl as hcl_d,
    func as func_d,
    affine as affine_d,
    memref as memref_d,
    linalg as linalg_d,
    arith as arith_d,
    scf as scf_d,
)
from hcl_mlir.ir import StringAttr
from hcl_mlir.passmanager import PassManager as mlir_pass_manager
from .ir.transform import create_buffer
from .ir.utils import MockBuffer
from .utils import get_mlir_dtype_from_str


def _mlir_lower_pipeline(module, **kwargs):
    hcl_d.loop_transformation(module)
    passes = ["affine-loop-normalize", "cse", "affine-simplify-structures"]
    if "canonicalize" in kwargs:
        passes += ["canonicalize"]
    if "lower_linalg" in kwargs:
        passes += ["convert-linalg-to-affine-loops"]
    pipeline = f'builtin.module(func.func({",".join(passes)}))'
    try:
        with module.context:
            mlir_pass_manager.parse(pipeline).run(module.operation)
        return module
    except Exception as e:
        print("Error: failed to run MLIR lower pipeline, printing module...")
        print(module)
        raise e


def lower_linalg_and_attach_names(module):
    op_names = []
    cnt_loop_nests = 0

    def is_linalg_op(op):
        return isinstance(
            op,
            (
                linalg_d.BatchMatmulOp,
                linalg_d.MatmulOp,
                linalg_d.SoftmaxOp,
                linalg_d.GenericOp,
                linalg_d.FillOp,
                linalg_d.AddOp,
                linalg_d.SubOp,
                linalg_d.DivOp,
                linalg_d.ExpOp,
                linalg_d.LogOp,
                linalg_d.AbsOp,
                linalg_d.TransposeOp,
                linalg_d.BroadcastOp,
            ),
        )

    def annotate_affine_for(op):
        nonlocal cnt_unnamed, cnt_loop_nests
        if isinstance(op, affine_d.AffineForOp):
            if ("loop_name" not in op.attributes) and ("op_name" not in op.attributes):
                if cnt_unnamed == 0:
                    op.attributes["op_name"] = StringAttr.get(op_names[cnt_loop_nests])
                loop_name = f"L_{cnt_unnamed}"
                cnt_unnamed += 1
                op.attributes["loop_name"] = StringAttr.get(loop_name)
            annotate_affine_for(op.body.operations[0])

    with module.context:
        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp):
                func = op
                for op_ in func.entry_block.operations:
                    if is_linalg_op(op_) or isinstance(op_, affine_d.AffineForOp):
                        op_names.append(op_.attributes["op_name"].value)

        _mlir_lower_pipeline(module, lower_linalg=True)
        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp):
                func = op
                for op_ in func.entry_block.operations:
                    cnt_unnamed = 0
                    annotate_affine_for(op_)
                    if isinstance(op_, affine_d.AffineForOp):
                        cnt_loop_nests += 1


def generate_input_output_buffers(top_func, flatten=False):
    res = {"inputs": [], "outputs": []}
    top_func_name = top_func.attributes["sym_name"].value
    with top_func.context, Location.unknown():
        first_op = top_func.entry_block.operations[0]
        new_in_types = []
        for i, arg in enumerate(top_func.arguments):
            create_buffer(arg, f"buf{i}", ip=first_op, flatten=flatten)
            if flatten:
                old_memref = MemRefType(arg.type)
                new_memref = MemRefType.get(
                    (np.prod(old_memref.shape),),
                    old_memref.element_type,
                )
                arg.set_type(new_memref)
                new_in_types.append(new_memref)
            else:
                new_in_types.append(arg.type)
            res["inputs"].append(MockBuffer(top_func_name, f"buf{i}"))
        # find return op
        new_out_types = []
        for op in top_func.entry_block.operations:
            if isinstance(op, func_d.ReturnOp):
                for i, arg in enumerate(op.operands):
                    buf = create_buffer(
                        arg,
                        f"result{i+len(top_func.arguments)}",
                        ip=op,
                        alloc_ip=first_op,
                        flatten=flatten,
                    )
                    # update returnop
                    op.operation.replace_uses_of_with(arg, buf.result)
                    new_out_types.append(buf.result.type)
                    res["outputs"].append(
                        MockBuffer(top_func_name, arg.owner.attributes["name"].value)
                    )
                break
        func_type = FunctionType.get(new_in_types, new_out_types)
        top_func.attributes["function_type"] = TypeAttr.get(func_type)
    return res


def decompose_library_function(module):
    with module.context, Location.unknown():
        # get all functions from origin module and find the function to replace
        body_op_to_remove = []
        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp) and not op.is_external:
                for body_op in op.entry_block.operations:
                    # put function into the module
                    if isinstance(body_op, linalg_d.SoftmaxOp):
                        generate_call_module(body_op, op, "softmax")
                        body_op_to_remove.append(body_op)
                    if isinstance(body_op, func_d.CallOp):
                        callee_value = body_op.attributes["callee"].value
                        if callee_value.startswith(("gelu", "layernorm", "tril")):
                            name = callee_value.split("_")[0]
                        else:
                            continue
                        generate_call_module(body_op, op, name)
                        body_op_to_remove.append(body_op)
            elif op.attributes["sym_name"].value.startswith(
                ("gelu", "layernorm", "tril")
            ):
                body_op_to_remove.append(op)
        # need to erase at the end
        for op in body_op_to_remove:
            op.operation.erase()
        return module


def call_ext_libs_in_ptr(module, ext_libs):
    lib_map = {lib.top: lib for lib in ext_libs}
    with module.context, Location.unknown():
        op_to_remove = []
        for op in module.body.operations:
            if (
                isinstance(op, func_d.FuncOp)
                and op.is_external
                and op.attributes["sym_name"].value in lib_map
            ):
                obj = lib_map[op.attributes["sym_name"].value]
                # external functions, reconstruct func type
                input_types = []
                for arg_type, _ in obj.args:
                    ele_type = get_mlir_dtype_from_str(arg_type)
                    memref = UnrankedMemRefType.get(ele_type, None)
                    input_types.append(memref)
                func_type = FunctionType.get(input_types, [])
                func_op = func_d.FuncOp(
                    name=obj.lib_name, type=func_type, ip=InsertionPoint(op)
                )
                func_op.attributes["sym_visibility"] = StringAttr.get("private")
                op_to_remove.append(op)
            elif isinstance(op, func_d.FuncOp):
                for body_op in op.entry_block.operations:
                    # update call function
                    if (
                        isinstance(body_op, func_d.CallOp)
                        and body_op.attributes["callee"].value in lib_map
                    ):
                        obj = lib_map[body_op.attributes["callee"].value]
                        for arg in body_op.operands:
                            memref = UnrankedMemRefType.get(arg.type.element_type, None)
                            cast = memref_d.CastOp(
                                memref, arg, ip=InsertionPoint(body_op)
                            )
                            body_op.operation.replace_uses_of_with(arg, cast.result)
                        # update callee name
                        body_op.attributes["callee"] = FlatSymbolRefAttr.get(
                            obj.lib_name
                        )
            elif op.attributes["sym_name"].value.startswith(("ext_libs")):
                op_to_remove.append(op)
        # need to erase at the end
        for op in op_to_remove:
            op.operation.erase()
        return module


def generate_call_module(target_op, func_op, name):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, f"ir/template/{name}_impl.mlir")
    with open(file_path, "r", encoding="utf-8") as f:
        template = f.read()
    op_to_remove = []
    mod = Module.parse(template)
    func = mod.body.operations[0]
    if name == "softmax":
        sym_name = f"{name}_{hash(target_op)}"
        args = func.arguments
        args[0].set_type(target_op.input.type)
        args[1].set_type(target_op.output.type)
        in_types = [args[0].type, args[1].type]
        out_types = [args[1].type]
        operands = [target_op.input, target_op.output]
    elif name in {"gelu", "layernorm", "tril"}:
        sym_name = target_op.attributes["callee"].value
        in_types = [arg.type for arg in target_op.operands_]
        for i, arg in enumerate(func.arguments):
            arg.set_type(in_types[i])
        out_types = [in_types[0]]
        operands = target_op.operands_
    func.attributes["sym_name"] = StringAttr.get(sym_name)
    func_type = FunctionType.get(in_types, out_types)
    func.attributes["function_type"] = TypeAttr.get(func_type)
    func.move_before(func_op)
    call_op = func_d.CallOp(
        out_types,
        FlatSymbolRefAttr.get(sym_name),
        operands,
        ip=InsertionPoint(target_op),
    )
    if name in {"gelu", "layernorm", "tril"}:
        target_op.result.replace_all_uses_with(call_op.result)

    for op in func.entry_block.operations:
        shape = MemRefType(out_types[0]).shape
        if name == "softmax":
            if isinstance(op, memref_d.AllocOp):
                alloc_op = memref_d.AllocOp(
                    MemRefType.get(
                        shape[:-1],
                        MemRefType(in_types[0]).element_type,
                    ),
                    [],
                    [],
                    ip=InsertionPoint(op),
                )
                op.result.replace_all_uses_with(alloc_op.result)
                op_to_remove.append(op)

            elif isinstance(op, linalg_d.GenericOp):
                update_generic_op(op, name, shape)
        elif name == "gelu":
            if isinstance(op, memref_d.AllocOp):
                alloc_op = memref_d.AllocOp(
                    MemRefType(out_types[0]),
                    [],
                    [],
                    ip=InsertionPoint(op),
                )
                op.result.replace_all_uses_with(alloc_op.result)
                op_to_remove.append(op)
            elif isinstance(op, linalg_d.GenericOp):
                update_generic_op(op, name, shape)
        elif name == "layernorm":
            if isinstance(op, memref_d.AllocOp):
                if op.attributes["name"].value == "output":
                    new_type = out_types[0]
                elif op.attributes["name"].value in {"mean", "mean2", "var"}:
                    new_type = MemRefType.get(
                        shape[:-1],
                        MemRefType(out_types[0]).element_type,
                    )
                alloc_op = memref_d.AllocOp(
                    new_type,
                    [],
                    [],
                    ip=InsertionPoint(op),
                )
                op.result.replace_all_uses_with(alloc_op.result)
                op_to_remove.append(op)
            elif isinstance(op, arith_d.ConstantOp):
                if op.attributes["name"].value == "dimension":
                    const_dtype = IntegerType.get_signless(32)
                    const_value = IntegerAttr.get(const_dtype, shape[-1])
                    # pylint: disable=too-many-function-args
                    const_op = arith_d.ConstantOp(
                        const_dtype,
                        const_value,
                        ip=InsertionPoint(op),
                    )
                    op.result.replace_all_uses_with(const_op.result)
                    op_to_remove.append(op)
            elif isinstance(op, linalg_d.GenericOp):
                update_generic_op(op, name, shape)
        elif name == "tril":
            if isinstance(op, memref_d.AllocOp):
                alloc_op = memref_d.AllocOp(
                    MemRefType.get(
                        shape,
                        MemRefType(in_types[0]).element_type,
                    ),
                    [],
                    [],
                    ip=InsertionPoint(op),
                )
                op.result.replace_all_uses_with(alloc_op.result)
                op_to_remove.append(op)
    # need to erase at the end
    for op in op_to_remove:
        op.operation.erase()


def update_generic_op(op, name, shape):
    in_str = ", ".join([f"d{i}" for i in range(len(shape))])
    out_str = ", ".join([f"d{i}" for i in range(len(shape) - 1)])
    affine_map_in = AffineMapAttr.parse(f"affine_map<({in_str})->({in_str})>")
    affine_map_out = AffineMapAttr.parse(f"affine_map<({in_str})->({out_str})>")
    affine_map_out2 = AffineMapAttr.parse(f"affine_map<({out_str})->({out_str})>")
    affine_map_out3 = AffineMapAttr.parse(f"affine_map<({in_str})->(d{len(shape)-1})>")
    iter_types_0 = [Attribute.parse("#linalg.iterator_type<parallel>")] * (
        len(shape) - 1
    ) + [Attribute.parse("#linalg.iterator_type<reduction>")]
    iter_types_1 = [Attribute.parse("#linalg.iterator_type<parallel>")] * len(shape)
    if name == "gelu":
        op.attributes["indexing_maps"] = ArrayAttr.get([affine_map_in, affine_map_in])
        op.attributes["iterator_types"] = ArrayAttr.get(iter_types_1)
    elif name == "softmax":
        if op.attributes["name"].value in {"max", "add"}:
            op.attributes["indexing_maps"] = ArrayAttr.get(
                [affine_map_in, affine_map_out]
            )
            op.attributes["iterator_types"] = ArrayAttr.get(iter_types_0)
        elif op.attributes["name"].value in {"exp", "div"}:
            op.attributes["indexing_maps"] = ArrayAttr.get(
                [affine_map_in, affine_map_out, affine_map_in]
            )
            op.attributes["iterator_types"] = ArrayAttr.get(iter_types_1)
        else:
            raise NotImplementedError("Unsupported softmax shape")
    elif name == "layernorm":
        if op.attributes["name"].value == "mean":
            op.attributes["indexing_maps"] = ArrayAttr.get(
                [affine_map_in] + [affine_map_out] * 4
            )
            op.attributes["iterator_types"] = ArrayAttr.get(iter_types_0)
        elif op.attributes["name"].value == "var":
            op.attributes["indexing_maps"] = ArrayAttr.get([affine_map_out2] * 5)
            op.attributes["iterator_types"] = ArrayAttr.get(
                [Attribute.parse("#linalg.iterator_type<parallel>")] * (len(shape) - 1)
            )
        elif op.attributes["name"].value == "output":
            op.attributes["indexing_maps"] = ArrayAttr.get(
                [affine_map_out] * 2
                + [affine_map_in]
                + [affine_map_out3] * 2
                + [affine_map_in]
            )
            op.attributes["iterator_types"] = ArrayAttr.get(iter_types_1)
        else:
            raise NotImplementedError("Unsupported gelu shape")
    else:
        raise NotImplementedError("Unsupported function")


def monitor_memory_usage(intermediate_module):
    def find_storeop_in_forop(op):
        result = None
        for body_op in op.body.operations:
            if isinstance(body_op, memref_d.StoreOp):
                result = body_op
            elif isinstance(body_op, scf_d.ForOp):
                result_iter = find_storeop_in_forop(body_op)
                if result is None:
                    if result_iter is not None:
                        result = result_iter
                        break
                    raise NotImplementedError("No storeop found")
        return result

    mem_alloc = {}
    zero_const = []
    table_data = []
    total_alloc_count = 0
    total_memory_bits = 0
    total_bram = 0
    for op in intermediate_module.body.operations:
        if isinstance(op, func_d.FuncOp):
            if not op.is_external:
                for body_op in op.entry_block.operations:
                    # record zero constants
                    if isinstance(body_op, arith_d.ConstantOp):
                        dtype = body_op.type
                        if not isinstance(dtype, IndexType):
                            value = body_op.literal_value
                            if value == 0:
                                name = str(body_op).split("=", maxsplit=1)[0].strip()
                                zero_const.append(name)
                    # record memref.alloc
                    if isinstance(body_op, memref_d.AllocOp):
                        alloc_name = str(body_op).split("=", maxsplit=1)[0].strip()
                        mem_alloc[alloc_name] = []
                        mem_type = body_op.result.type
                        mem_shape = mem_type.shape
                        mem_dtype = str(mem_type.element_type)
                        mem_bits = 1
                        for dim in mem_shape:
                            mem_bits *= dim
                        data_bits = int(re.search(r"\d+", mem_dtype).group())
                        mem_bits *= data_bits
                        bram = math.ceil(mem_bits / (18 * 1024))
                        store_count = 0
                        mem_alloc[alloc_name].append(
                            [mem_shape, mem_dtype, mem_bits, bram, store_count]
                        )
                        total_alloc_count += 1
                        total_memory_bits += mem_bits
                        total_bram += bram
                    # record storage to memref.alloc
                    elif isinstance(body_op, scf_d.ForOp):
                        store_op = find_storeop_in_forop(body_op)
                        if isinstance(store_op, memref_d.StoreOp):
                            value_name = (
                                str(store_op.value.owner)
                                .split("=", maxsplit=1)[0]
                                .strip()
                            )
                            if value_name not in zero_const:
                                memref_name = (
                                    str(store_op.memref.owner)
                                    .split("=", maxsplit=1)[0]
                                    .strip()
                                )
                                if memref_name in mem_alloc:
                                    mem_alloc[memref_name].append(
                                        str(store_op.value.owner)
                                    )
                                    mem_alloc[memref_name][0][-1] += 1
    for key, value in mem_alloc.items():
        table_data.append(
            [
                key,
                value[0][0],
                value[0][1],
                value[0][2],
                value[0][3],
                value[0][4],
                "\n".join(value[1:]),
            ]
        )
    table_data.append(
        [
            "Total(" + str(total_alloc_count) + ")",
            "",
            "",
            total_memory_bits,
            total_bram,
            "",
            "*data storage: data stored into an allocated memory. Doesn't include init.",
        ]
    )
    table_headers = [
        "name",
        "shape",
        "dtype",
        "mem(bits)",
        "BRAM(18K)",
        "store counts",
        "data storage",
    ]
    table = tabulate(table_data, headers=table_headers, tablefmt="grid")
    return str(table)
