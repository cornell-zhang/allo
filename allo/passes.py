# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unexpected-keyword-arg, no-value-for-parameter, too-many-nested-blocks

import os
import numpy as np

from ._mlir.ir import (
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
)
from ._mlir.dialects import (
    allo as allo_d,
    func as func_d,
    affine as affine_d,
    memref as memref_d,
    linalg as linalg_d,
    arith as arith_d,
)
from ._mlir.ir import StringAttr
from ._mlir.passmanager import PassManager as mlir_pass_manager
from .ir.transform import create_buffer, store_tensor
from .ir.utils import MockBuffer
from .utils import get_mlir_dtype_from_str


def _mlir_lower_pipeline(module, **kwargs):
    allo_d.loop_transformation(module)
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


def generate_input_output_buffers(top_func, flatten=False, mappings=None):
    res = {"inputs": [], "outputs": []}
    top_func_name = top_func.attributes["sym_name"].value
    if mappings is None:
        mappings = [None] * len(top_func.arguments)
    with top_func.context, Location.unknown():
        first_op = top_func.entry_block.operations[0]
        new_in_types = []
        in_bufs = []
        for i, arg in enumerate(top_func.arguments):
            if not isinstance(arg.type, MemRefType):
                new_in_types.append(arg.type)
                continue
            buf = create_buffer(
                arg, f"buf{i}", ip=first_op, flatten=flatten, mapping=mappings[i]
            )
            in_bufs.append(buf)
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
                if len(mappings) < len(op.operands) + len(top_func.arguments):
                    mappings += [None] * len(op.operands)
                if len(op.operands) > 0:
                    for i, arg in enumerate(op.operands):
                        if not isinstance(arg.type, MemRefType):
                            new_out_types.append(arg.type)
                            continue
                        buf = create_buffer(
                            arg,
                            f"result{i+len(top_func.arguments)}",
                            ip=op,
                            alloc_ip=first_op,
                            flatten=flatten,
                            mapping=mappings[len(top_func.arguments) + i],
                        )
                        # update returnop
                        op.operation.replace_uses_of_with(arg, buf.result)
                        new_out_types.append(buf.result.type)
                        res["outputs"].append(
                            MockBuffer(
                                top_func_name, arg.owner.attributes["name"].value
                            )
                        )
                else:
                    # the last argument is set as the return value by default
                    store_tensor(
                        in_bufs[-1].result,
                        top_func.arguments[-1],
                        f"result{len(top_func.arguments)}",
                        ip=op,
                        flatten=flatten,
                    )
                    res["outputs"].append(
                        MockBuffer(top_func_name, f"result{len(top_func.arguments)}")
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
                for arg_type, shape in obj.args:
                    ele_type = get_mlir_dtype_from_str(arg_type)
                    if len(shape) == 0:
                        memref = ele_type
                    else:
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
                            if not isinstance(arg.type, MemRefType):
                                continue
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
