# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unexpected-keyword-arg, no-value-for-parameter, too-many-nested-blocks

import os
import re
import numpy as np

from ._mlir.ir import (
    Location,
    MemRefType,
    UnrankedMemRefType,
    FunctionType,
    TypeAttr,
    UnitAttr,
    FlatSymbolRefAttr,
    ArrayAttr,
    Attribute,
    AffineMapAttr,
    InsertionPoint,
    Module,
    IntegerAttr,
    IntegerType,
    Operation,
    BlockArgument,
)
from ._mlir.dialects import (
    allo as allo_d,
    func as func_d,
    affine as affine_d,
    memref as memref_d,
    scf as scf_d,
    linalg as linalg_d,
    arith as arith_d,
)
from ._mlir.ir import StringAttr
from ._mlir.passmanager import PassManager as mlir_pass_manager
from .ir.transform import find_func_in_module
from .ir.transform import wrap_data_movement
from .ir.utils import MockBuffer
from .utils import get_mlir_dtype_from_str, c2allo_type


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


# pylint: disable=too-many-branches
def generate_input_output_buffers(module, top_func_name, flatten=False, mappings=None):
    results = {"inputs": [], "outputs": []}
    top_func = find_func_in_module(module, top_func_name)

    if mappings is None:
        mappings = [None] * len(top_func.arguments)

    load_store_mapping = analyze_arg_load_store(module)
    # Build Buffer-Load functions
    load_func_names = []
    with module.context, Location.unknown():
        ip = InsertionPoint(top_func)
        # Create Load function for each input
        for idx, arg in enumerate(top_func.arguments):
            if not isinstance(arg.type, MemRefType):
                load_func_names.append("")
                continue

            if load_store_mapping[top_func_name][idx] in {"in", "both"}:
                func_name = f"load_buf{idx}"
                load_func_names.append(func_name)
                wrap_data_movement(
                    arg,
                    ip,
                    func_name,
                    from_memory=True,
                    flatten=flatten,
                    mapping=mappings[idx],
                )

    # Find ReturnOp
    for op in top_func.entry_block.operations:
        if isinstance(op, func_d.ReturnOp):
            op_return = op
            break

    # Build Buffering functions
    store_func_names = []
    with module.context, Location.unknown():
        if len(mappings) < len(op_return.operands) + len(top_func.arguments):
            mappings += [None] * len(op_return.operands)
        if len(op_return.operands) > 0:  # Return value exist
            ip = InsertionPoint(top_func)
            for idx, res in enumerate(op_return.operands):
                if not isinstance(res.type, MemRefType):
                    store_func_names.append("")
                    continue

                func_name = f"store_res{idx + len(top_func.arguments)}"
                store_func_names.append(func_name)

                wrap_data_movement(
                    res,
                    ip,
                    func_name,
                    from_memory=False,
                    flatten=flatten,
                    mapping=mappings[idx],
                )

        else:
            for idx, arg in enumerate(top_func.arguments):
                if not isinstance(arg.type, MemRefType):
                    # scalar
                    continue
                if load_store_mapping[top_func_name][idx] in {"out", "both"}:
                    ip = InsertionPoint(top_func)
                    func_name = f"store_res{idx}"
                    store_func_names.append(func_name)

                    wrap_data_movement(
                        arg,
                        ip,
                        func_name,
                        from_memory=False,
                        flatten=flatten,
                        mapping=mappings[-1],
                    )

    # Modify Top function
    with top_func.context, Location.unknown():
        ip_first = InsertionPoint(top_func.entry_block.operations[0])

        # Modify Loading
        new_in_types = []
        bufs = {}  # arg idx->buf
        with ip_first:
            for idx, arg in enumerate(top_func.arguments):
                # Process non-MemRefType
                if not isinstance(arg.type, MemRefType):
                    new_in_types.append(arg.type)
                    continue

                # Build AllocOP for buffer
                alloc_op = memref_d.AllocOp(
                    MemRefType(arg.type),
                    [],
                    [],
                )
                alloc_op.attributes["name"] = StringAttr.get(f"buf{idx}")

                # Replace original argument with buffer
                arg.replace_all_uses_with(alloc_op.result)

                # Update shape of arguments in top function
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

                # Build CallOp for buffer loading
                if load_store_mapping[top_func_name][idx] in {
                    "in",
                    "both",
                }:
                    func_d.CallOp(
                        [],
                        FlatSymbolRefAttr.get(load_func_names[idx]),
                        [arg, alloc_op.result],
                    )
                    results["inputs"].append(MockBuffer(top_func_name, f"buf{idx}"))

                # Record buffers
                bufs[idx] = alloc_op

        # Modify Storing
        new_out_types = []
        ip_return = InsertionPoint(op_return)
        if len(op_return.operands) > 0:  # Return Value Exist
            for idx, arg in enumerate(op_return.operands):
                # Process non-MemRefType
                if not isinstance(arg.type, MemRefType):
                    new_out_types.append(arg.type)
                    continue

                # Build AllocOP for buffer
                if not flatten:
                    store_memref = MemRefType(arg.type)
                else:
                    store_memref = MemRefType.get(
                        (np.prod(MemRefType(arg.type).shape),),
                        MemRefType(arg.type).element_type,
                    )

                alloc_op = memref_d.AllocOp(
                    store_memref,
                    [],
                    [],
                    ip=ip_first,
                )

                alloc_op.attributes["name"] = StringAttr.get(
                    f"res{idx + len(top_func.arguments)}"
                )

                # Update returnop
                op_return.operation.replace_uses_of_with(arg, alloc_op.result)
                new_out_types.append(alloc_op.result.type)

                # Build CallOp for buffer loading
                func_d.CallOp(
                    [],
                    FlatSymbolRefAttr.get(store_func_names[idx]),
                    [arg, alloc_op.result],
                    ip=ip_return,
                )

                results["outputs"].append(
                    MockBuffer(top_func_name, arg.owner.attributes["name"].value)
                )

        else:
            # argument as output
            for idx, arg in enumerate(top_func.arguments):
                if not isinstance(arg.type, MemRefType):
                    continue
                if load_store_mapping[top_func_name][idx] in {"out", "both"}:
                    func_name = f"store_res{idx}"
                    func_d.CallOp(
                        [],
                        FlatSymbolRefAttr.get(func_name),
                        [bufs[idx].result, arg],
                        ip=ip_return,
                    )

                    results["outputs"].append(MockBuffer(top_func_name, f"result{idx}"))

        func_type = FunctionType.get(new_in_types, new_out_types)
        top_func.attributes["function_type"] = TypeAttr.get(func_type)

    return results


# pylint: disable=dangerous-default-value
def analyze_arg_load_store_in_func(func, mapping={}):
    res = []
    if func.is_external:
        return ["in"] * len(func.type.inputs)
    for _, arg in enumerate(func.arguments):
        if not isinstance(arg.type, MemRefType):
            res.append("scalar")
            continue
        # 10: in, 01: out, 11: both, 00: func
        io_type = 0
        for use in arg.uses:
            if isinstance(
                use.owner, (memref_d.LoadOp, affine_d.AffineLoadOp, allo_d.StreamGetOp)
            ):
                io_type |= 2
            elif isinstance(
                use.owner,
                (memref_d.StoreOp, affine_d.AffineStoreOp, allo_d.StreamPutOp),
            ):
                io_type |= 1
            elif isinstance(use.owner, func_d.CallOp):
                callee = use.owner.attributes["callee"].value
                if callee in mapping:
                    callee_arg_type = mapping[callee][use.operand_number]
                    if callee_arg_type == "out":
                        io_type |= 1
                    elif callee_arg_type == "in":
                        io_type |= 2
                    elif callee_arg_type == "both":
                        io_type |= 3
                    else:
                        io_type |= 0
        match io_type:
            case 1:
                res.append("out")
            case 2:
                res.append("in")
            case 3:
                res.append("both")
            case 0:
                res.append("func")
    return res


def analyze_arg_load_store(mod):
    res = {}
    for func in mod.body.operations:
        if not isinstance(func, func_d.FuncOp):
            continue
        func_res = analyze_arg_load_store_in_func(func, res)
        res[func.attributes["sym_name"].value] = func_res
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
                    ele_type = get_mlir_dtype_from_str(c2allo_type[arg_type])
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


def analyze_use_def(mod):
    ret_vals = {}  # func_name -> return value
    # union find structure
    uf_mapping = {}  # name -> label ID
    parent = []

    def uf_find(i):
        if parent[i] == i:
            return i
        # path compression
        result = uf_find(parent[i])
        parent[i] = result
        return result

    def uf_union(name_i, name_j):
        parent[uf_find(uf_mapping[name_i])] = uf_find(uf_mapping[name_j])

    def uf_add(name):
        if name not in uf_mapping:
            uf_mapping[name] = len(uf_mapping)
            parent.append(uf_mapping[name])

    def recover_sets():
        if len(parent) == 0:
            return []
        res = [set() for _ in range(max(parent) + 1)]
        for buf, label in uf_mapping.items():
            while label != parent[label]:
                label = parent[label]
            res[label].add(buf)
        return res

    def add_use(val, val_name):
        vals = [val]
        if isinstance(val.type, MemRefType) and val.type.rank == 0:
            # scalar, handle the following case
            # %alloc_2 = memref.alloc() {name = "D"} : memref<i32>
            # affine.store %20, %alloc_2[] {to = "D"} : memref<i32>
            # %21 = affine.load %alloc_2[] {from = "D"} : memref<i32>
            # %22 = call @foo(%21) : (i32) -> i32
            for use in val.uses:
                if isinstance(use.owner, (memref_d.LoadOp, affine_d.AffineLoadOp)):
                    vals.append(use.owner.result)
        # pylint: disable=redefined-argument-from-local
        for val in vals:
            if isinstance(val.owner, Operation) and "func.call" in str(val.owner):
                # not sure why cannot use isinstance(val.owner, func_d.CallOp)
                # return value
                callee = val.owner.attributes["callee"].value
                uf_union(val_name, ret_vals[callee])
                uf_union(ret_vals[callee], val_name)
            for use in val.uses:
                if isinstance(use.owner, func_d.CallOp):
                    callee = use.owner.attributes["callee"].value
                    target_name = f"{callee}:{use.operand_number}"
                    uf_union(val_name, target_name)
                    uf_union(target_name, val_name)

    for func in mod.body.operations:
        if not isinstance(func, func_d.FuncOp):
            continue
        func_name = func.attributes["sym_name"].value
        for i, arg in enumerate(func.arguments):
            arg_name = f"{func_name}:{i}"
            uf_add(arg_name)
            add_use(arg, arg_name)
        for op in func.entry_block.operations:
            if isinstance(op, (memref_d.AllocOp, func_d.CallOp, memref_d.GetGlobalOp)):
                if "name" in op.attributes:
                    buf_name = f"{func_name}:{op.attributes['name'].value}"
                elif " = " in str(op):
                    buf_name = f"{func_name}:{str(op).split(' = ', maxsplit=1)[0]}"
                else:
                    # call op does not have return value
                    continue
                uf_add(buf_name)
                add_use(op.result, buf_name)
            if isinstance(op, func_d.ReturnOp):
                for i, ret in enumerate(op.operands):
                    owner = ret.owner
                    if "name" in owner.attributes:
                        buf_name = f"{func_name}:{owner.attributes['name'].value}"
                    elif "from" in owner.attributes:
                        buf_name = f"{func_name}:{owner.attributes['from'].value}"
                    elif " = " in str(owner):
                        buf_name = (
                            f"{func_name}:{str(owner).split(' = ', maxsplit=1)[0]}"
                        )
                    ret_vals[func_name] = buf_name

    # recover final sets
    res = recover_sets()
    return res


def analyze_read_write_patterns(mlir_func, external_kernel_lib: dict = {}):
    """
    Analyze the read/write patterns of function arguments to determine which are inputs and outputs.
    Handles subview operations and common linalg operations.

    Parameters:
    -----------
    mlir_func: An MLIR function operation

    Returns:
    --------
    tuple: (input_indices, output_indices) lists of argument indices
    """
    input_indices = set()
    output_indices = set()

    # Track subview operations to map derived views back to original memrefs
    subview_map = {}  # Maps Value IDs of subviews to their source argument indices
    for arg in mlir_func.arguments:
        work_list = [arg]
        visited = set()
        while work_list:
            value = work_list.pop()
            if value in visited:
                continue
            visited.add(value)

            for use in value.uses:
                user = use.owner
                if user.name == "memref.subview":
                    subview_map[user.result] = arg.arg_number
                    work_list.append(user.result)

    # Helper to resolve a value to its original argument index if it's a subview
    def resolve_to_func_arg_index(value):
        if BlockArgument.isinstance(value):
            arg = BlockArgument(value)
            if isinstance(arg.owner.owner, func_d.FuncOp):
                return arg.arg_number
        if value in subview_map:
            return subview_map[value]
        return None

    # Dictionary of common linalg operations and their input/output patterns
    # Pattern format: (number_of_inputs, number_of_outputs)
    linalg_op_patterns = {
        "linalg.broadcast": (1, 1),  # 1 inputs, 1 output
        "linalg.transpose": (1, 1),  # 1 inputs, 1 output
        "linalg.matmul": (2, 1),  # 2 inputs, 1 output
        "linalg.batch_matmul": (2, 1),  # 2 inputs, 1 output
        "linalg.conv_2d_nchw_fchw": (2, 1),  # 2 inputs, 1 output
        "linalg.pooling_nchw_max": (2, 1),  # 2 inputs, 1 output
        "linalg.pooling_nchw_sum": (2, 1),  # 2 inputs, 1 output
        "linalg.add": (2, 1),  # 2 inputs, 1 output
        "linalg.mul": (2, 1),  # 2 inputs, 1 output
        "linalg.div": (2, 1),  # 2 inputs, 1 output
        "linalg.sub": (2, 1),  # 2 inputs, 1 output
        "linalg.max": (2, 1),  # 2 inputs, 1 output
        "linalg.min": (2, 1),  # 2 inputs, 1 output
        "linalg.fill": (1, 1),  # 1 input (value), 1 output (buffer)
        "linalg.exp": (1, 1),  # 1 inputs, 1 output
        "linalg.log": (1, 1),  # 1 inputs, 1 output
        "linalg.abs": (1, 1),  # 1 inputs, 1 output
        "linalg.copy": (1, 1),  # 1 input, 1 output
        "linalg.conv_2d": (2, 1),  # 2 inputs (input, kernel), 1 output
        "linalg.pooling": (1, 1),  # 1 input, 1 output
        "linalg.softmax": (1, 1),  # 1 input, 1 output
        "linalg.generic": None,  # Special handling required
        "linalg.indexed_generic": None,  # Special handling required
    }

    # Recursively walk through all operations in the function body
    def walk_operations(block):
        for op in block.operations:
            op_name = str(op.operation.name)

            # user defined external kernel
            if isinstance(op, func_d.CallOp) and op.callee.value in external_kernel_lib:
                callee_name = op.callee.value
                ext_module = external_kernel_lib[callee_name]
                for idx in ext_module.input_idx:
                    input_indices.add(resolve_to_func_arg_index(op.operands[idx]))
                for idx in ext_module.output_idx:
                    output_indices.add(resolve_to_func_arg_index(op.operands[idx]))
            elif op_name in {"memref.load", "affine.load", "allo.load_slice"}:
                if len(op.operands) > 0:
                    input_indices.add(resolve_to_func_arg_index(op.operands[0]))
            elif op_name in {"memref.store", "affine.store"}:
                if len(op.operands) > 1:
                    output_indices.add(resolve_to_func_arg_index(op.operands[1]))
            elif op_name == "allo.store_slice":
                assert len(op.operands) >= 2
                input_indices.add(resolve_to_func_arg_index(op.operands[0]))
                output_indices.add(resolve_to_func_arg_index(op.operands[1]))
            elif op_name == "memref.copy" and len(op.operands) >= 2:
                # First operand is source, second is destination
                input_indices.add(resolve_to_func_arg_index(op.operands[0]))
                output_indices.add(resolve_to_func_arg_index(op.operands[1]))
            # Handle linalg operations using the pattern dictionary
            elif op_name.startswith("linalg."):
                pattern = linalg_op_patterns.get(op_name)
                # Handle operations with known patterns
                if pattern is not None:
                    num_ins, num_outs = pattern
                    total_operands = len(op.operands)
                    # First check if the ins/outs are marked explicitly
                    explicit_ins_outs = False
                    # Check for explicit ins/outs attributes or regions
                    if "ins" in op.attributes and "outs" in op.attributes:
                        explicit_ins_outs = True
                        ins_attr = op.attributes["ins"]
                        outs_attr = op.attributes["outs"]
                        # Parse the attributes to get indices
                        ins_indices = []
                        outs_indices = []
                        if hasattr(ins_attr, "__iter__"):
                            ins_indices = list(ins_attr)
                        if hasattr(outs_attr, "__iter__"):
                            outs_indices = list(outs_attr)
                        # Check if our argument is used in any of these positions
                        for idx in ins_indices:
                            if idx < total_operands:
                                input_indices.add(
                                    resolve_to_func_arg_index(op.operands[idx])
                                )

                        for idx in outs_indices:
                            if idx < total_operands:
                                output_indices.add(
                                    resolve_to_func_arg_index(op.operands[idx])
                                )

                    # If there are no explicit attributes, use the pattern
                    if not explicit_ins_outs:
                        # First num_ins operands are inputs
                        for i in range(min(num_ins, total_operands)):
                            input_indices.add(resolve_to_func_arg_index(op.operands[i]))
                        # Last num_outs operands are outputs
                        for i in range(
                            max(0, total_operands - num_outs), total_operands
                        ):
                            output_indices.add(
                                resolve_to_func_arg_index(op.operands[i])
                            )

                # Special handling for generic operations
                elif op_name in {"linalg.generic", "linalg.indexed_generic"}:
                    # These operations have explicit indexing_maps that define inputs and outputs
                    # We need to look at the string representation to determine usage
                    op_str = str(op)
                    arg_refs = re.findall(r"%arg\d+", op_str)
                    for arg_ref in arg_refs:
                        index = int(arg_ref.group(1))
                        if (
                            "ins(" in op_str
                            and arg_ref in op_str.split("ins(")[1].split("outs(")[0]
                        ):
                            input_indices.add(index)
                        if (
                            "outs(" in op_str
                            and arg_ref in op_str.split("outs(")[1].split(")")[0]
                        ):
                            output_indices.add(index)

            # Fallback: Check if any argument is directly used in any other operation
            else:
                for operand in op.operands:
                    input_indices.add(resolve_to_func_arg_index(operand))

            # Recursively process nested regions if any
            for region in op.regions:
                for inner_block in region.blocks:
                    walk_operations(inner_block)

    # Start walking from the function's entry block
    for block in mlir_func.body.blocks:
        walk_operations(block)

    memref_input_indices, memref_output_indices = set(), set()
    for idx in input_indices:
        if idx is not None and isinstance(mlir_func.arguments[idx].type, MemRefType):
            memref_input_indices.add(idx)
    for idx in output_indices:
        if idx is not None and isinstance(mlir_func.arguments[idx].type, MemRefType):
            memref_output_indices.add(idx)
    # Parameters that are both read and written are considered outputs
    pure_inputs = list(memref_input_indices - memref_output_indices)
    return pure_inputs, list(memref_output_indices)


def df_pipeline(module, initiation_interval=1, rewind=False):

    def pipe_loop_innermost(forop, ii, rewind):
        inner_forops = []
        for op in forop.body.operations:
            if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
                inner_forops.append(op)
        if inner_forops:
            for inner_forop in inner_forops:
                pipe_loop_innermost(inner_forop, ii, rewind)
        else:
            forop.attributes["pipeline_ii"] = ii
            if rewind:
                forop.attributes["rewind"] = UnitAttr.get()
            # print('Pipeline Once.')

    with module.context:
        i32 = IntegerType.get_unsigned(32)
        ii = IntegerAttr.get(i32, initiation_interval)
        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp):
                func = op
                for op_ in func.entry_block.operations:
                    if isinstance(op_, (scf_d.ForOp, affine_d.AffineForOp)):
                        pipe_loop_innermost(op_, ii, rewind)
