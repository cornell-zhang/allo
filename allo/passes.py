# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

import numpy as np

from hcl_mlir.ir import Location, MemRefType, FunctionType, TypeAttr
from hcl_mlir.dialects import (
    hcl as hcl_d,
    func as func_d,
    affine as affine_d,
    linalg as linalg_d,
)
from hcl_mlir.ir import StringAttr
from hcl_mlir.passmanager import PassManager as mlir_pass_manager
from .ir.transform import create_buffer


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
                break
        func_type = FunctionType.get(new_in_types, new_out_types)
        top_func.attributes["function_type"] = TypeAttr.get(func_type)
