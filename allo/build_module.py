# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

from hcl_mlir.dialects import (
    hcl as hcl_d,
    func as func_d,
    affine as affine_d,
    linalg as linalg_d,
)
from hcl_mlir.ir import StringAttr
from hcl_mlir.passmanager import PassManager as mlir_pass_manager


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


def lower_linalg_and_attach_names(ctx, module):
    def annotate_affine_for(op):
        nonlocal cnt_unnamed, cnt_ForBlockOp
        if isinstance(op, affine_d.AffineForOp):
            if ("loop_name" not in op.attributes) and ("op_name" not in op.attributes):
                if cnt_unnamed == 0:
                    buffer_name = f"linalg_buffer_{cnt_ForBlockOp}"
                    op.attributes["op_name"] = StringAttr.get(
                        ctx.buffers[buffer_name].value
                    )
                loop_name = f"L_{cnt_unnamed}"
                cnt_unnamed += 1
                op.attributes["loop_name"] = StringAttr.get(loop_name)
            annotate_affine_for(op.body.operations[0])

    with module.context:
        for op in module.body.operations:
            if (
                isinstance(op, func_d.FuncOp)
                and op.name.value == ctx.top_func.name.value
            ):
                func = op
                cnt_ForBlockOp = 0
                for op in func.entry_block.operations:
                    if isinstance(
                        op,
                        (
                            linalg_d.BatchMatmulOp,
                            linalg_d.MatmulOp,
                            linalg_d.SoftmaxOp,
                            linalg_d.FillOp,
                            linalg_d.AddOp,
                            linalg_d.SubOp,
                            linalg_d.DivOp,
                            linalg_d.ExpOp,
                            linalg_d.LogOp,
                            linalg_d.AbsOp,
                            affine_d.AffineForOp,
                        ),
                    ):
                        if not isinstance(
                            op,
                            affine_d.AffineForOp,
                        ):
                            buffer_name = f"linalg_buffer_{cnt_ForBlockOp}"
                            ctx.buffers[buffer_name] = op.attributes["op_name"]
                        cnt_ForBlockOp += 1
                break

        _mlir_lower_pipeline(module, lower_linalg=True)
        for op in module.body.operations:
            if (
                isinstance(op, func_d.FuncOp)
                and op.name.value == ctx.top_func.name.value
            ):
                func = op
                cnt_ForBlockOp = 0
                for op in func.entry_block.operations:
                    cnt_unnamed = 0
                    annotate_affine_for(op)
                    if isinstance(op, affine_d.AffineForOp):
                        cnt_ForBlockOp += 1
                break
