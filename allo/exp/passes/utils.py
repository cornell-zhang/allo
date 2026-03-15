# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo._mlir.dialects import affine as affine_d, allo as allo_d, func as func_d


def unroll_meta_for(module):
    def unroll(operations):
        for op in operations:
            for region in op.regions:
                for block in region.blocks:
                    unroll(block.operations)
            if isinstance(op, affine_d.AffineForOp):
                if (
                    "loop_type" in op.attributes
                    and op.attributes["loop_type"].value == "unroll"
                ):
                    allo_d.explicit_unroll(op)

    for func in module.body.operations:
        if isinstance(func, func_d.FuncOp):
            for block in func.body:
                unroll(block.operations)
