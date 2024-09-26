# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# RUN: %PYTHON %s

from allo_mlir.ir import *
from allo_mlir.dialects import func, arith, memref, affine
from allo_mlir.dialects import allo as allo_d
import allo_mlir

with Context() as ctx, Location.unknown() as loc:
    allo_d.register_dialect(ctx)
    module = Module.create()
    f32 = F32Type.get()
    i32 = IntegerType.get_signless(32)
    memref_type = MemRefType.get((1024, 1024), f32)

    with InsertionPoint(module.body):
        op = allo_d.CreateOpHandleOp(StringAttr.get("s"))
        allo_d.CreateLoopHandleOp(op.result, StringAttr.get("i"))
        allo_d.CreateLoopHandleOp(op.result, StringAttr.get("j"))
        allo_d.CreateLoopHandleOp(op.result, StringAttr.get("k"))

        @func.FuncOp.from_py_func(memref_type, memref_type, memref_type)
        def gemm(A, B, C):
            for_i = allo_mlir.make_for(0, 1024, name="i")
            with InsertionPoint(for_i.body.operations[0]):
                for_j = allo_mlir.make_for(0, 1024, name="j")
                with InsertionPoint(for_j.body.operations[0]):
                    for_k = allo_mlir.make_for(0, 1024, name="k")
                    with InsertionPoint(for_k.body.operations[0]):
                        a = memref.LoadOp(
                            A, [for_i.induction_variable,
                                for_k.induction_variable]
                        )
                        b = memref.LoadOp(
                            B, [for_k.induction_variable,
                                for_j.induction_variable]
                        )
                        c = memref.LoadOp(
                            C, [for_i.induction_variable,
                                for_j.induction_variable]
                        )
                        prod = arith.MulFOp(a.result, b.result)
                        sum_ = arith.AddFOp(prod.result, c.result)
                        memref.StoreOp(
                            sum_.result,
                            C,
                            [for_i.induction_variable, for_j.induction_variable],
                        )

            for_i = allo_mlir.make_for(0, 1024, name="i")
            with InsertionPoint(for_i.body.operations[0]):
                for_j = allo_mlir.make_for(0, 1024, name="j")
                with InsertionPoint(for_j.body.operations[0]):
                    for_k = allo_mlir.make_for(0, 1024, name="k")
                    with InsertionPoint(for_k.body.operations[0]):
                        # make if
                        d0 = AffineDimExpr.get(0)
                        d1 = AffineDimExpr.get(1)
                        if_cond_set = IntegerSet.get(2, 0, [d0 - d1], [False])
                        attr = allo_d.IntegerSetAttr.get(if_cond_set)
                        set_operands = [
                            for_i.induction_variable,
                            for_j.induction_variable,
                        ]
                        if_op = affine.AffineIfOp(attr, set_operands)
                        with InsertionPoint(if_op.then_block):
                            a = affine.AffineLoadOp(
                                A, [for_i.induction_variable,
                                    for_k.induction_variable]
                            )
                            b = affine.AffineLoadOp(
                                B, [for_k.induction_variable,
                                    for_j.induction_variable]
                            )
                            c = affine.AffineLoadOp(
                                C, [for_i.induction_variable,
                                    for_j.induction_variable]
                            )
                            prod = arith.MulFOp(a.result, b.result)
                            sum_ = arith.AddFOp(prod.result, c.result)
                            memref.StoreOp(
                                sum_.result,
                                C,
                                [for_i.induction_variable,
                                    for_j.induction_variable],
                            )
                            affine.AffineYieldOp([])

            return C

    module.dump()
    Module.parse(str(module))
    print("Built done!")
