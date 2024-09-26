# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# RUN: %PYTHON %s

from allo_mlir.build_ir import IterVar
from allo_mlir.ir import *
from allo_mlir.dialects import func, arith, memref, affine
from allo_mlir.dialects import allo as allo_d
import allo_mlir

with Context() as ctx, Location.unknown() as loc:
    allo_d.register_dialect(ctx)
    module = Module.create()
    f32 = F32Type.get()
    i32 = IntegerType.get_signless(32)
    idx_type = IndexType.get()
    memref_type = MemRefType.get((1024,), f32)
    allo_mlir.enable_build_inplace()

    with InsertionPoint(module.body):

        @func.FuncOp.from_py_func(memref_type)
        def kernel(A):
            for_i = allo_mlir.make_for(0, 1024, name="i")
            allo_mlir.GlobalInsertionPoint.save(
                InsertionPoint(for_i.body.operations[0]))
            var = IterVar(for_i.induction_variable)
            var.dtype = idx_type
            with InsertionPoint(for_i.body.operations[0]):

                def make_if_block(arg):
                    a = memref.LoadOp(A, [for_i.induction_variable])
                    cst = allo_mlir.ConstantOp(idx_type, 0)
                    cmp = allo_mlir.CmpOp(var, cst, arg)
                    if_op = allo_mlir.make_if(cmp)
                    with InsertionPoint(if_op.then_block.operations[0]):
                        add = arith.AddFOp(a.result, a.result)

                make_if_block("eq")
                # make_if_block("ne")
                make_if_block("lt")
                make_if_block("le")
                make_if_block("gt")
                make_if_block("ge")

            return A

    module.dump()
    Module.parse(str(module))
    print("Built done!")
