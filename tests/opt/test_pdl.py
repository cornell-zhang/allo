from allo._mlir import ir, passmanager, rewrite
from allo._mlir.rewrite import PDLModule
from allo._mlir.dialects import arith, func, pdl
from allo._mlir.dialects import allo as allo_d
from allo._mlir.dialects.builtin import module
from allo._mlir.ir import *


def test_string_based():
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True

    with ctx, ir.Location.unknown():
        module = ir.Module.parse(
            """
    module {
    module @ir {
        func.func @add_func(%arg0: index, %arg1: index) -> index {
        %0 = arith.addi %arg0, %arg1 : index
        return %0 : index
        }
    }
    }
        """,
            ctx,
        )

        pdl_module = ir.Module.parse(
            """
    module {
    pdl.pattern @addi_to_mul : benefit(1) {
        %0 = type : index
        %1 = operand : %0
        %2 = operand : %0
        %3 = operation "arith.addi"(%1, %2 : !pdl.value, !pdl.value)  -> (%0 : !pdl.type)
        rewrite {
        %4 = operation "arith.muli"(%1, %2 : !pdl.value, !pdl.value)  -> (%0 : !pdl.type)
        replace %3 with %4
        }
    }
    }
    """,
            ctx,
        )
        frozen = PDLModule(pdl_module).freeze()
        rewrite.apply_patterns_and_fold_greedily(module, frozen)
        print(module)


def test_module_based():
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True

    with ctx, ir.Location.unknown():
        ir_module = ir.Module.parse(
            """
    module {
    module @ir {
        func.func @add_func(%arg0: index, %arg1: index) -> index {
        %0 = arith.addi %arg0, %arg1 : index
        return %0 : index
        }
    }
    }
        """,
            ctx,
        )

    with ctx, Location.unknown():
        m = Module.create()
        with InsertionPoint(m.body):
            # Change all arith.addi with index types to arith.muli.
            @pdl.pattern(benefit=1, sym_name="addi_to_mul")
            def pat():
                # Match arith.addi with index types.
                index_type = pdl.TypeOp(IndexType.get())
                operand0 = pdl.OperandOp(index_type)
                operand1 = pdl.OperandOp(index_type)
                op0 = pdl.OperationOp(
                    name="arith.addi", args=[operand0, operand1], types=[index_type]
                )

                # Replace the matched op with arith.muli.
                @pdl.rewrite()
                def rew():
                    newOp = pdl.OperationOp(
                        name="arith.muli", args=[operand0, operand1], types=[index_type]
                    )
                    pdl.ReplaceOp(op0, with_op=newOp)

    frozen = rewrite.PDLModule(m).freeze()
    # Could apply frozen pattern set multiple times.
    rewrite.apply_patterns_and_fold_greedily(ir_module, frozen)
    print(ir_module)


if __name__ == "__main__":
    test_string_based()
    test_module_based()
