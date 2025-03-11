# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from allo._mlir.ir import (
    Location,
    Module,
)
from allo._mlir.dialects import (
    func as func_d,
    affine as affine_d,
    memref as memref_d,
)


def is_terminator(op):
    ## TODO: can we do trait inspection with python bindings?
    return isinstance(op, (affine_d.AffineYieldOp, func_d.ReturnOp))


def check_perfect_affine_kernel(module: Module) -> bool:
    """
    Checks whether the module is a perfect affine kernel (https://arxiv.org/pdf/2501.09118).

    A perfect affine kernel is defined as a module with perfectly nested affine.for loops
    with constant bounds. In each perfect nest, if an affine.for contains an inner loop,
    it must be the only operation in its body (ignoring the terminator).
    The innermost loop can contain any operations.
    At the top level, alloc operations and returns are allowed.
    """

    def is_constant_affine_for(loop_op):
        def is_constant_affine_map(affine_map):
            return affine_map.n_dims == 0 and len(affine_map.results) == 1

        try:
            lower_map = loop_op.lowerBoundMap.value
            lower_operands = loop_op.lowerBoundOperands
            upper_map = loop_op.upperBoundMap.value
            upper_operands = loop_op.upperBoundOperands

            lower_constant = is_constant_affine_map(lower_map) and not len(
                lower_operands
            )
            upper_constant = is_constant_affine_map(upper_map) and not len(
                upper_operands
            )
            return lower_constant and upper_constant

        except AttributeError as e:
            print(f"Error accessing affine.for properties: {e}")
            return False

    def check_perfect_loop_nest(loop_op):
        # check for constant loop bounds
        if not is_constant_affine_for(loop_op):
            print("not constant affine for", loop_op)
            return False

        # check each affine.for has one region with a single block.
        region = loop_op.regions[0]
        block = region.blocks[0]
        body_ops = [op for op in block.operations if not is_terminator(op)]
        if not body_ops:
            return False

        inner_loops = [op for op in body_ops if isinstance(op, affine_d.AffineForOp)]
        if inner_loops:
            # If there are inner loops, ensure that's the only op in the body
            if len(body_ops) != 1:
                print(
                    "Loop", loop_op, "has extra ops besides the inner loop:", body_ops
                )
                return False
            return check_perfect_loop_nest(inner_loops[0])
        return True

    # Since all functions are inlined, we only need to check the main function
    def check_function_perfect_affine(func):
        top_level_ops = [
            op for op in func.entry_block.operations if not is_terminator(op)
        ]
        if not top_level_ops:
            print("no top level ops", top_level_ops)
            return False

        # Check each top-level op
        for op in top_level_ops:
            # Allow allocs and returns at top level
            if (
                isinstance(op, memref_d.AllocOp)
                or is_terminator(op)
                or op.OPERATION_NAME.endswith(".constant")
            ):
                continue
            # Check if it's a perfectly nested affine loop
            elif isinstance(op, affine_d.AffineForOp) and check_perfect_loop_nest(op):
                continue
            else:
                print(
                    op, "is not a perfect affine loop or allowed top-level operation."
                )
                return False
        return True

    with module.context, Location.unknown():
        # Since all functions are inlined, we only need to check one function
        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp):
                func_name = op.attributes["sym_name"].value
                if not check_function_perfect_affine(op):
                    print("Function", func_name, "is not a perfect affine kernel.")
                    return False
    return True
