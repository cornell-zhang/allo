# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from allo._mlir.ir import (
    Location,
    Module,
)
from allo._mlir.dialects import (
    func as func_d,
    affine as affine_d,
    memref as memref_d,
)
from allo.customize import Schedule
from allo.ir.types import MemRefType
from allo._mlir.ir import WalkResult


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
    At the top level, alloc operations, constant declarations, and returns are allowed.
    """

    def is_constant_affine_for(loop_op):
        def is_constant_affine_map(affine_map):
            return affine_map.n_dims == 0 and len(affine_map.results) == 1

        try:
            lower_map = loop_op.lowerBoundMap.value
            lower_operands = loop_op.lowerBoundOperands
            upper_map = loop_op.upperBoundMap.value
            upper_operands = loop_op.upperBoundOperands

            lower_constant = (
                is_constant_affine_map(lower_map) and len(lower_operands) == 0
            )
            upper_constant = (
                is_constant_affine_map(upper_map) and len(upper_operands) == 0
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
            if isinstance(op, affine_d.AffineForOp) and check_perfect_loop_nest(op):
                continue

            # Disallowed top level op
            print(op, "is not a perfect affine loop or allowed top-level operation.")
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


def check_call_graph_acyclic(module: Module) -> bool:
    callgraph = defaultdict(list)
    with module.context, Location.unknown():
        for func_op in module.body.operations:
            if isinstance(func_op, func_d.FuncOp):
                for op in func_op.entry_block.operations:
                    if isinstance(op, func_d.CallOp):
                        callgraph[func_op.attributes["sym_name"].value].append(
                            op.callee.value
                        )

    def dfs(node, visited):
        if visited[node] == 1:
            return False
        if visited[node] == 2:
            return True
        visited[node] = 1
        for neighbor in callgraph[node]:
            if not dfs(neighbor, visited):
                return False
        visited[node] = 2
        return True

    visited = defaultdict(int)
    return not any(
        visited[node] != 2 and not dfs(node, visited) for node in list(callgraph.keys())
    )


def check_all_functions_inlined(mod: Module, top_fn_name: str) -> bool:
    with mod.context, Location.unknown():
        for op in mod.body.operations:
            if isinstance(op, func_d.FuncOp):
                if op.attributes["sym_name"].value != top_fn_name:
                    return False
    return True


def check_single_producer_single_consumer(module):
    spsc = True

    def checker(op):
        nonlocal spsc
        if isinstance(op.opview, (func_d.CallOp, memref_d.AllocOp)):
            for produced_val in op.results:
                if produced_val is None or not isinstance(
                    produced_val.type, MemRefType
                ):
                    return WalkResult(0)
                if len(produced_val.type.shape) == 0:
                    return WalkResult(0)
                consumers = sum(
                    1
                    for use in produced_val.uses
                    if isinstance(
                        use.owner,
                        (func_d.CallOp, memref_d.LoadOp, affine_d.AffineLoadOp),
                    )
                    # ignore if inside if-statement
                    and not isinstance(use.owner.parent.opview, affine_d.AffineForOp)
                )
                if consumers > 1:
                    print("Multiple consumers of buffer allocated by,", op)
                    spsc = False
                    return WalkResult(0)
        return WalkResult(0)

    with module.context:
        for func in module.body.operations:
            if not isinstance(func, func_d.FuncOp):
                continue
            func.walk(checker)
    return spsc


def check_preprocess_ok(schedule: Schedule) -> bool:
    module = schedule.module
    top_fn_name = schedule.top_func_name
    return (
        check_perfect_affine_kernel(module)
        and check_call_graph_acyclic(module)
        and check_all_functions_inlined(module, top_fn_name)
        and check_single_producer_single_consumer(module)
    )
