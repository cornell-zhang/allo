# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module
from collections import defaultdict
from typing import NamedTuple

from .._mlir.ir import (
    Location,
    Module,
    Operation,
    AffineMap,
    AffineExpr,
    Block,
)
from .._mlir.dialects import (
    func as func_d,
    affine as affine_d,
    memref as memref_d,
)
from ..ir.types import MemRefType
from .._mlir.ir import WalkResult


# PREPROCESSING
def is_terminator(op: Operation) -> bool:
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
            # TODO: we could theoretically allow constant loads/stores here as well
            if (
                isinstance(op, (memref_d.AllocOp))
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


def check_single_producer_single_consumer(module: Module) -> bool:
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


def check_preprocess_ok(schedule) -> bool:
    module = schedule.module
    top_fn_name = schedule.top_func_name
    return (
        check_perfect_affine_kernel(module)
        and check_call_graph_acyclic(module)
        and check_all_functions_inlined(module, top_fn_name)
        and check_single_producer_single_consumer(module)
    )


# DFG
LoopInfo = NamedTuple(
    "LoopInfo",
    [
        ("op", Operation),
        ("lower_bound", int),
        ("upper_bound", int),
        ("step", int),
        ("trip_count", int),
    ],
)


def get_minimal_access_pattern(op: Operation, loop_info: list[LoopInfo]) -> AffineMap:
    """
    Analyzes an affine load or store operation and returns an AffineMap representing
    minimal access function under the permutation provided in loop_info.

    For instance, if the loop nest is [i, j, k], and we load fromm buffer[j, i],
    the minimal access function would be (i, j) -> (j, i) since the load is independent
    of the last induction variable k.
    """
    # Handle load or store
    mapOperands = []
    accessMap = None
    op = op.opview
    if isinstance(op, (affine_d.AffineLoadOp, affine_d.AffineStoreOp)):
        accessMap = op.map.value
        mapOperands = op.indices
    else:
        assert False, "op is not an affine load or store operation"

    if len(mapOperands) == 0:
        return accessMap

    band = [loop.op for loop in loop_info]
    relevantLoops = []
    relevantOperands = []

    for forOp in band:
        iv = forOp.induction_variable
        if any(operand == iv for operand in mapOperands):
            relevantLoops.append(forOp)
            relevantOperands.append(iv)

    assert len(relevantOperands) == len(
        mapOperands
    ), "Complex access patterns are not supported"

    # at this point, relevantOperands holds the induction variables in the order of the loop nest
    # and mapOperands holds the operands in the order they appear for the AffineMap
    # build a permutation map of the relevant induction variables
    orderingIndices = []
    for operand in mapOperands:
        orderingIndices.append(relevantOperands.index(operand))

    permutationMap = AffineMap.get_permutation(orderingIndices, op.context)
    return compose_affine_maps(permutationMap, accessMap)


def inverse_permutation(permutation: list[int]) -> list[int]:
    inverse = [0] * len(permutation)
    for i, j in enumerate(permutation):
        inverse[j] = i
    return inverse


def compose_affine_maps(inner_map: AffineMap, outer_map: AffineMap) -> AffineMap:
    """
    Compose two affine maps. Reimplementation of MLIR's AffineMap::compose.
    """
    assert len(inner_map.results) == outer_map.n_dims, (
        f"Composition error: Inner map produces {len(inner_map.results)} results, "
        f"but outer map expects {outer_map.n_dims} dimensions"
    )

    composed_exprs = []
    for expr in outer_map.results:
        composed_expr = expr.compose(inner_map)
        composed_exprs.append(composed_expr)

    return AffineMap.get(
        inner_map.n_dims,
        inner_map.n_symbols + outer_map.n_symbols,
        composed_exprs,
        inner_map.context,
    )


def is_reduction_loop(for_op: affine_d.AffineForOp) -> bool:
    """
    Determines if an AffineForOp represents a reduction loop.
    """
    assert isinstance(for_op, affine_d.AffineForOp), "Expected an AffineForOp"

    load_ops = []
    store_ops = []
    unsupported_ops = False

    def is_locally_defined(memref, loop_op):
        if not hasattr(memref, "owner") or not memref.owner:
            return False

        current = memref.owner

        if isinstance(current, Block):
            return False

        while current:
            parent = current.parent
            if parent is not None and parent == loop_op:
                return True
            current = parent
        return False

    def walk_callback(op):
        op_view = op.opview

        if isinstance(op_view, affine_d.AffineLoadOp):
            if not is_locally_defined(op_view.memref, for_op):
                load_ops.append(op)

        elif isinstance(op_view, affine_d.AffineStoreOp):
            if not is_locally_defined(op_view.memref, for_op):
                store_ops.append(op)

        # check for unsupported operations
        elif (
            not isinstance(
                op_view,
                (affine_d.AffineForOp, affine_d.AffineYieldOp, affine_d.AffineIfOp),
            )
            and not isinstance(op_view, memref_d.AllocOp)
            and not (
                op_view.OPERATION_NAME.endswith(".constant")
                or op_view.OPERATION_NAME.startswith("arith.")
            )
        ):
            nonlocal unsupported_ops
            unsupported_ops = True
            return WalkResult(1)

        return WalkResult(0)

    for_op.operation.walk(walk_callback)

    if unsupported_ops:
        return False

    for store_op in store_ops:
        store_memref = store_op.opview.memref
        store_block = store_op.parent

        for load_op in load_ops:
            load_memref = load_op.opview.memref
            load_block = load_op.parent

            if store_memref == load_memref and store_block == load_block:
                return True

    return False


def compute_loop_II(
    for_op: affine_d.AffineForOp,
    loop_info: list[LoopInfo],
    mem_r_port=2,
    mem_w_port=1,
    latencies=None,
) -> int:
    """
    Calculate a rough estimate for the innermost loop pipeline initiation interval.
    """
    if latencies is None:
        latencies = {
            "arith.addf": 4,
            "arith.subf": 4,
            "arith.mulf": 5,
            "arith.divf": 10,
            "arith.addi": 1,
            "arith.subi": 1,
            "arith.muli": 2,
            "default": 1,
        }

    loads = []
    stores = []

    def collect_mem_ops(op):
        op_view = op.opview
        if isinstance(op_view, affine_d.AffineLoadOp):
            loads.append(op.opview)
        elif isinstance(op_view, affine_d.AffineStoreOp):
            stores.append(op.opview)
        return WalkResult(0)

    for_op.operation.walk(collect_mem_ops)

    load_res_mii = (len(loads) + mem_r_port - 1) // mem_r_port
    store_res_mii = (len(stores) + mem_w_port - 1) // mem_w_port
    ii = max(load_res_mii, store_res_mii)
    innermost_loop = loop_info[-1]
    inner_loop_iv = innermost_loop.op.opview.induction_variable
    innermost_loop_dim = AffineExpr.get_dim(len(loop_info) - 1, for_op.context)

    for load in loads:
        load_memref = load.memref
        load_parent = load.parent
        for store in stores:
            store_memref = store.memref
            store_parent = store.parent
            if (
                store_parent == load_parent
                and load_memref == store_memref
                and inner_loop_iv in store.indices
                and inner_loop_iv in load.indices
            ):
                store_map = get_minimal_access_pattern(store, loop_info)
                load_map = get_minimal_access_pattern(load, loop_info)
                load_dependencies = [
                    res
                    for res in load_map.results
                    if str(innermost_loop_dim) in str(res)
                ]
                store_dependencies = [
                    res
                    for res in store_map.results
                    if str(innermost_loop_dim) in str(res)
                ]
                # if there are dependencies to diff loop iter, there is a carried loop dependency
                if any(dep not in store_dependencies for dep in load_dependencies):
                    ii = max(ii, estimate_critical_path(load, store, latencies))
            ii = max(1, ii)

    return ii


def estimate_critical_path(load_op, store_op, latencies):
    """
    Estimate the critical path latency from load to store.
    """
    # minimum latency for a load and store operation
    min_latency = 2

    current_value = load_op.result

    stored_value = store_op.value

    if not current_value or not stored_value:
        return min_latency

    path_latency = latencies.get(load_op.operation.name, 1)

    visited = set([load_op.operation])

    max_steps = 10
    step_count = 0

    while current_value != stored_value and step_count < max_steps:
        step_count += 1

        users = list(current_value.uses)
        if not users:
            break

        found_next = False
        for use in users:
            next_op = use.owner

            if next_op in visited:
                continue

            visited.add(next_op)

            if next_op == store_op.operation:
                path_latency += latencies.get(next_op.name, latencies["default"])
                found_next = True
                break

            if not next_op.results:
                continue

            op_latency = latencies.get(next_op.name, latencies["default"])
            path_latency += op_latency
            current_value = next_op.results[0]
            found_next = True
            break

        if not found_next:
            break

    return max(path_latency, min_latency)
