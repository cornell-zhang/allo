# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from allo._mlir.ir import (
    Location,
    InsertionPoint,
    IntegerSetAttr,
    Block,
)
from allo._mlir.dialects import (
    func as func_d,
    affine as affine_d,
    memref as memref_d,
)
from allo._mlir.ir import StringAttr
from allo._mlir.passmanager import PassManager as mlir_pass_manager
from allo.customize import Schedule
from allo.ir.transform import find_func_in_module
from .util import (
    check_perfect_affine_kernel,
    check_call_graph_acyclic,
    check_all_functions_inlined,
)


def dataflow_optimization_pass(schedule: Schedule, debugPoint=None) -> Schedule:
    assert check_call_graph_acyclic(schedule.module), "Call graph is not acyclic"
    top_fn_name = schedule.top_func.name.value
    mod = _mlir_preprocess(schedule.module, top_fn_name)
    if debugPoint == "mlir_preprocess":
        return Schedule(
            mod,
            find_func_in_module(mod, top_fn_name),
            schedule.func_args,
            schedule.ip,
            schedule.ext_libs,
            schedule.inst_list,
        )
    assert check_all_functions_inlined(
        mod, top_fn_name
    ), "All functions are not inlined"
    assert check_perfect_affine_kernel(
        mod
    ), "Input kernel is not a perfect affine kernel"

    # Dataflow canonicalization pass
    mod = _dataflow_canonicalization_pass(mod)
    if debugPoint == "dataflow_canonicalization":
        return Schedule(
            mod,
            find_func_in_module(mod, top_fn_name),
            schedule.func_args,
            schedule.ip,
            schedule.ext_libs,
            schedule.inst_list,
        )

    # other passes (not implemented)
    # dfg = build_dataflow_graph(mod)
    # build_performance_model(mod, dfg)
    return Schedule(
        mod,
        find_func_in_module(mod, top_fn_name),
        schedule.func_args,
        schedule.ip,
        schedule.ext_libs,
        schedule.inst_list,
    )


def _mlir_preprocess(module, top_func_name):
    """
    Performs linalg-to-affine lowering, then aggressive inlining on an MLIR module, then removes all (dead) functions except the top-level function.
    """
    # Configure for maximum inlining - no recursion limit and always inline
    MAX_ITER = INLINE_THRESHOLD = 999999
    pipeline = f"builtin.module(convert-linalg-to-affine-loops,inline{{max-iterations={MAX_ITER} inlining-threshold={INLINE_THRESHOLD}}},symbol-privatize{{exclude={top_func_name}}},symbol-dce)"
    try:
        with module.context:
            mlir_pass_manager.parse(pipeline).run(module.operation)
        return module
    except Exception as e:
        print("Error: failed to run MLIR passes, printing module...")
        print(module)
        raise e


def _dataflow_canonicalization_pass(module):
    """
    Implements the dataflow canonicalization pass as described in the Stream-HLS paper (https://arxiv.org/pdf/2501.09118)

    This pass ensures that the program is compatible with dataflow architectures by transforming shared buffers to adhere to single-producer-single-consumer patterns. This pass does not handle complex patterns involving multiple producers writing to the same buffer, except in the case of reduction loops.
    """
    with module.context, Location.unknown():
        for op in module.body.operations:
            if not isinstance(op, func_d.FuncOp):
                continue
            canonicalize_fn(op)
    return module


def canonicalize_fn(op: func_d.FuncOp):
    ops = list(op.entry_block.operations)
    for op_in_block in ops:
        if isinstance(op_in_block, memref_d.AllocOp):
            canonicalize_alloc(op_in_block)


def canonicalize_alloc(alloc_op):
    loads = []  # (op, idx)
    stores = []  # ops
    returns = []  # ops
    for use in alloc_op.result.uses:
        user = use.owner
        if isinstance(user, (memref_d.LoadOp, affine_d.AffineLoadOp, func_d.CallOp)):
            for idx, operand in enumerate(user.operands):
                if operand == alloc_op.result:
                    loads.append((user, idx))
        elif isinstance(user, (memref_d.StoreOp, affine_d.AffineStoreOp)):
            stores.append(user)
        elif isinstance(user, func_d.ReturnOp):
            returns.append(user)

    memref_type = alloc_op.result.type
    shape = memref_type.shape
    orig_name = (
        alloc_op.attributes["name"].value if "name" in alloc_op.attributes else "buffer"
    )
    if len(shape) == 0:
        # should constants be propogated?
        return

    # single store with multiple loads.
    if len(stores) == 1 and len(loads) > 1:
        store = stores[0]
        for i, (load, idx) in enumerate(loads[1:]):
            new_alloc = alloc_op.operation.clone(ip=InsertionPoint(alloc_op))
            new_alloc.attributes["name"] = StringAttr.get(f"{orig_name}_split_{i}")
            store_dup = store.clone(ip=InsertionPoint(store))
            store_dup.operation.replace_uses_of_with(alloc_op.result, new_alloc.result)
            load.operation.operands[idx] = new_alloc.result
        return

    # store-load-store-load loop redunction pattern
    if len(stores) == 2 and len(loads) + len(returns) == 2:
        l_ops = [l[0] for l in loads]
        l_ops.extend(returns)
        if store_load_store_load_pattern(alloc_op, l_ops, stores):
            return

    # multiple loads and multiple stores
    if len(stores) >= 2 and len(loads) >= 2:
        print(stores, loads)
        raise NotImplementedError(
            f"Complex pattern detected in alloc op {alloc_op}; additional canonicalization not implemented yet."
        )


def store_load_store_load_pattern(alloc_op, loads, stores):
    """
    Transforms reduction loops to satisfy the condition that the number of writes to a shared buffer equals the number of reads.
    """
    assert len(loads) == 2 and len(stores) == 2

    loop_load, loop_store = None, None

    # find loop_load and loop_store
    for load in loads:
        for store in stores:
            if load.parent == store.parent:
                loop_load = load
                loop_store = store
                break
        if loop_load:
            break

    if not loop_load or not loop_store:
        return False

    store_op = [s for s in stores if s != loop_store][0]
    load_op = [l for l in loads if l != loop_load][0]

    # check for unsupported store_op in an if block
    parent = store_op.parent
    while parent:
        if isinstance(parent, affine_d.AffineIfOp):
            return False
        parent = parent.parent

    loop_nest = []
    current_op = loop_load.parent
    while current_op:
        if isinstance(current_op.opview, affine_d.AffineForOp):
            loop_nest.append(current_op.opview)
        current_op = current_op.parent
    if not loop_nest:
        return False

    ip = InsertionPoint(alloc_op)
    memref_type = alloc_op.result.type
    new_alloc1 = memref_d.AllocOp(memref_type, [], [], ip=ip)
    new_alloc2 = memref_d.AllocOp(memref_type, [], [], ip=ip)

    loop_ivs = [loop.induction_variable for loop in loop_nest]  # innermost IV first

    with InsertionPoint.at_block_begin(loop_nest[0].body):
        first_iter_set = affine_d.IntegerSet.get(
            1, 0, [affine_d.AffineExpr.get_dim(0)], [True]
        )

        first_iter_if = affine_d.AffineIfOp(
            results_=[], _gen_arg_0=[loop_ivs[0]], loc=Location.unknown()
        )

        first_iter_if.attributes["condition"] = IntegerSetAttr.get(first_iter_set)

    # In the if block, load from original buffer and store to new_alloc1
    then_block = Block.create_at_start(parent=first_iter_if.thenRegion)
    with InsertionPoint(then_block):
        new_load = affine_d.AffineLoadOp(
            memref_type.element_type, alloc_op.result, loop_load.indices, loop_load.map
        )
        affine_d.AffineStoreOp(
            new_load.result, new_alloc1.result, loop_load.indices, loop_load.map
        )
        affine_d.AffineYieldOp([])

    loop_load.operation.replace_uses_of_with(alloc_op.result, new_alloc1.result)
    loop_store.operation.replace_uses_of_with(alloc_op.result, new_alloc1.result)

    inner_loop_upper_map = loop_nest[-1].upperBoundMap.value

    # check upper bound is a constant
    if not (
        inner_loop_upper_map.n_dims == 0 and len(inner_loop_upper_map.results) == 1
    ):
        return False

    loop_upper_bound = inner_loop_upper_map.results[0]

    last_iter_set = affine_d.IntegerSet.get(
        1, 0, [affine_d.AffineExpr.get_dim(0) - loop_upper_bound + 1], [True]
    )

    last_iter_if = affine_d.AffineIfOp(
        results_=[], _gen_arg_0=[loop_ivs[0]], loc=Location.unknown(), ip=ip
    )

    last_iter_if.move_after(loop_store)

    last_iter_if.attributes["condition"] = IntegerSetAttr.get(last_iter_set)

    final_then_block = Block.create_at_start(parent=last_iter_if.thenRegion)
    with InsertionPoint(final_then_block):
        final_loop_load = affine_d.AffineLoadOp(
            memref_type.element_type,
            new_alloc1.result,
            loop_load.indices,
            loop_load.map,
        )
        affine_d.AffineStoreOp(
            final_loop_load.result, new_alloc2.result, loop_load.indices, loop_load.map
        )
        affine_d.AffineYieldOp([])

    load_op.operation.replace_uses_of_with(alloc_op.result, new_alloc2.result)

    return True


# def build_dataflow_graph(module: Module):
#     pass


# def build_performance_model(module: Module, dfg):
#     pass
