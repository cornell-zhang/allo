# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

from allo._mlir.ir import (
    Location,
    InsertionPoint,
    IntegerSetAttr,
    Block,
    Module,
    StringAttr,
    UnitAttr,
    IntegerAttr,
    IntegerType,
    WalkResult,
)
from allo._mlir.dialects import (
    func as func_d,
    affine as affine_d,
    memref as memref_d,
)
from allo._mlir.passmanager import PassManager as mlir_pass_manager
from allo.customize import Schedule
from allo.ir.transform import find_func_in_module
from allo.ir.utils import MockBuffer
import allo

from .util import (
    check_perfect_affine_kernel,
    check_call_graph_acyclic,
    check_all_functions_inlined,
)

from .dfg import DFG, DFGNodeType, NodeInfo
from .primitives import SchedulePrimitive

DEBUG_POINTS = [
    "mlir_preprocess",
    "dataflow_canonicalization",
    "outline_loops",
    "loop_opts",
    None,
]
PARALLELISM_MODELS = ["graph", "node", "combined"]


def dataflow_optimization_pass(
    schedule: Schedule, debug_point=None, kind=None, verify=False, verbose=False
) -> Schedule:
    """
    Applies autoscheduler optimization passes to the schedule.
    """
    assert (
        debug_point is None or debug_point in DEBUG_POINTS
    ), f"Invalid debug point: {debug_point}"
    assert (
        kind is None or kind in PARALLELISM_MODELS
    ), f"Invalid parallelism model: {kind}"

    assert check_call_graph_acyclic(schedule.module), "Call graph is not acyclic"
    top_fn_name = schedule.top_func.name.value
    mod = _mlir_preprocess(schedule.module, top_fn_name)
    if debug_point == "mlir_preprocess":
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
    try:
        mod_dcp = _dataflow_canonicalization_pass(mod)
    except Exception as e:
        print("Error: failed to run dataflow canonicalization pass, printing module...")
        print(mod)
        raise e
    if debug_point == "dataflow_canonicalization":
        return Schedule(
            mod_dcp,
            find_func_in_module(mod_dcp, top_fn_name),
            schedule.func_args,
            schedule.ip,
            schedule.ext_libs,
            schedule.inst_list,
        )

    dfg = DFG.from_module(mod_dcp)

    # name all unnamed buffers
    mod_dcp = name_buffers_pass(mod_dcp)

    mod_outlined, node_to_fn = outline_loops_pass(mod_dcp, dfg)

    schedule = Schedule(
        mod_outlined,
        find_func_in_module(mod_outlined, top_fn_name),
        schedule.func_args,
        schedule.ip,
        schedule.ext_libs,
        schedule.inst_list,
    )

    if debug_point == "outline_loops":
        return schedule

    try:
        import past  # pylint: disable=unused-import
    except ImportError:
        verify = False

    if verify:
        # hacky clone
        mod_outlined_clone = Module.parse(
            mod_outlined.operation.get_asm(), mod_outlined.context
        )
        original_schedule = Schedule(
            mod_outlined_clone,
            find_func_in_module(mod_outlined_clone, top_fn_name),
            schedule.func_args,
            schedule.ip,
            schedule.ext_libs,
            schedule.inst_list,
        )

    loop_opts = []
    fifos = []
    # build performance model
    match kind:
        case "graph":
            permutations = dfg.create_graph_parallelism_performance_model(
                verbose=verbose
            )
            loop_opts.extend(
                extract_reorder_and_pipeline(permutations, dfg, node_to_fn, schedule)
            )
            fifos.extend(
                extract_buffer_to_fifo(permutations, dfg, node_to_fn, schedule)
            )
        case "node":
            # TODO: implement node parallelism performance model
            pass
        case "combined":
            # TODO: implement combined parallelism performance model
            pass
        case _:
            raise ValueError(f"Invalid parallelism model: {kind}")

    if verbose:
        print("Loop opts:")
        for primitive in loop_opts:
            print(f"\t{primitive}")
        print("FIFOs:")
        for primitive in fifos:
            print(f"\t{primitive}")

    for primitive in loop_opts:
        primitive.applyTo(schedule)

    if debug_point == "loop_opts":
        return schedule

    if verify:
        verifier = allo.verify(schedule, original_schedule)
        assert (
            verifier
        ), "Failed verification: Schedule is not equivalent to original schedule"

    for primitive in fifos:
        primitive.applyTo(schedule)

    return schedule


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
    ret = []
    for use in alloc_op.result.uses:
        user = use.owner
        if isinstance(user, (memref_d.LoadOp, affine_d.AffineLoadOp, func_d.CallOp)):
            for idx, operand in enumerate(user.operands):
                if operand == alloc_op.result:
                    loads.append((user, idx))
        elif isinstance(user, (memref_d.StoreOp, affine_d.AffineStoreOp)):
            stores.append(user)
        elif isinstance(user, func_d.ReturnOp):
            ret.append(user)
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
            name = f"{orig_name}_split_{i}"
            new_alloc.attributes["name"] = StringAttr.get(name)
            store_dup = store.clone(ip=InsertionPoint(store))
            store_dup.operation.replace_uses_of_with(alloc_op.result, new_alloc.result)
            store_dup.attributes["from"] = StringAttr.get(name)
            load.operation.operands[idx] = new_alloc.result
        return

    # store-load-store-load loop redution pattern
    if len(stores) == 2 and (len(loads) == 2 or len(loads) == 1 and len(ret) == 1):
        l_ops = [l[0] for l in loads]
        if store_load_store_load_pattern(alloc_op, l_ops, stores, ret):
            return

    # multiple loads and multiple stores
    if len(stores) >= 2 and len(loads) >= 1:
        raise NotImplementedError(
            f"Complex pattern detected in alloc op {alloc_op}; additional canonicalization not implemented yet."
        )


def store_load_store_load_pattern(alloc_op, loads, stores, ret):
    """
    Transforms reduction loops to satisfy the condition that the number of writes to a shared buffer equals the number of reads.
    """
    assert len(loads) + len(ret) == 2 and len(stores) == 2

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

    load_op = [l for l in loads if l != loop_load][0] if len(ret) == 0 else ret[0]

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

    inner_loop_upper_map = loop_nest[0].upperBoundMap.value

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


def name_buffers_pass(module: Module):
    unnamed_ct = 0

    def name_buffer_helper(op):
        nonlocal unnamed_ct
        if isinstance(op.opview, memref_d.AllocOp):
            if "name" not in op.attributes:
                op.attributes["name"] = StringAttr.get(f"_buffer_{unnamed_ct}")
                for use in op.result.uses:
                    if isinstance(use.owner, memref_d.LoadOp):
                        use.owner.attributes["from"] = StringAttr.get(
                            f"_buffer_{unnamed_ct}"
                        )
                    elif isinstance(use.owner, memref_d.StoreOp):
                        use.owner.attributes["to"] = StringAttr.get(
                            f"_buffer_{unnamed_ct}"
                        )
                unnamed_ct += 1
        return WalkResult(0)

    with module.context:
        module.operation.walk(name_buffer_helper)
        return module


def outline_loops_pass(
    module: Module, dfg: DFG = None
) -> tuple[Module, dict[int, str]]:
    with module.context:
        for func in module.body.operations:
            if not isinstance(func, func_d.FuncOp):
                continue
            for op in func.body.blocks[0]:
                if isinstance(op, affine_d.AffineForOp):
                    op.attributes["top_level"] = UnitAttr.get()
        if dfg:
            for node_id in dfg.nodes:
                node = dfg.nodes[node_id]
                if node.type == DFGNodeType.AFFINE:
                    node.op.attributes["node_id"] = IntegerAttr.get(
                        IntegerType.get_unsigned(32), node_id
                    )

    module_content = module.operation.get_asm()

    with open(
        os.path.join(os.path.dirname(__file__), "outline_loops.mlir"),
        "r",
        encoding="utf-8",
    ) as f:
        transform_content = f.read()

    combined_content = f"{module_content}\n{transform_content}"

    with module.context as ctx:
        ctx.allow_unregistered_dialects = True
        try:
            combined_module = Module.parse(combined_content, ctx)
            pipeline = "builtin.module(transform-interpreter{entry-point=outline_affine_loops},canonicalize)"
            mlir_pass_manager.parse(pipeline).run(combined_module.operation)
            processed_module, node_to_fn_map = post_process_module(
                Module.parse(
                    combined_module.operation.regions[0]
                    .blocks[0]
                    .operations[0]
                    .get_asm(),
                    ctx,
                )
            )
            return processed_module, node_to_fn_map

        except Exception as e:
            print("Error: failed to run MLIR passes, printing module...")
            print(combined_content)
            raise e


def post_process_module(module: Module) -> tuple[Module, dict[int, str]]:
    """
    Post-processes a module by adding loop names and operation names
    and builds a mapping from node IDs to function names.

    Args:
        module: The MLIR module to process.

    Returns:
        the processed module and a dictionary mapping node IDs to function names.
    """
    loop_counter = 0
    node_to_fn = {}

    def process_op(op, func_name):
        nonlocal loop_counter

        if isinstance(op, affine_d.AffineForOp):
            if "loop_name" not in op.attributes:
                op.attributes["loop_name"] = StringAttr.get(f"L_{loop_counter}")
                loop_counter += 1

            if "top_level" in op.attributes:
                assert "node_id" in op.attributes
                node_id = int(op.attributes["node_id"].value)

                node_to_fn[node_id] = func_name

                if "op_name" not in op.attributes:
                    op.attributes["op_name"] = StringAttr.get(f"kernel_{node_id}")

                del op.attributes["top_level"]
                del op.attributes["node_id"]

            for nested_op in op.body.operations:
                process_op(nested_op, func_name)
        elif hasattr(op, "regions"):
            for region in op.regions:
                for block in region.blocks:
                    for nested_op in block.operations:
                        process_op(nested_op, func_name)

    with module.context:
        for func in module.body.operations:
            if not isinstance(func, func_d.FuncOp):
                continue

            func_name = func.name.value

            for op in func.body.blocks[0]:
                process_op(op, func_name)

    return module, node_to_fn


def extract_reorder_and_pipeline(
    permutations, dfg: DFG, node_to_fn: dict[int, str], schedule: Schedule
) -> list[SchedulePrimitive]:
    schedule_primitives = []
    for node_id, perm_idx in permutations:
        loop_band_collection = list(
            v for _, v in schedule.get_loops(node_to_fn[node_id])
        )
        # support only perfect affine kernels
        assert (
            len(loop_band_collection) == 1
        ), "Only perfect affine kernels are supported"
        loop_band = list(loop_band_collection[0].loops.values())

        node_info: NodeInfo = dfg.get_node(node_id).node_info[perm_idx]
        if perm_idx == 0:
            schedule_primitives.append(
                SchedulePrimitive.pipeline(loop_band[-1], node_info.II)
            )
        else:
            perm = node_info.permutation
            new_loop_order = [loop_band[i] for i in perm]
            schedule_primitives.append(SchedulePrimitive.reorder(new_loop_order))
            schedule_primitives.append(
                SchedulePrimitive.pipeline(new_loop_order[-1], node_info.II)
            )

    return schedule_primitives


def extract_buffer_to_fifo(
    permutations, dfg: DFG, node_to_fn: dict[int, str], schedule: Schedule
):
    unnamed_ct = 0
    top_level_fn_name = schedule.top_func_name
    permutations = dict(permutations)
    schedule_primitives = []
    for node_idx, node in dfg.nodes.items():
        if node.type != DFGNodeType.AFFINE:
            continue
        dst_node_info = node.node_info[permutations[node_idx]]
        for edge in dfg.in_edges[node_idx]:
            src_node = dfg.nodes[edge.id]
            if src_node.type != DFGNodeType.AFFINE:
                continue
            src_node_info = src_node.node_info[permutations[edge.id]]
            memref = edge.value
            assert memref in dst_node_info.loads_map
            assert memref in src_node_info.stores_map
            if (
                dst_node_info.loads_map[memref].access_map
                == src_node_info.stores_map[memref].access_map
            ):
                if "name" not in memref.owner.attributes:
                    with schedule.module.context:
                        memref.owner.attributes["name"] = StringAttr.get(
                            f"fifo_buffer_{unnamed_ct}"
                        )
                        dst_node_info.loads_map[memref].op.attributes["from"] = (
                            StringAttr.get(f"fifo_buffer_{unnamed_ct}")
                        )
                        src_node_info.stores_map[memref].op.attributes["to"] = (
                            StringAttr.get(f"fifo_buffer_{unnamed_ct}")
                        )
                    unnamed_ct += 1
                buffer = MockBuffer(
                    top_level_fn_name, memref.owner.attributes["name"].value
                )
                fn_name = node_to_fn[node_idx]
                schedule_primitives.append(
                    SchedulePrimitive.buffer_to_fifo(buffer, fn_name)
                )

    return schedule_primitives
