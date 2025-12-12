# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, no-value-for-parameter, redundant-keyword-arg, unexpected-keyword-arg
import os

from .._mlir.ir import (
    Location,
    InsertionPoint,
    Block,
    Module,
    StringAttr,
    UnitAttr,
    IntegerAttr,
    IntegerType,
    WalkResult,
    Operation,
    MemRefType,
)
from .._mlir.dialects import (
    func as func_d,
    affine as affine_d,
    memref as memref_d,
)
from .._mlir.passmanager import PassManager as mlir_pass_manager
from ..customize import Schedule
from ..ir.transform import find_func_in_module
from ..ir.utils import MockBuffer
from ..verify import verify


from .util import (
    check_perfect_affine_kernel,
    check_call_graph_acyclic,
    check_all_functions_inlined,
)

from .dfg import DFG, DFGNodeType, NodeInfo, DFGAnalysisResult, LoopInfo
from .primitives import SchedulePrimitive, UnresolvedFIFOPrimitive
from .config import AutoschedulerConfig

DEBUG_POINTS = [
    "mlir_preprocess",
    "dataflow_canonicalization",
    "outline_loops",
    "loop_opts",
    None,
]
PARALLELISM_MODELS = ["graph", "node", "combined"]


def dataflow_optimization_pass(
    schedule: Schedule,
    cfg: AutoschedulerConfig,
) -> Schedule:
    """
    Applies autoscheduler optimization passes to the schedule.
    """
    assert (
        cfg.debug_point is None or cfg.debug_point in DEBUG_POINTS
    ), f"Invalid debug point: {cfg.debug_point}"
    assert (
        cfg.kind is None or cfg.kind in PARALLELISM_MODELS
    ), f"Invalid parallelism model: {cfg.kind}"

    assert check_call_graph_acyclic(schedule.module), "Call graph is not acyclic"
    top_fn_name = schedule.top_func.name.value
    mod = _mlir_preprocess(schedule.module, top_fn_name)
    if cfg.debug_point == "mlir_preprocess":
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
    if cfg.debug_point == "dataflow_canonicalization":
        return Schedule(
            mod_dcp,
            find_func_in_module(mod_dcp, top_fn_name),
            schedule.func_args,
            schedule.ip,
            schedule.ext_libs,
            schedule.inst_list,
        )

    dfg = DFG.from_module(mod_dcp, cfg.dsp_factors, cfg.mem_w_ports, cfg.mem_r_ports)

    # name all unnamed buffers
    mod_dcp = name_buffers_pass(mod_dcp)

    # build performance model
    match cfg.kind:
        case "graph":
            result: DFGAnalysisResult = dfg.create_performance_model(
                enable_tile=False,
                verbose=cfg.verbose,
                dsp_limit=cfg.dsp_limit,
                tiling_limit=cfg.tiling_limit,
            )

        case "node":
            # solve seperately for a fixed permutation
            permutation_result: DFGAnalysisResult = dfg.create_performance_model(
                enable_tile=False,
                verbose=cfg.verbose,
                dsp_limit=cfg.dsp_limit,
                tiling_limit=cfg.tiling_limit,
            )

            # fix the permutation to the previously solved solution and solve for tiling factors
            result: DFGAnalysisResult = dfg.create_performance_model(
                permutation_result.loop_permutations,
                enable_tile=True,
                verbose=cfg.verbose,
                dsp_limit=cfg.dsp_limit,
                tiling_limit=cfg.tiling_limit,
            )

        case "combined":
            result: DFGAnalysisResult = dfg.create_performance_model(
                enable_tile=True,
                verbose=cfg.verbose,
                dsp_limit=cfg.dsp_limit,
                tiling_limit=cfg.tiling_limit,
            )
        case _:
            raise ValueError(f"Invalid parallelism model: {cfg.kind}")

    # extract FIFO primitives prior to outlining, since this requires IR manipulation
    # dependent on references from the inlined result
    fifos = extract_buffer_to_fifo(result, dfg, schedule, top_fn_name)
    mod_outlined, node_to_fn = outline_loops_pass(schedule.module, dfg)
    # construct the schedule with the outlined module
    schedule = Schedule(
        mod_outlined,
        find_func_in_module(mod_outlined, top_fn_name),
        schedule.func_args,
        schedule.ip,
        schedule.ext_libs,
        schedule.inst_list,
    )

    if cfg.debug_point == "outline_loops":
        return schedule

    # clone the original schedule for verification
    try:
        import past  # pylint: disable=unused-import
    except ImportError:
        cfg.verify = False

    if cfg.verify:
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

    # loop opt extraction post-outlining
    loop_opts = extract_reorder_and_pipeline(result, dfg, node_to_fn, schedule)
    loop_tiling = extract_tiling(result, node_to_fn, schedule)

    for primitive in loop_opts:
        primitive.applyTo(schedule)

    for primitive in loop_tiling:
        primitive.applyTo(schedule)
    post_process = _tiling_post_process(schedule)

    for primitive in post_process:
        primitive.applyTo(schedule)

    if cfg.verbose:
        print("Loop opts:")
        for primitive in loop_opts:
            print(f"\t{primitive}")
        print("Loop tiling:")
        for primitive in loop_tiling:
            print(f"\t{primitive}")

    if cfg.debug_point == "loop_opts":
        return schedule

    if cfg.verbose:
        print("FIFOs:")
        for primitive in fifos:
            print(f"\t{primitive}")

    if cfg.verify:
        verifier = verify(schedule, original_schedule)
        assert (
            verifier
        ), "Failed verification: Schedule is not equivalent to original schedule"

    _canonicalize(schedule)

    for primitive in fifos:
        if isinstance(primitive, UnresolvedFIFOPrimitive):
            primitive = primitive.resolve(top_fn_name, node_to_fn)
        primitive.applyTo(schedule)

    return schedule


def _mlir_preprocess(module, top_func_name):
    """
    Performs linalg-to-affine lowering, then aggressive inlining on an MLIR module, then removes all (dead) functions except the top-level function.
    """
    # Configure for maximum inlining - no recursion limit and always inline
    MAX_ITER = INLINE_THRESHOLD = 999999
    pipeline = (
        f"builtin.module("
        f"convert-linalg-to-affine-loops,"
        f"inline{{max-iterations={MAX_ITER} inlining-threshold={INLINE_THRESHOLD}}},"
        f"symbol-privatize{{exclude={top_func_name}}},"
        f"symbol-dce,"
        f"func.func(affine-scalrep)"
        f")"
    )
    try:
        with module.context:
            mlir_pass_manager.parse(pipeline).run(module.operation)
        return module
    except Exception as e:
        print("Error: failed to run MLIR passes, printing module...")
        print(module)
        raise e


def _canonicalize(schedule: Schedule) -> Schedule:
    pipeline = "builtin.module(canonicalize)"
    try:
        with schedule.module.context:
            mlir_pass_manager.parse(pipeline).run(schedule.module.operation)
        return schedule
    except Exception as e:
        print("Error: failed to run MLIR passes, printing module...")
        print(schedule.module)
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

    irrelevant_loops = [
        loop for loop in loop_nest if loop.induction_variable not in loop_load.indices
    ]

    irrelevant_ivs = [loop.opview.induction_variable for loop in irrelevant_loops]

    if len(irrelevant_ivs) == 0:
        return False

    with InsertionPoint.at_block_begin(loop_nest[0].body) as ip:
        first_iter_set = affine_d.IntegerSet.get(
            len(irrelevant_ivs),
            0,
            [affine_d.AffineExpr.get_dim(i) for i in range(len(irrelevant_ivs))],
            [True] * len(irrelevant_ivs),
        )

        first_iter_if = affine_d.AffineIfOp(
            first_iter_set,
            results_=[],
            cond_operands=irrelevant_ivs,
            loc=Location.unknown(),
            ip=ip,
        )

    # In the if block, load from original buffer and store to new_alloc1
    then_block = first_iter_if.then_block
    with InsertionPoint.at_block_begin(then_block):
        new_load = affine_d.AffineLoadOp(
            memref_type.element_type, alloc_op.result, loop_load.indices, loop_load.map
        )
        affine_d.AffineStoreOp(
            new_load.result, new_alloc1.result, loop_load.indices, loop_load.map
        )
        affine_d.AffineYieldOp([])

    loop_load.operation.replace_uses_of_with(alloc_op.result, new_alloc1.result)
    loop_store.operation.replace_uses_of_with(alloc_op.result, new_alloc1.result)

    upper_bounds = [loop.upperBoundMap.value.results[0] for loop in irrelevant_loops]
    last_iter_set = affine_d.IntegerSet.get(
        len(irrelevant_ivs),
        0,
        [
            affine_d.AffineExpr.get_dim(i) - upper_bound + 1
            for i, upper_bound in enumerate(upper_bounds)
        ],
        [True] * len(irrelevant_ivs),
    )

    last_iter_if = affine_d.AffineIfOp(
        last_iter_set,
        results_=[],
        cond_operands=irrelevant_ivs,
        loc=Location.unknown(),
        ip=ip,
    )

    last_iter_if.move_after(loop_store)

    final_then_block = last_iter_if.then_block
    with InsertionPoint.at_block_begin(final_then_block):
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
                buffer_name = f"_buffer_{unnamed_ct}"
                op.attributes["name"] = StringAttr.get(buffer_name)
                unnamed_ct += 1
            else:
                buffer_name = op.attributes["name"].value
            for use in op.result.uses:
                if isinstance(use.owner, (memref_d.LoadOp, affine_d.AffineLoadOp)):
                    use.owner.attributes["from"] = StringAttr.get(buffer_name)
                elif isinstance(use.owner, (memref_d.StoreOp, affine_d.AffineStoreOp)):
                    use.owner.attributes["to"] = StringAttr.get(buffer_name)

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
            pipeline = "builtin.module(transform-interpreter{entry-point=outline_affine_loops})"
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
    analysis_result: DFGAnalysisResult,
    dfg: DFG,
    node_to_fn: dict[int, str],
    schedule: Schedule,
) -> list[SchedulePrimitive]:
    schedule_primitives = []
    permutations = analysis_result.loop_permutations

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
            schedule_primitives.append(SchedulePrimitive.pipeline(loop_band[-1], 1))
        else:
            perm = node_info.permutation
            new_loop_order = [loop_band[i] for i in perm]
            schedule_primitives.append(SchedulePrimitive.reorder(new_loop_order))
            schedule_primitives.append(
                SchedulePrimitive.pipeline(new_loop_order[-1], 1)
            )

    return schedule_primitives


def extract_buffer_to_fifo(
    analysis_result: DFGAnalysisResult, dfg: DFG, schedule: Schedule, top_func: str
) -> list[SchedulePrimitive | UnresolvedFIFOPrimitive]:
    permutations = dict(analysis_result.loop_permutations)
    result = []
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
            ) and not isinstance(memref.owner, Block):
                assert (
                    "name" in memref.owner.attributes
                ), f"Buffer {memref.owner} has no name"

                # Check tiling factor to see if buffer needs to be converted to
                # array of FIFOs
                buffer_name = memref.owner.attributes["name"].value

                if analysis_result.tiling_factors is not None:
                    node_tiling_factors = analysis_result.tiling_factors.get(
                        node_idx, []
                    )
                    new_buffer, n_dims = create_fifo_array(
                        schedule,
                        memref.owner,
                        node_tiling_factors,
                        dst_node_info.loads_map[memref].op,
                        node.loop_info,
                        buffer_name,
                    )
                    if n_dims < 0:
                        # If n_dims is negative, it means the buffer is not tiled
                        _insert_guard(
                            src_node_info.stores_map[memref].op, src_node.loop_info
                        )
                        _insert_guard(
                            dst_node_info.loads_map[memref].op, node.loop_info
                        )
                        result.append(UnresolvedFIFOPrimitive(buffer_name, node_idx))
                        continue

                    new_buffer_uses = list(new_buffer.result.uses)
                    new_load = [
                        use
                        for use in new_buffer_uses
                        if isinstance(use.owner, affine_d.AffineLoadOp)
                    ]
                    new_store = [
                        use
                        for use in new_buffer_uses
                        if isinstance(use.owner, affine_d.AffineStoreOp)
                    ]
                    assert (
                        len(new_load) == 1
                    ), f"Expected one load for {buffer_name}, found {len(new_load)}"
                    assert (
                        len(new_store) == 1
                    ), f"Expected one store for {buffer_name}, found {len(new_store)}"

                    _insert_guard(new_store[0].owner, src_node.loop_info)
                    _insert_guard(new_load[0].owner, node.loop_info)

                    result.append(
                        SchedulePrimitive.buffer_to_fifo(
                            MockBuffer(top_func, buffer_name),
                            list(range(n_dims // 2)),
                            0,
                        )
                    )

                else:
                    _insert_guard(
                        src_node_info.stores_map[memref].op, src_node.loop_info
                    )
                    _insert_guard(dst_node_info.loads_map[memref].op, node.loop_info)

                    result.append(UnresolvedFIFOPrimitive(buffer_name, node_idx))
            else:
                # TODO: probably can use a partition instead of a fifo here based on the
                # loop tiling here?
                continue

    return result


def _get_affine_if_op(op):
    parent = op.parent
    while parent is not None and not isinstance(parent.opview, affine_d.AffineIfOp):
        parent = parent.parent
    return parent


def _insert_guard(op: Operation, loops: list[LoopInfo]):
    op = op.opview
    assert isinstance(op, (affine_d.AffineLoadOp, affine_d.AffineStoreOp))
    affine_if_op = _get_affine_if_op(op)
    if not affine_if_op:
        # insert guard before the fifo write/read
        irrelevant_loops = [
            loop
            for loop in loops
            if loop.op.opview.induction_variable not in op.indices
        ]
        if not irrelevant_loops:
            return
        with op.context:
            if isinstance(op, affine_d.AffineLoadOp):
                guard_condition = affine_d.IntegerSet.get(
                    len(irrelevant_loops),
                    0,
                    [
                        affine_d.AffineExpr.get_dim(i) - loop_info.lower_bound
                        for i, loop_info in enumerate(irrelevant_loops)
                    ],
                    [True] * len(irrelevant_loops),
                )
            else:
                guard_condition = affine_d.IntegerSet.get(
                    len(irrelevant_loops),
                    0,
                    [
                        affine_d.AffineExpr.get_dim(i)
                        - loop_info.upper_bound
                        + loop_info.step
                        for i, loop_info in enumerate(irrelevant_loops)
                    ],
                    [True] * len(irrelevant_loops),
                )

            with InsertionPoint(op) as ip:
                guard_if = affine_d.AffineIfOp(
                    guard_condition,
                    results_=[],
                    cond_operands=[
                        loop.op.opview.induction_variable for loop in irrelevant_loops
                    ],
                    loc=Location.unknown(),
                    ip=ip,
                )

                then_block = guard_if.then_block
                yield_op = affine_d.AffineYieldOp(
                    [],
                    ip=InsertionPoint.at_block_begin(then_block),
                    loc=Location.unknown(),
                )
                op.move_before(yield_op)
            if isinstance(op, affine_d.AffineLoadOp):
                fn_op = op.parent
                while not isinstance(fn_op.opview, func_d.FuncOp):
                    fn_op = fn_op.parent
                assert isinstance(fn_op.opview, func_d.FuncOp)

                alloc_op = memref_d.AllocOp(
                    op.memref.type,
                    [],
                    [],
                    ip=InsertionPoint.at_block_begin(fn_op.opview.body.blocks[0]),
                    loc=Location.unknown(),
                )

                # memref load and store to this alloc in the if statement
                with InsertionPoint.at_block_terminator(then_block) as ip:
                    store_op = affine_d.AffineStoreOp(
                        op.result,
                        alloc_op.result,
                        op.indices,
                        op.map,
                        ip=ip,
                        loc=Location.unknown(),
                    )
                    load_op = affine_d.AffineLoadOp(
                        op.result.type,
                        alloc_op.result,
                        op.indices,
                        op.map,
                        ip=ip,
                        loc=Location.unknown(),
                    )
                load_op.move_after(guard_if)
                for use in op.result.uses:
                    if use.owner.operation != store_op:
                        use.owner.operation.replace_uses_of_with(
                            op.result, load_op.result
                        )


# Node-parallel specific code
def extract_tiling(
    analysis_results: DFGAnalysisResult,
    node_to_fn: dict[int, str],
    schedule: Schedule,
):
    tiling_factors = analysis_results.tiling_factors

    if not tiling_factors:
        return []

    tiling_primitives = []
    for node_id, factors in tiling_factors.items():
        fn_name = node_to_fn[node_id]
        loop_band_collection = list(v for _, v in schedule.get_loops(fn_name))

        assert len(loop_band_collection) == 1, "Only perfect affine kernels supported"

        loop_band = list(loop_band_collection[0].loops.values())

        for depth, factor in sorted(factors):
            if factor > 1:
                loop = loop_band[depth]
                tiling_primitives.append(SchedulePrimitive.split(loop, factor))

    return tiling_primitives


def create_fifo_array(
    schedule: Schedule,
    old_alloc: Operation,
    tiling_factors: list[tuple[int, int]],
    dst_op: Operation,
    dst_loop_info: list[LoopInfo],
    buffer_name: str,
) -> tuple[memref_d.AllocOp, int]:
    """Create a FIFO array to replace the original buffer."""
    tiling_factors = dict(tiling_factors)
    relevant_loop_depths = [
        i
        for i, loop in enumerate(dst_loop_info)
        if loop.op.opview.induction_variable in dst_op.opview.indices
    ]
    fifo_dims = [tiling_factors[depth] for depth in relevant_loop_depths]
    original_dims = old_alloc.result.type.shape
    extra_dims = [
        original_dim // fifo_dim
        for original_dim, fifo_dim in zip(original_dims, fifo_dims)
    ]

    if all(dim == 1 for dim in fifo_dims):
        return old_alloc, -1

    with schedule.module.context, Location.unknown():
        old_type = old_alloc.result.type
        element_type = old_type.element_type

        fifo_type = MemRefType.get(fifo_dims + extra_dims, element_type)

        # Create new allocation at the same location as the old one
        ip = InsertionPoint(old_alloc)
        new_alloc = memref_d.AllocOp(fifo_type, [], [], ip=ip)
        new_alloc.attributes["name"] = StringAttr.get(buffer_name)

        # replace old allocation with new FIFO array in the schedule
        uses_to_update = list(old_alloc.result.uses)
        for use in uses_to_update:
            op = use.owner.opview

            if isinstance(op, (affine_d.AffineLoadOp, affine_d.AffineStoreOp)):
                indices = list(op.indices)

                fifo_exprs = []
                extra_exprs = []
                for i, dim_size in enumerate(fifo_dims):
                    if dim_size > 1:
                        fifo_exprs.append(
                            affine_d.AffineExpr.get_mod(
                                affine_d.AffineExpr.get_dim(i),
                                affine_d.AffineExpr.get_constant(dim_size),
                            )
                        )
                    else:
                        fifo_exprs.append(affine_d.AffineExpr.get_dim(i))

                for i, dim_size in enumerate(fifo_dims):
                    if dim_size > 1:
                        extra_exprs.append(
                            affine_d.AffineExpr.get_floor_div(
                                affine_d.AffineExpr.get_dim(i),
                                affine_d.AffineExpr.get_constant(dim_size),
                            )
                        )
                    else:
                        extra_exprs.append(affine_d.AffineExpr.get_dim(i))

                mod_map = affine_d.AffineMap.get(
                    len(fifo_exprs), 0, fifo_exprs + extra_exprs
                )

                if isinstance(op, affine_d.AffineLoadOp):
                    new_op = affine_d.AffineLoadOp(
                        op.result.type,
                        new_alloc.result,
                        indices,
                        map=mod_map,
                        ip=InsertionPoint(op),
                    )
                    op.result.replace_all_uses_with(new_op.result)

                else:  # AffineStoreOp
                    new_op = affine_d.AffineStoreOp(
                        op.value,
                        new_alloc.result,
                        indices,
                        map=mod_map,
                        ip=InsertionPoint(op),
                    )

                for attr_name in ("from", "to"):
                    if attr_name in op.attributes:
                        new_op.attributes[attr_name] = op.attributes[attr_name]

                op.erase()
        old_alloc.erase()

    return new_alloc, len(fifo_dims) + len(
        extra_dims
    )  # return the number of dimensions in the new FIFO array


def _tiling_post_process(schedule: Schedule):
    """
    Post-process the schedule after tiling to ensure all inner loops are placed after all outer loops and are fully unrolled.
    """
    primitives = []

    for func in schedule.module.body.operations:
        if not isinstance(func, func_d.FuncOp):
            continue

        func_name = func.name.value

        loop_bands = list(v for _, v in schedule.get_loops(func_name))
        if len(loop_bands) == 0:
            continue
        assert len(loop_bands) == 1, "Only perfect affine kernels supported"

        loop_band = loop_bands[0]
        loops = list(loop_band.loops.values())

        outer_loops = []
        inner_loops = []
        outer_indices = []
        inner_indices = []

        for i, loop_wrapper in enumerate(loops):
            loop = loop_wrapper.loop
            if "loop_name" in loop.attributes:
                loop_name = loop.attributes["loop_name"].value
                if loop_name.endswith(".outer"):
                    outer_loops.append(loop_wrapper)
                    outer_indices.append(i)
                elif loop_name.endswith(".inner"):
                    inner_loops.append(loop_wrapper)
                    inner_indices.append(i)

        # Skip if no split loops found
        if not outer_loops or not inner_loops:
            continue

        # Check if reordering is needed
        if outer_indices and inner_indices:
            if min(inner_indices) < max(outer_indices):
                # Create new loop order: all outer loops first, then all inner loops
                new_loop_order = []

                for i, loop in enumerate(loops):
                    if i in outer_indices:
                        new_loop_order.append(loop)

                for i, loop in enumerate(loops):
                    if i in inner_indices:
                        new_loop_order.append(loop)

                for i, loop in enumerate(loops):
                    if i not in outer_indices and i not in inner_indices:
                        new_loop_order.append(loop)

                primitives.append(SchedulePrimitive.reorder(new_loop_order))

                loops = new_loop_order

        for loop in inner_loops:
            primitives.append(SchedulePrimitive.unroll(loop, 0))

        if outer_loops:
            innermost_outer = outer_loops[-1]
            primitives.append(SchedulePrimitive.pipeline(innermost_outer, 1))

    return primitives
