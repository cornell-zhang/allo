# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
import enum
import sys
import itertools
import gurobipy as gp
from gurobipy import GRB
from allo._mlir.dialects import (
    func as func_d,
    affine as affine_d,
    memref as memref_d,
)
from allo._mlir.ir import WalkResult, Operation, AffineMap, Block
from allo.ir.types import MemRefType
from .util import (
    LoopInfo,
    is_reduction_loop,
    is_terminator,
    get_minimal_access_pattern,
    compute_loop_II,
)


class DFGNodeType(enum.Enum):
    AFFINE = 0
    RET = 1
    ALLOC = 2
    CONST = 3


class Edge:
    """Represents a data dependency between nodes in the dataflow graph."""

    def __init__(self, dst_id: int, value=None, src_op=None, dst_op=None):
        self.id = dst_id  # Destination node ID
        self.value = value  # The value/memref being communicated
        self.src_op = src_op  # Source operation (e.g., store)
        self.dst_op = dst_op  # Destination operation (e.g., load)

    def __repr__(self):
        return f"Edge(id={self.id}, value={self.value.get_name()})"


class EdgeInfo:
    def __init__(self, access_map: AffineMap, op=None):
        self.access_map = access_map
        self.first_element_time = 0
        self.last_element_time = 0
        self.op = op

    def __repr__(self):
        return f"EdgeInfo(access_map={self.access_map}, op={self.op})"


class NodeInfo:
    def __init__(self, permutation):
        # Map permutation to edge info associated with that permutation
        self.stores_map = {}
        self.loads_map = {}
        self.II = 0
        self.trip_count = 0
        self.permutation = permutation

    def __repr__(self):
        return f"NodeInfo(permutation={self.permutation})"


class Node:
    """Represents a computation node (typically a loop nest) in the dataflow graph."""

    def __init__(self, node_id: int, op=None, op_type=None):
        self.id = node_id  # Unique identifier
        self.op = op  # The operation (typically a loop)
        self.loads = []  # Load operations within this node
        self.stores = []  # Store operations within this node
        self.allocations = []  # Allocation operations
        self.loop_info = []  # Loop information (in the original permutation)
        self.node_info = []  # Node information precomputed per permutation
        self.type = op_type  # Node type (AFFINE, RET, ALLOC, CONST)
        self.DSP_factor = 0  # DSP factor for this node
        self.is_reduction = None  # Whether this node is a reduction loop

    def __repr__(self):
        return f"Node(id={self.id}, op={str(self.op.operation.name) if self.op else 'None'})"

    def get_load_op_count(self, memref) -> int:
        """Count load operations for a specific memref."""
        return sum(1 for load_op in self.loads if self._get_memref(load_op) == memref)

    def get_store_op_count(self, memref) -> int:
        """Count store operations for a specific memref."""
        return sum(
            1 for store_op in self.stores if self._get_memref(store_op) == memref
        )

    def _get_memref(self, op):
        """Get the memref from an operation."""
        if hasattr(op, "memref"):
            return op.memref
        # Fall back to checking operands
        for operand in op.operands:
            if hasattr(operand, "type") and "memref" in str(operand.type):
                return operand
        return None


class DFG:
    """Dataflow Graph representing an MLIR module."""

    def __init__(
        self, block=None, dsp_factors=None, mem_r_ports=None, mem_w_ports=None
    ):
        self.block: Block = block
        self.nodes: dict[int, Node] = {}  # Map from node ID to Node
        self.in_edges = defaultdict(list)  # Map from node ID to incoming edges
        self.out_edges = defaultdict(list)  # Map from node ID to outgoing edges
        self.edges = []  # list of all edges
        self.memref_edge_count = defaultdict(int)  # Count edges per memref
        self.next_node_id = 0
        self.dsp_factors = (
            {
                "arith.mulf": 3,
                "arith.addf": 0,
                "arith.subf": 0,
                "arith.divf": 14,
                "arith.remf": 14,
                "arith.muli": 1,
                "arith.addi": 0,
                "arith.subi": 0,
                "arith.divi": 8,
                "arith.divu": 8,
                "arith.remi": 8,
                "arith.remu": 8,
            }
            if not dsp_factors
            else dsp_factors
        )
        self.mem_r_ports = mem_r_ports
        self.mem_w_ports = mem_w_ports

    def add_node(self, op, op_type) -> int:
        """Add a node to the graph and return its ID."""
        node = Node(self.next_node_id, op, op_type)
        self.nodes[node.id] = node
        self.next_node_id += 1
        return node.id

    def get_node(self, node_id: int) -> Node:
        """Get a node by its ID."""
        if node_id not in self.nodes:
            raise KeyError(f"Node ID {node_id} not found in graph")
        return self.nodes[node_id]

    def has_edge(self, src_id: int, dst_id: int, value=None) -> bool:
        """Check if there's an edge from src_id to dst_id."""
        if src_id not in self.out_edges or dst_id not in self.in_edges:
            return False

        has_out_edge = any(
            edge.id == dst_id and (not value or edge.value == value)
            for edge in self.out_edges[src_id]
        )
        has_in_edge = any(
            edge.id == src_id and (not value or edge.value == value)
            for edge in self.in_edges[dst_id]
        )

        return has_out_edge and has_in_edge

    def add_edge(self, src_id: int, dst_id: int, value=None, src_op=None, dst_op=None):
        """Add an edge from src_id to dst_id."""
        if not self.has_edge(src_id, dst_id, value):
            self.out_edges[src_id].append(Edge(dst_id, value, src_op, dst_op))
            self.in_edges[dst_id].append(Edge(src_id, value, src_op, dst_op))
            self.edges.append((src_id, dst_id, value))

            if value and hasattr(value, "type") and "memref" in str(value.type):
                self.memref_edge_count[value] += 1

    def _compute_loop_info(self, loop_op) -> list[LoopInfo]:
        """Calculate loop info, outermost loop first."""
        lower_bound = int(str(loop_op.lowerBoundMap.value.results[0]))
        upper_bound = int(str(loop_op.upperBoundMap.value.results[0]))

        step = 1
        if loop_op.step is not None:
            step = int(loop_op.step)

        if step == 0:
            return 0

        if upper_bound < lower_bound:
            return 0

        inner_for_ops = [op for op in loop_op.body.operations if self._is_loop_op(op)]
        current_trip_count = (upper_bound - lower_bound + step - 1) // step

        if len(inner_for_ops) == 0:
            return [
                LoopInfo(loop_op, lower_bound, upper_bound, step, current_trip_count)
            ]

        if len(inner_for_ops) == 1:
            inner_loop_infos = self._compute_loop_info(inner_for_ops[0])
            return [
                LoopInfo(loop_op, lower_bound, upper_bound, step, current_trip_count)
            ] + inner_loop_infos

        raise NotImplementedError(
            "Nested loops with more than 1 inner loop are not supported"
        )

    def _compute_first_and_last_element_time(
        self, op: Operation, loop_band: list[LoopInfo]
    ) -> tuple[int, int]:
        """Compute the first and last element time for an operation."""
        op = op.opview
        if isinstance(op, (affine_d.AffineLoadOp, affine_d.AffineStoreOp)):
            mapOperands = op.indices
        else:
            assert False, "op is not an affine load or store operation"

        # Compute the first and last element time
        first_element_time = 0
        last_element_time = 0
        curr_factor = 1
        innermost_first = list(reversed(loop_band))
        for idx, loop in enumerate(innermost_first):
            prev_trip_count = 1 if idx == 0 else innermost_first[idx - 1].trip_count
            curr_factor *= prev_trip_count
            iv = loop.op.opview.induction_variable
            if iv in mapOperands:
                first_element_time += 0
                last_element_time += (loop.trip_count - 1) * curr_factor

            # not in loop bound, compute conservative estimate. for loads it is 0, for stores it is trip_count - 1
            elif isinstance(op, affine_d.AffineLoadOp):
                first_element_time += 0
                last_element_time += 0
            else:
                first_element_time += (loop.trip_count - 1) * curr_factor
                last_element_time += (loop.trip_count - 1) * curr_factor
        return first_element_time, last_element_time

    def _populate_node_info(self, node_id: int) -> bool:
        """Populate node information for a specific node."""
        node: Node = self.get_node(node_id)
        if node.type != DFGNodeType.AFFINE:
            return False

        top_level_for = node.loop_info[0].op

        def update_dsp_count(op):
            if (
                hasattr(op, "opview")
                and hasattr(op.opview, "operation")
                and hasattr(op.opview.operation, "name")
            ):
                op_name = op.opview.operation.name
                if op_name.startswith("arith."):
                    node.DSP_factor += self.dsp_factors.get(op_name, 0)
            return WalkResult(0)

        top_level_for.walk(update_dsp_count)
        node.is_reduction = is_reduction_loop(top_level_for)

        num_loops = len(node.loop_info)

        all_permutations = itertools.permutations(range(num_loops))

        for perm in all_permutations:
            node_info = NodeInfo(perm)
            node.node_info.append(node_info)
            permuted_loops = [node.loop_info[i] for i in perm]
            for load in node.loads:
                memref = self._get_memref(load)
                access_map = get_minimal_access_pattern(load, permuted_loops)
                node_info.loads_map[memref] = EdgeInfo(access_map, load)
                first_element_time, last_element_time = (
                    self._compute_first_and_last_element_time(load, permuted_loops)
                )
                node_info.loads_map[memref].first_element_time = first_element_time
                node_info.loads_map[memref].last_element_time = last_element_time

            for store in node.stores:
                memref = self._get_memref(store)
                access_map = get_minimal_access_pattern(store, permuted_loops)
                node_info.stores_map[memref] = EdgeInfo(access_map, store)
                first_element_time, last_element_time = (
                    self._compute_first_and_last_element_time(store, permuted_loops)
                )
                node_info.stores_map[memref].first_element_time = first_element_time
                node_info.stores_map[memref].last_element_time = last_element_time

            # Compute II
            node_info.II = compute_loop_II(top_level_for, permuted_loops)
        return True

    def init(self) -> bool:
        """Initialize the dataflow graph from the MLIR module."""
        if not self.block:
            return False

        # Map from loops to node IDs
        loop_to_node_id = {}

        # Map from memrefs to set of nodes that access them
        memref_stores = defaultdict(set)
        memref_loads = defaultdict(set)
        memref_allocs = defaultdict(set)

        # Create nodes
        for op in self.block.operations:
            if self._is_loop_op(op):
                node_id = self.add_node(op, DFGNodeType.AFFINE)
                loop_to_node_id[op] = node_id

                self._collect_memory_ops(op, self.nodes[node_id])
                self.nodes[node_id].loop_info = self._compute_loop_info(op)

                for load_op in self.nodes[node_id].loads:
                    memref = self._get_memref(load_op)
                    if memref:
                        memref_loads[memref].add((node_id, load_op))

                for store_op in self.nodes[node_id].stores:
                    memref = self._get_memref(store_op)
                    if memref:
                        memref_stores[memref].add((node_id, store_op))

                for alloc_op in self.nodes[node_id].allocations:
                    memref = self._get_memref(alloc_op)
                    if memref:
                        memref_allocs[memref].add((node_id, alloc_op))

            elif is_terminator(op):
                node_id = self.add_node(op, DFGNodeType.RET)
                memrefs = self._get_memrefs_for_return(op)
                for memref in memrefs:
                    memref_loads[memref].add((node_id, op))

            elif self._is_alloc_op(op) or self._is_constant_op(op):
                continue

            else:
                raise NotImplementedError(
                    f"Unsupported operation found in kernel: {op}"
                )

        # add edges
        for memref, node_ids in memref_loads.items():
            loads_node_list = sorted(list(node_ids), key=lambda x: x[0])
            stores_node_list = sorted(list(memref_stores[memref]), key=lambda x: x[0])
            for load_node_id, load_op in loads_node_list:
                for store_node_id, store_op in stores_node_list:
                    if load_node_id != store_node_id:
                        self.add_edge(
                            store_node_id,
                            load_node_id,
                            value=memref,
                            src_op=store_op,
                            dst_op=load_op,
                        )

        # Third pass: populate node information for each loop permutation
        for node in self.nodes.values():
            self._populate_node_info(node.id)

        return True

    def _is_loop_op(self, op) -> bool:
        """Check if an operation is a loop."""
        return isinstance(op.opview, affine_d.AffineForOp)

    def _is_load_op(self, op) -> bool:
        """Check if an operation is a load."""
        return isinstance(op.opview, (affine_d.AffineLoadOp, memref_d.LoadOp))

    def _is_store_op(self, op) -> bool:
        """Check if an operation is a store."""
        return isinstance(op.opview, (affine_d.AffineStoreOp, memref_d.StoreOp))

    def _is_alloc_op(self, op) -> bool:
        """Check if an operation is an allocation."""
        return isinstance(op.opview, memref_d.AllocOp)

    def _is_constant_op(self, op) -> bool:
        """Check if an operation is a constant."""
        return op.OPERATION_NAME.endswith(".constant")

    def _get_memref(self, op):
        """Get the memrefs from operation."""
        assert not isinstance(op.opview, func_d.ReturnOp)
        if hasattr(op, "memref"):
            return op.memref

        for operand in op.operands:
            if hasattr(operand, "type") and isinstance(operand.type, MemRefType):
                return operand
        return None

    def _get_memrefs_for_return(self, op) -> list:
        return [
            operand for operand in op.operands if isinstance(operand.type, MemRefType)
        ]

    def _collect_memory_ops(self, op: Operation, node: Node):
        def collector_callback(current_op):
            if self._is_load_op(current_op):
                node.loads.append(current_op)
            elif self._is_store_op(current_op):
                node.stores.append(current_op)
            elif self._is_alloc_op(current_op):
                node.allocations.append(current_op)

            return WalkResult(0)

        assert self._is_loop_op(op)
        op.walk(collector_callback)

    def _find_store_op(self, node, memref):
        """Find a store operation for a specific memref in a node."""
        for store_op in node.stores:
            if self._get_memref(store_op) == memref:
                return store_op
        return None

    def _find_load_op(self, node, memref):
        """Find a load operation for a specific memref in a node."""
        for load_op in node.loads:
            if self._get_memref(load_op) == memref:
                return load_op
        return None

    def topological_sort(self) -> list[int]:
        """Sort nodes in topological order."""
        visited = set()
        temp_mark = set()
        order = []

        def visit(node_id):
            if node_id in temp_mark:
                # Cycle detected
                return False
            if node_id in visited:
                return True

            temp_mark.add(node_id)

            for edge in self.out_edges.get(node_id, []):
                if not visit(edge.id):
                    return False

            temp_mark.remove(node_id)
            visited.add(node_id)
            order.append(node_id)
            return True

        for node_id in self.nodes:
            if node_id not in visited:
                if not visit(node_id):
                    # Cycle detected
                    return []

        return list(reversed(order))

    def print_as_dot(self, output_file_path):
        # if output_file_path is None, write to stdout
        with (
            open(output_file_path, "w", encoding="utf-8")
            if output_file_path
            else sys.stdout
        ) as f:
            f.write("digraph DataFlowGraph {\n")

            f.write('  node [shape=box, style=filled, fontname="Arial"];\n')
            f.write('  edge [fontname="Arial"];\n\n')

            f.write("  // Nodes\n")
            for node_id, node in self.nodes.items():
                if node.type == DFGNodeType.AFFINE:
                    color = "lightblue"
                    label = f"Node {node_id} (AFFINE)"

                    if node.op:
                        loads = [load.operands[0].get_name() for load in node.loads]
                        stores = [store.operands[1].get_name() for store in node.stores]
                        op_str = f"Loads: {loads}\nStores: {stores}"
                        op_str = op_str.replace('"', '\\"').replace("\n", "\\n")
                        label = f"{label}\\n{op_str}"
                elif node.type == DFGNodeType.RET:
                    color = "pink"
                    label = f"Node {node_id} (RETURN)"
                elif node.type == DFGNodeType.ALLOC:
                    color = "lightgray"
                    label = f"(ALLOC) {node.allocations[0].memref.get_name()}"
                elif node.type == DFGNodeType.CONST:
                    color = "lightyellow"
                    label = f"(CONST) {node.op}"
                else:
                    color = "white"
                    label = f"Node {node_id} (UNKNOWN)"

                f.write(f'  node{node_id} [label="{label}", fillcolor="{color}"];\n')

            f.write("\n  // Edges\n")
            for src_id, dst_list in self.out_edges.items():
                for edge in dst_list:
                    dst_id = edge.id

                    edge_attrs = []
                    if edge.value:
                        edge_label = edge.value.get_name().replace('"', '\\"')
                        edge_attrs.append(f'label="{edge_label}"')

                    f.write(f"  node{src_id} -> node{dst_id}")
                    if edge_attrs:
                        f.write(f' [{", ".join(edge_attrs)}]')
                    f.write(";\n")

            f.write("}\n")

    def create_graph_parallelism_performance_model(
        self, debug_output=None, verbose=False
    ):
        model = gp.Model("graph_parallelism_performance_model")

        model.setParam("OutputFlag", 1 if verbose else 0)

        # Get topological order and verify no cycles
        topo_order = self.topological_sort()
        if not topo_order:
            print("Error: Cycle detected in graph")
            return False

        # Find sink nodes
        sink_node_ids = self._find_sink_nodes()

        # Create variables for the optimization model
        b_vars = self._create_permutation_variables(model)
        st_vars, fw_vars, lw_vars = self._create_timing_variables(model)

        # Add constraints
        self._add_permutation_constraints(model, b_vars)
        self._add_start_time_constraints(
            model, b_vars, st_vars, fw_vars, lw_vars, topo_order
        )
        self._add_first_write_time_constraints(
            model, b_vars, st_vars, fw_vars, topo_order
        )
        self._add_last_write_time_constraints(
            model, b_vars, st_vars, lw_vars, topo_order
        )

        max_lw = model.addVar(name="max_last_write_time")
        for sink_node_id in sink_node_ids:
            model.addConstr(max_lw >= lw_vars[sink_node_id])

        model.setObjective(max_lw, GRB.MINIMIZE)
        model.optimize()
        if debug_output:
            model.write(f"{debug_output}.lp")
        # Return optimal permutation assignments
        # format is list of (node_id, perm_idx)
        return [k for k, b_var in b_vars.items() if b_var.x > 0.5]

    def _find_sink_nodes(self):
        """Find the sink node (return node) in the graph."""
        sink_nodes = [
            node_id
            for node_id in self.nodes
            if len(self.out_edges.get(node_id, [])) == 0
        ]
        assert len(sink_nodes) > 0, "Expected at least one sink node"
        return sink_nodes

    def _create_permutation_variables(self, model):
        """Create binary variables for node permutations."""
        b_vars = {}
        for node_id, node in self.nodes.items():
            if node.type == DFGNodeType.AFFINE:
                for perm_idx, _ in enumerate(node.node_info):
                    b_vars[(node_id, perm_idx)] = model.addVar(
                        vtype=GRB.BINARY, name=f"b{node_id}_{perm_idx}"
                    )
            # if node.is_reduction:
            #     model.addConstr(b_vars[(node_id, 0)] == 1)
        return b_vars

    def _create_timing_variables(self, model):
        """Create variables for start time, first write time, and last write time."""
        st_vars = {}  # Start time variables
        fw_vars = {}  # First write time variables
        lw_vars = {}  # Last write time variables

        for node_id in self.nodes:
            st_vars[node_id] = model.addVar(
                vtype=GRB.INTEGER, lb=0, name=f"st{node_id}"
            )
            fw_vars[node_id] = model.addVar(
                vtype=GRB.INTEGER, lb=0, name=f"fw{node_id}"
            )
            lw_vars[node_id] = model.addVar(
                vtype=GRB.INTEGER, lb=0, name=f"lw{node_id}"
            )

        return st_vars, fw_vars, lw_vars

    def _add_permutation_constraints(self, model, b_vars):
        """Add constraints to ensure exactly one permutation is chosen per node."""
        for node_id, node in self.nodes.items():
            if node.type == DFGNodeType.AFFINE and node.node_info:
                perm_vars = [
                    b_vars[(node_id, perm_idx)]
                    for perm_idx in range(len(node.node_info))
                ]
                model.addConstr(
                    gp.quicksum(perm_vars) == 1,
                    name=f"permutation_constraint_{node_id}",
                )

    def _add_start_time_constraints(
        self, model, b_vars, st_vars, fw_vars, lw_vars, topo_order
    ):
        r"""st(n) = max_{n' \in ins(n)} [\sum_{b \in B_n} \sum_{b' \in B_n'} Arrives(n, n') * b * b']"""
        for node_id in topo_order:
            in_edges = self.in_edges.get(node_id, [])
            # Handle root nodes (no incoming edges)
            if not in_edges:
                model.addConstr(st_vars[node_id] == 0, name=f"st_root_{node_id}")
                continue

            arrives_terms = self._compute_arrival_terms(
                model, node_id, in_edges, b_vars, fw_vars, lw_vars
            )

            if arrives_terms:
                model.addConstr(
                    st_vars[node_id] == gp.max_(arrives_terms),
                    name=f"st_constr_{node_id}",
                )

    def _compute_arrival_terms(
        self, model, node_id, in_edges, b_vars, fw_vars, lw_vars
    ):
        """Arrives(n, n') = fw(n') if dst_access == src_access else lw(n')"""
        arrives_terms = []

        for edge in in_edges:
            src_id = edge.id
            dst_node = self.get_node(node_id)
            src_node = self.get_node(src_id)

            if (
                src_node.type != DFGNodeType.AFFINE
                or dst_node.type != DFGNodeType.AFFINE
            ):
                continue

            for dst_perm_idx, dst_info in enumerate(dst_node.node_info):
                for src_perm_idx, src_info in enumerate(src_node.node_info):
                    val = edge.value
                    assert val in dst_info.loads_map and val in src_info.stores_map
                    dst_access = dst_info.loads_map[val].access_map
                    src_access = src_info.stores_map[val].access_map

                    # TODO: need to check trip counts?
                    # fifo case
                    if dst_access == src_access:
                        term = model.addVar(
                            vtype=GRB.INTEGER,
                            name=f"arrive_{src_id}_{node_id}_{src_perm_idx}_{dst_perm_idx}",
                        )
                        model.addConstr(
                            term
                            == b_vars[(src_id, src_perm_idx)]
                            * b_vars[(node_id, dst_perm_idx)]
                            * fw_vars[src_id]
                        )

                        arrives_terms.append(term)

                    else:
                        term = model.addVar(
                            vtype=GRB.INTEGER,
                            name=f"arrive_{src_id}_{node_id}_{src_perm_idx}_{dst_perm_idx}",
                        )
                        model.addConstr(
                            term
                            == b_vars[(src_id, src_perm_idx)]
                            * b_vars[(node_id, dst_perm_idx)]
                            * (lw_vars[src_id])
                        )
                        arrives_terms.append(term)

        return arrives_terms

    def _add_first_write_time_constraints(
        self, model, b_vars, st_vars, fw_vars, topo_order
    ):
        r"""fw(n) = st(n) + \sum_{b \in B_n} [FW_n * II_n * b]"""
        for node_id in topo_order:
            node = self.get_node(node_id)
            if node.type != DFGNodeType.AFFINE or not self.out_edges.get(node_id, []):
                continue

            fw_terms = [st_vars[node_id]]
            for perm_idx, node_info in enumerate(node.node_info):
                for out_edge in self.out_edges[node_id]:
                    src_op = out_edge.src_op
                    if src_op in node_info.stores_map:
                        edge_info = node_info.stores_map[src_op]
                        first_time = edge_info.first_element_time
                        ii = node_info.II

                        term = model.addVar(
                            vtype=GRB.INTEGER, name=f"fw_term_{node_id}_{perm_idx}"
                        )

                        model.addConstr(
                            term == ii * b_vars[(node_id, perm_idx)] * first_time,
                            name=f"fw_term_{node_id}_{perm_idx}",
                        )

                        fw_terms.append(term)

            model.addConstr(
                fw_vars[node_id] == gp.quicksum(fw_terms), name=f"fw_constr_{node_id}"
            )

    def _add_last_write_time_constraints(
        self, model, b_vars, st_vars, lw_vars, topo_order
    ):
        r"""lw(n) = max_{n' \in ins(n)} [Depend(n, n') + Epilogue(n, n')]"""
        for node_id in topo_order:
            in_edges = self.in_edges.get(node_id, [])
            if not in_edges:
                continue

            if self.get_node(node_id).type == DFGNodeType.RET:
                # if it is a return, consider all incoming edges and let the last write time be the maximum of the last write times of the incoming edges
                lw_terms = [lw_vars[edge.id] for edge in in_edges]
                model.addConstr(
                    lw_vars[node_id] == gp.max_(lw_terms), name=f"lw_constr_{node_id}"
                )
                continue

            lw_terms = []

            # For each incoming edge, compute LW constraints
            for edge in in_edges:
                src_id = edge.id
                dst_node = self.get_node(node_id)

                # Compute relative last read terms
                rlr_terms = self._compute_relative_last_read_terms(
                    model, edge, node_id, dst_node, b_vars, st_vars
                )

                # Compute dependency term
                depend_term = self._compute_depend_term(
                    model, src_id, node_id, rlr_terms, st_vars, lw_vars
                )

                # Compute epilogue term
                epilogue_term = self._compute_epilogue_term(
                    model, src_id, node_id, dst_node, depend_term, lw_vars, b_vars
                )

                # Combine terms for this edge
                lw_term = model.addVar(
                    vtype=GRB.INTEGER, name=f"lw_term_{src_id}_{node_id}"
                )

                model.addConstr(
                    lw_term == depend_term + epilogue_term,
                    name=f"lw_term_{src_id}_{node_id}",
                )
                lw_terms.append(lw_term)

            if lw_terms:
                model.addGenConstrMax(
                    lw_vars[node_id], lw_terms, name=f"lw_constr_{node_id}"
                )

    def _compute_relative_last_read_terms(
        self, model, edge, node_id, dst_node, b_vars, st_vars
    ):
        """Compute relative last read terms for selected permutation [II * lr_time * b]"""
        rlr_terms = []
        src_id = edge.id

        if dst_node.type == DFGNodeType.AFFINE:
            rlr_terms.append(st_vars[node_id])

            for perm_idx, node_info in enumerate(dst_node.node_info):
                dst_op = edge.dst_op
                if dst_op in node_info.loads_map:
                    lr_time = node_info.loads_map[dst_op].last_element_time
                    ii = node_info.II

                    term = model.addVar(
                        vtype=GRB.INTEGER,
                        lb=0,
                        name=f"rlr_term_{src_id}_{node_id}_{perm_idx}",
                    )

                    model.addConstr(term == lr_time * ii * b_vars[(node_id, perm_idx)])

                    rlr_terms.append(term)

        return rlr_terms

    def _compute_depend_term(self, model, src_id, node_id, rlr_terms, st_vars, lw_vars):
        r"""Depend(n, n') = max(st(n) + sum(b \in B_n) [LR_n^n'], lw(n'))"""
        depend_term = model.addVar(vtype=GRB.INTEGER, name=f"depend_{src_id}_{node_id}")

        st_plus_lr = model.addVar(
            vtype=GRB.INTEGER, lb=0, name=f"st_plus_lr_{src_id}_{node_id}"
        )
        model.addConstr(
            st_plus_lr == st_vars[node_id] + gp.quicksum(rlr_terms),
            name=f"st_plus_lr_constr_{src_id}_{node_id}",
        )

        model.addConstr(
            depend_term == gp.max_(st_plus_lr, lw_vars[src_id]),
            name=f"depend_{src_id}_{node_id}",
        )

        return depend_term

    def _compute_epilogue_term(
        self, model, src_id, node_id, dst_node, depend_term, lw_vars, b_vars
    ):
        r"""Epilogue(n, n') = sum(b \in B_n) [(lw_n - LR_n^{n'}) * b]"""
        epilogue_term = model.addVar(
            vtype=GRB.INTEGER, name=f"epilogue_{src_id}_{node_id}"
        )

        epi_terms = []
        if dst_node.type == DFGNodeType.AFFINE:
            for dst_perm_idx, _ in enumerate(dst_node.node_info):
                # (lw_n - lr_n^{n'}) * b
                term = model.addVar(
                    vtype=GRB.INTEGER,
                    name=f"epi_term_{src_id}_{node_id}_{dst_perm_idx}",
                )

                model.addConstr(
                    term
                    == (depend_term - lw_vars[src_id]) * b_vars[(node_id, dst_perm_idx)]
                )
                epi_terms.append(term)

        model.addConstr(
            epilogue_term == gp.quicksum(epi_terms),
            name=f"epilogue_{src_id}_{node_id}",
        )

        return epilogue_term

    @classmethod
    def from_module(cls, module, dsp_factors=None, mem_r_ports=None, mem_w_ports=None):
        """Create a dataflow graph from an MLIR module."""
        dfg = cls(
            dsp_factors=dsp_factors, mem_r_ports=mem_r_ports, mem_w_ports=mem_w_ports
        )

        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp):
                dfg.block = op.entry_block
                break

        dfg.init()

        return dfg
