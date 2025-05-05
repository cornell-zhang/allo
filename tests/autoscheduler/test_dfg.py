# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import allo
from allo.ir.types import int32, float32

from allo.autoscheduler.dfg import DFG, DFGNodeType, Node
from allo.autoscheduler.passes import dataflow_optimization_pass
from allo.autoscheduler.util import compose_affine_maps
from allo._mlir.ir import AffineMap, AffineExpr


def check_edges(dfg: DFG):
    for src, dst, val in dfg.edges:
        assert any(
            edge.id == dst for edge in dfg.out_edges[src]
        ), f"Edge from {src} to {dst} not in out_edges"
        assert any(
            edge.id == src for edge in dfg.in_edges[dst]
        ), f"Edge from {src} to {dst} not in in_edges"
        if dfg.get_node(src).type in (
            DFGNodeType.ALLOC,
            DFGNodeType.RET,
        ) or dfg.get_node(dst).type in (DFGNodeType.ALLOC, DFGNodeType.RET):
            continue
        assert any(
            store.operands[1] == val for store in dfg.get_node(src).stores
        ), f"Value {val} not in stores of {dfg.get_node(src)}"
        assert any(
            load.operands[0] == val for load in dfg.get_node(dst).loads
        ), f"Value {val} not in loads of {dfg.get_node(dst)}"


def check_node_info(dfg: DFG):
    for node_id, node in dfg.nodes.items():
        if node.type != DFGNodeType.AFFINE:
            continue

        assert len(node.loop_info) > 0, f"Node {node_id} has no loop info"

        num_loops = len(node.loop_info)
        expected_permutations = math.factorial(num_loops)
        assert (
            len(node.node_info) == expected_permutations
        ), f"Node {node_id} has {len(node.node_info)} permutations, expected {expected_permutations}"
        assert (
            node.DSP_factor >= 0
        ), f"Node {node_id} has negative DSP factor: {node.DSP_factor}"


def matrix_multiply(A: int32[8, 8], B: int32[8, 8]) -> int32[8, 8]:
    C: int32[8, 8] = 0
    for i in range(8):
        for j in range(8):
            for k in range(8):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
    return C


def three_mm(
    A: int32[8, 8], B: int32[8, 8], C: int32[8, 8], D: int32[8, 8]
) -> int32[8, 8]:
    E: int32[8, 8] = matrix_multiply(A, B)
    F: int32[8, 8] = matrix_multiply(C, D)
    return matrix_multiply(E, F)


def test_3mm():
    s = allo.customize(three_mm)
    s = dataflow_optimization_pass(s, debug_point="dataflow_canonicalization")
    module = s.module
    print(module)
    dfg = DFG.from_module(module)
    check_edges(dfg)
    check_node_info(dfg)


def func() -> int32[10, 10, 10]:
    A: int32[10, 10, 10]
    for i, j in allo.grid(10, 10):
        for k in range(0, 10, 2):
            A[k, i, j] += 0
    return A


def test_loop_info():
    s = allo.customize(func)
    module = s.module
    dfg = DFG.from_module(module)
    affine_nodes = [
        node for node in dfg.nodes.values() if node.type == DFGNodeType.AFFINE
    ]
    assert len(affine_nodes) == 1
    node = affine_nodes[0]
    assert len(node.loop_info) == 3
    assert all(loop.lower_bound == 0 for loop in node.loop_info)
    assert all(loop.upper_bound == 10 for loop in node.loop_info)
    assert all(loop.step == 1 and loop.trip_count == 10 for loop in node.loop_info[:2])
    assert node.loop_info[2].step == 2 and node.loop_info[2].trip_count == 5


def test_node_info():
    s = allo.customize(func)
    s = dataflow_optimization_pass(s, debug_point="dataflow_canonicalization")

    module = s.module
    dfg = DFG.from_module(module)
    print(module)

    affine_node: Node = [
        node for node in dfg.nodes.values() if node.type == DFGNodeType.AFFINE
    ][0]

    assert len(affine_node.node_info) == math.factorial(len(affine_node.loop_info))
    original_access_map = AffineMap.get_permutation([2, 0, 1], module.context)

    for info in affine_node.node_info:
        # check access map
        perm = info.permutation
        inverted_perm = [perm.index(i) for i in range(len(perm))]
        inverse_loop_map = AffineMap.get_permutation(inverted_perm, module.context)
        # sanity check composing function with inverse is identity
        assert compose_affine_maps(
            AffineMap.get_permutation(perm, module.context), inverse_loop_map
        ) == AffineMap.get_identity(3, module.context)

        # check access pattern
        permuted_access_map = compose_affine_maps(inverse_loop_map, original_access_map)
        assert (
            permuted_access_map
            == info.stores_map[affine_node.stores[0].opview.memref].access_map
        )
        assert (
            permuted_access_map
            == info.loads_map[affine_node.loads[0].opview.memref].access_map
        )

        # check II is 1 for all permutations
        assert info.II == 1

        # check first and last access times
        assert (
            info.stores_map[affine_node.stores[0].opview.memref].first_element_time == 0
        )
        assert (
            info.loads_map[affine_node.loads[0].opview.memref].first_element_time == 0
        )

        assert (
            info.stores_map[affine_node.stores[0].opview.memref].last_element_time
            == 499
        )
        assert (
            info.loads_map[affine_node.loads[0].opview.memref].last_element_time == 499
        )


def test_DSP():
    def dsp_test() -> int32[10, 10]:
        A: int32[10, 10]
        for i in range(10):
            for j in range(10):
                A[i, j] = 0
        B: float32
        for i in range(10):
            B += 1.8  # 0 DSP
            B *= 2.0  # 3 DSP
            B *= 5.1  # 3 DSP
            B /= 1.1  # 14 DSP
        return A

    s = allo.customize(dsp_test)
    s = dataflow_optimization_pass(s, debug_point="dataflow_canonicalization")
    module = s.module
    dfg = DFG.from_module(module)
    affine_nodes = [
        node for node in dfg.nodes.values() if node.type == DFGNodeType.AFFINE
    ]
    for i, affine_node in enumerate(affine_nodes):
        assert affine_node.DSP_factor == [0, 21][i]


def test_II():
    def loop_with_dependency(A: int32[10, 10]) -> int32[10, 10]:
        for j in range(10):
            for i in range(1, 10):
                A[i, j] = A[i - 1, j] + 1
        return A

    s = allo.customize(loop_with_dependency)
    s = dataflow_optimization_pass(s, debug_point="dataflow_canonicalization")

    module = s.module
    dfg = DFG.from_module(module)
    affine_node = [
        node for node in dfg.nodes.values() if node.type == DFGNodeType.AFFINE
    ][0]
    for info in affine_node.node_info:
        # original loop order has carried dependency on i
        if list(info.permutation) == [0, 1]:
            assert info.II == 4

        # permuted loop order has no carried dependency on innermost loop (j)
        elif list(info.permutation) == [1, 0]:
            assert info.II == 1


def test_reduction():
    def reduction(A: int32[10]) -> int32[1]:
        sum: int32[1]
        for i in range(10):
            sum[0] += A[i]
        return sum

    s = allo.customize(reduction)
    s = dataflow_optimization_pass(s, debug_point="dataflow_canonicalization")
    module = s.module
    print(s.module)
    dfg = DFG.from_module(module)
    affine_node = [
        node for node in dfg.nodes.values() if node.type == DFGNodeType.AFFINE
    ][0]
    assert affine_node.is_reduction == True

    def not_reduction() -> int32[10]:
        B: int32[10]
        for i in range(10):
            B[i] = 0
        return B

    s = allo.customize(not_reduction)
    s = dataflow_optimization_pass(s, debug_point="dataflow_canonicalization")
    module = s.module

    dfg = DFG.from_module(module)
    affine_node = [
        node for node in dfg.nodes.values() if node.type == DFGNodeType.AFFINE
    ][0]
    assert affine_node.is_reduction == False


if __name__ == "__main__":
    pytest.main([__file__])
