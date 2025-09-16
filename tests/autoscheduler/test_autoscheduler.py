# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
import gurobipy as gp
from gurobipy import GurobiError

from allo.ir.types import int32, float32
from allo.autoscheduler.dfg import DFG
from allo.autoscheduler.passes import dataflow_optimization_pass
from allo.autoscheduler.config import AutoschedulerConfig


def test_simple_graph_parallel():
    def simple() -> int32[10, 10]:
        A: int32[10, 10]
        B: int32[10, 10]
        for i in range(10):
            for j in range(10):
                A[i, j] = i + j

        for j in range(10):
            for i in range(10):
                B[i, j] = A[i, j] + 1

        return B

    s = allo.customize(simple)
    config = AutoschedulerConfig.builder().with_debug_point("dataflow_canonicalization")
    s = dataflow_optimization_pass(s, config)

    dfg = DFG.from_module(s.module)
    sol = dfg.create_performance_model(enable_tile=False, debug_output="simple")
    assert sol.loop_permutations[0][1] != sol.loop_permutations[1][1]


def test_simple2():
    def simple() -> int32[10, 10]:
        A: int32[10, 10]
        B: int32[10, 10]
        for i in range(10):
            for j in range(10):
                A[i, j] = i + j

        for i in range(10):
            for j in range(10):
                B[i, j] = A[i, j] + 1

        return B

    s = allo.customize(simple)
    config = AutoschedulerConfig.builder().with_debug_point("dataflow_canonicalization")
    s = dataflow_optimization_pass(s, config)

    dfg = DFG.from_module(s.module)
    dfg.print_as_dot("simple2.dot")
    sol = dfg.create_performance_model(enable_tile=False, debug_output="simple2")
    assert sol.loop_permutations[0][1] == sol.loop_permutations[1][1]


def matrix_multiply(A: int32[8, 8], B: int32[8, 8]) -> int32[8, 8]:
    C: int32[8, 8] = 0
    for i in range(8):
        for j in range(8):
            for k in range(8):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
    return C


def test_3mm():
    def three_mm(
        A: int32[8, 8], B: int32[8, 8], C: int32[8, 8], D: int32[8, 8]
    ) -> int32[8, 8]:
        E: int32[8, 8] = matrix_multiply(A, B)
        F: int32[8, 8] = matrix_multiply(C, D)
        return matrix_multiply(E, F)

    s = allo.customize(three_mm)
    config = AutoschedulerConfig.builder().with_debug_point("dataflow_canonicalization")
    s = dataflow_optimization_pass(s, config)
    module = s.module

    dfg = DFG.from_module(module)
    try:
        res = dfg.create_performance_model(enable_tile=False, debug_output="simple")
    except GurobiError as e:
        if "Model too large for size-limited license" in str(e):
            pytest.skip(
                "Skipping test: model too large for size-limited Gurobi license"
            )
        else:
            raise e


def test_simple_node_parallel():
    def simple() -> float32[64, 64]:
        A: float32[64, 64]
        B: float32[64, 64]
        for i in range(64):
            for j in range(15):
                A[i, j] = i + j

        for j in range(15):
            for i in range(64):
                B[i, j] = A[i, j] + 1

        return B

    s = allo.customize(simple)
    config = AutoschedulerConfig.builder().with_debug_point("dataflow_canonicalization")
    s = dataflow_optimization_pass(s, config)

    dfg = DFG.from_module(s.module)
    res = dfg.create_performance_model(enable_tile=False, debug_output="simple")
    tiling_factors = dfg.create_performance_model(
        pinned_permutations=res.loop_permutations,
        enable_tile=True,
        debug_output="simple-node",
        dsp_limit=20,
        verbose=True,
    )

    sol = tiling_factors.tiling_factors
    assert len(sol.keys()) == 2
    assert all(
        tiling_factor[1] in (4, 5)
        for factors in sol.values()
        for tiling_factor in factors
    )


def test_simple_node_parallel_full_unroll():
    def simple() -> float32[64, 64]:
        A: float32[64, 64]
        B: float32[64, 64]
        for i in range(64):
            for j in range(64):
                A[i, j] = i + j

        for j in range(64):
            for i in range(64):
                B[i, j] = A[i, j] + 1

        return B

    s = allo.customize(simple)
    config = AutoschedulerConfig.builder().with_debug_point("dataflow_canonicalization")
    s = dataflow_optimization_pass(s, config)

    dfg = DFG.from_module(s.module)
    res = dfg.create_performance_model(enable_tile=False, debug_output="simple")
    tiling_factors = dfg.create_performance_model(
        pinned_permutations=res.loop_permutations,
        enable_tile=True,
        debug_output="simple-node",
        dsp_limit=2**12,
        verbose=True,
    )

    sol = tiling_factors.tiling_factors
    assert len(sol) == 2
    assert all(factor[1] == 32 for factors in sol.values() for factor in factors)


def test_simple_node_parallel_infeasible():
    def simple() -> float32[64, 64]:
        A: float32[64, 64]
        B: float32[64, 64]
        for i in range(64):
            for j in range(64):
                A[i, j] = i + j

        for j in range(64):
            for i in range(64):
                B[i, j] = A[i, j] + 1

        return B

    s = allo.customize(simple)
    config = AutoschedulerConfig.builder().with_debug_point("dataflow_canonicalization")
    s = dataflow_optimization_pass(s, config)

    dfg = DFG.from_module(s.module)
    res = dfg.create_performance_model(
        enable_tile=False,
        verbose=True,
        debug_output="simple-node",
    )

    with pytest.raises(RuntimeError, match="Optimization failed with status 3"):
        dfg.create_performance_model(
            pinned_permutations=res.loop_permutations,
            enable_tile=True,
            debug_output="simple-node",
            dsp_limit=0,
            verbose=True,
        )


if __name__ == "__main__":
    pytest.main([__file__])
