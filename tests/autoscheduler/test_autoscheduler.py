# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
import gurobipy as gp
from gurobipy import GurobiError

from allo.ir.types import int32, float32
from allo.autoscheduler.dfg import DFG
from allo.autoscheduler.passes import dataflow_optimization_pass


def check_gurobi_license():
    """Check if Gurobi license is valid for large tests."""
    try:
        env = gp.Env(empty=True)
        env.start()
    except GurobiError as e:
        return False
    return True


def test_simple():
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
    s = dataflow_optimization_pass(s, debug_point="dataflow_canonicalization")
    dfg = DFG.from_module(s.module)

    permutations = dfg.createGraphParallelismPerformanceModel()
    assert permutations[0][1] != permutations[1][1]


def matrix_multiply(A: int32[8, 8], B: int32[8, 8]) -> int32[8, 8]:
    C: int32[8, 8] = 0
    for i in range(8):
        for j in range(8):
            for k in range(8):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
    return C


def test_3mm():
    if not check_gurobi_license():
        print("❌ Gurobi license error: skipping test_3mm")
        return

    def three_mm(
        A: int32[8, 8], B: int32[8, 8], C: int32[8, 8], D: int32[8, 8]
    ) -> int32[8, 8]:
        E: int32[8, 8] = matrix_multiply(A, B)
        F: int32[8, 8] = matrix_multiply(C, D)
        return matrix_multiply(E, F)

    s = allo.customize(three_mm)
    s = dataflow_optimization_pass(s, debug_point="dataflow_canonicalization")
    module = s.module

    dfg = DFG.from_module(module)
    dfg.createGraphParallelismPerformanceModel()


if __name__ == "__main__":
    pytest.main([__file__])
