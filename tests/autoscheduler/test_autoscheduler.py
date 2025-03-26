# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import allo
from allo.ir.types import int32, float32

from allo.autoscheduler.dfg import DFG, DFGNodeType, Node
from allo.autoscheduler.passes import dataflow_optimization_pass, outline_loops
from allo.autoscheduler.util import compose_affine_maps
from allo._mlir.ir import WalkResult, AffineMap

def test_simple():
    def simple() -> int32[10,10]:
        A: int32[10,10]
        B: int32[10,10]
        for i in range(10):
            for j in range(10):
                A[i, j] = i + j
        
        for j in range(10):
            for i in range(10):
                B[i, j] = A[i, j] + 1
        
        return B
    
    s = allo.customize(simple)
    s = dataflow_optimization_pass(s, debugPoint="dataflow_canonicaliation")
    dfg = DFG.from_module(s.module)

    module = s.module
    module = outline_loops(module, dfg)
    print(module)
    assert False

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
    def three_mm(
        A: int32[8, 8], B: int32[8, 8], C: int32[8, 8], D: int32[8, 8]
    ) -> int32[8, 8]:
        E: int32[8, 8] = matrix_multiply(A, B)
        F: int32[8, 8] = matrix_multiply(C, D)
        return matrix_multiply(E, F)

    s = allo.customize(three_mm)
    s = dataflow_optimization_pass(s, debugPoint="dataflow_canonicaliation")
    module = s.module

    dfg = DFG.from_module(module)
    dfg.createGraphParallelismPerformanceModel()

if __name__ == "__main__":
    pytest.main([__file__])
