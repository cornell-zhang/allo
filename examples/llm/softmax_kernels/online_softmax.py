import math, allo
import numpy as np
from allo._mlir import ir as mlir_ir
StringAttr = mlir_ir.StringAttr
from allo.ir.types import float32, int32, index
from allo.autoscheduler.passes import dataflow_optimization_pass
from allo.autoscheduler.config import AutoschedulerConfig
from allo.autoscheduler.dfg import DFG
from gurobipy import GurobiError
from allo.customize import Partition as partition
from allo.ir.utils import MockBuffer

L = 64
max_val: float32 = -3.402823466e+38
def online_softmax(QK_in: float32[L, L]) -> float32[L, L]:
    max_vals: float32[L] = max_val
    local_max_curr: float32 = max_val
    local_max_prev: float32 = max_val
    exp_prev: float32 = 0.0
    exp_curr: float32 = 0.0
    for j1 in allo.grid(L, name = "j1"):
        if QK_in[i_pos, j1] > local_max_curr:
            local_max_prev = local_max_curr
            local_max_curr = QK_in[i_pos, j1]
        exp_curr = allo.exp(local_max_prev - local_max_curr)*exp_prev + allo.exp(QK_in[i_pos, j1] - local_max_curr)
        exp_prev = exp_curr
    max_vals[i] = local_max_curr
    return max_vals

