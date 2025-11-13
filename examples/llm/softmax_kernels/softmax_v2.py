import math, allo
import numpy as np
from allo._mlir import ir as mlir_ir
StringAttr = mlir_ir.StringAttr
from allo.ir.types import float32, int32
from allo.autoscheduler.passes import dataflow_optimization_pass
from allo.autoscheduler.config import AutoschedulerConfig
from allo.autoscheduler.dfg import DFG
from gurobipy import GurobiError
from allo.customize import Partition as partition
from allo.ir.utils import MockBuffer

L = 64
Ty = float32
MIN_FLOAT32:Ty = -3.402823466e+38  # minimum float32 value

def softmax_v1(QK_in: Ty[L, L]) -> Ty[L, L]:
    exp_buf:Ty[L, L]      # TEMP for exponentials
    QK_out: Ty[L, L]
    max_vals: Ty[L] = MIN_FLOAT32
    rows_total: Ty[L] = 0.0
    invs: Ty[L]
    # --- loop 1
    for i1 in range(L):
        for j1 in range(L):
            v:Ty = QK_in[i1, j1]
            m:Ty = max_vals[i1]                   # affine.load
            # this select lowers to arithmetic, not a branch
            max_vals[i1] = v if v > m else m   # affine.store
        for j2 in range(L, name="j2"):
            e:Ty = allo.exp(QK_in[i1, j2] - max_vals[i1])
            exp_buf[i1, j2] = e
            rows_total[i1] += e
        inv:Ty = 1.0 / rows_total[i1] #this does not catch a division by zero
        invs[i1] = inv
    # --- loop 2: get output
    for i2 in allo.grid(L, name="i2"):
        tmp: Ty = invs[i2]
        for j3 in allo.grid(L, name="j3"):
            QK_out[i2, j3] = exp_buf[i2, j3] * tmp

    return QK_out

def softmax_v2(QK_in: Ty[L, L]) -> Ty[L, L]:
    QK_out: Ty[L, L] = 0.0
    rows_total: Ty[L] = 0.0
    exp_buf:Ty[L, L] = 0.0      # TEMP for exponentials
    max_vals: Ty[L] = MIN_FLOAT32
    for i in allo.grid(L):
        # Use local variable to break dependency chain
        local_max: Ty = MIN_FLOAT32
        for j1 in allo.grid(L):
            val: Ty = QK_in[i, j1]
            local_max = val if val > local_max else local_max
        max_vals[i] = local_max
        # for j1 in range(L):
        #     v:Ty = QK_in[i, j1]
        #     m:Ty = max_vals[i]                   # affine.load
        #     max_vals[i] = v if v > m else m   # affine.store

        for j2 in allo.grid(L):
            e:Ty = allo.exp(QK_in[i, j2] - max_vals[i])
            exp_buf[i, j2] = e
            rows_total[i] += e
            
        inv:Ty = 1.0 / rows_total[i] # this does not catch a division by zero
        for j3 in allo.grid(L):
            QK_out[i, j3] = exp_buf[i, j3] * inv

    return QK_out

def custom_softmax(QK_in: Ty[L, L]) -> Ty[L, L]:
    exp_buf:Ty[L, L] = 0.0      # TEMP for exponentials
    QK_out: Ty[L, L] = 0.0
    max_vals: Ty[L] = MIN_FLOAT32
    rows_total: Ty[L] = 0.0
    invs: Ty[L] = 0.0

    # --- Loop 1: scan QK_in and update max_vals in a perfect 2D nest
    for i1 in allo.grid(L, name = "i1"):
        local_max: Ty = MIN_FLOAT32
        for j1 in allo.grid(L, name = "j1"):
            v:Ty = QK_in[i1, j1]
            m:Ty = local_max             
            local_max = v if v > m else m  
        max_vals[i1] = local_max

    # --- loop 2: compute exponentials and row sums 
    for i2 in allo.grid(L, name = "i2"):
        local_max: Ty = max_vals[i2]
        for j2 in allo.grid(L, name = "j2"):
            e:Ty = allo.exp(QK_in[i2, j2] - local_max)
            exp_buf[i2, j2] = e

    # --- loop 3: compute row sums
    for i4 in allo.grid(L, name = "i4"):
        for j4 in allo.grid(L, name = "j4"):
            rows_total[i4] += exp_buf[i4, j4]
        inv:Ty = 1.0 / rows_total[i4] #this does not catch a division by zero
        invs[i4] = inv

    # --- loop 4: compute output
    for i3, j3 in allo.grid(L, L, name = "i3"):
        QK_out[i3, j3] = exp_buf[i3, j3] * invs[i3]
    return QK_out

def test_softmax_v1():
    QK_in = np.random.rand(L, L).astype(np.float32)
    QK_out = np.zeros((L, L), dtype=np.float32)
    base_sch = allo.customize(softmax_v1)
    opt_sch = allo.customize(softmax_v1)
    schedule_softmax_v1(opt_sch)

    #base_mod = base_sch.build()
    #opt_mod = opt_sch.build()
    #np.testing.assert_allclose(base_mod(QK_in), opt_mod(QK_in),  rtol=1e-5, atol=1e-5)
    #print("passed functional simulation 1")
    opt_mod = opt_sch.build(target="vitis_hls", mode="csyn", project="softmax_v1_opt.prj")()


def test_softmax_v2():
    #generate random input
    QK_in = np.random.rand(L, L).astype(np.float32)
    QK_out = np.zeros((L, L), dtype=np.float32)
    #create base and optimized schedules
    base_sch = allo.customize(softmax_v2)
    opt_sch = allo.customize(softmax_v2)

    schedule_softmax_v2(opt_sch)

    #base_mod = base_sch.build()
    #opt_mod = opt_sch.build()
    #np.testing.assert_allclose(base_mod(QK_in), opt_mod(QK_in),  rtol=1e-5, atol=1e-5)
    #print("passed functional simulation 1")
    opt_mod = opt_sch.build(target="vitis_hls", mode="csyn", project="softmax_v2_opt.prj")()

    # opt_mod = opt_sch.build(target="vitis_hls", mode="sw_emu", project="sw_softmax_v2_opt.prj")(QK_in, QK_out)
    # np.testing.assert_allclose(base_mod(QK_in), QK_out,  rtol=1e-5, atol=1e-5)
    # print("passed functional simulation 3")

def test_custom_softmax():
    QK_in = np.random.rand(L, L).astype(np.float32)
    s = allo.customize(custom_softmax)
    schedule_custom_softmax(s)
    mod = s.build(target="vitis_hls", mode="csyn", project="custom_softmax.prj")()

def test_function_equivalence():
    QK_in = np.random.rand(L, L).astype(np.float32)
    QK_out1 = np.zeros((L, L), dtype=np.float32)
    QK_out2 = np.zeros((L, L), dtype=np.float32)
    base_mod = allo.customize(custom_softmax).build()
    opt2_sch = allo.customize(softmax_v2)
    opt1_sch = allo.customize(softmax_v1)

    schedule_softmax_v2(opt2_sch)
    schedule_softmax_v1(opt1_sch)

    opt1_mod = opt1_sch.build(target="vitis_hls", mode="sw_emu", project="sw_softmax_v1_opt.prj")(QK_in, QK_out1)
    opt2_mod = opt2_sch.build(target="vitis_hls", mode="sw_emu", project="sw_softmax_v2_opt.prj")(QK_in, QK_out2)

    np.testing.assert_allclose(base_mod(QK_in), QK_out1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(base_mod(QK_in), QK_out2, rtol=1e-5, atol=1e-5)   
    print("passed functional simulation 2")

def schedule_custom_softmax(opt_sch):
    # partitions for loop 1
    #opt_sch.partition(opt_sch.QK_in, partition_type=partition.Cyclic, dim=2, factor = 8)
    
    # partitions for loop 2
    opt_sch.partition(opt_sch.exp_buf, partition_type=partition.Cyclic, dim=2, factor = 8)
    opt_sch.partition(opt_sch.max_vals, partition_type=partition.Cyclic, dim=1, factor = 8)

    # partitions for loop 4
    opt_sch.partition(opt_sch.QK_out, partition_type=partition.Cyclic, dim=2, factor = 32)

    # partitions for loop 4
    #opt_sch.partition(opt_sch.invs, partition_type=partition.Cyclic, dim=1, factor = 4)

    # opt_sch.unroll("i2", factor = 4)
    # opt_sch.pipeline("j1", initiation_interval=2)

    # opt_sch.pipeline("j2", initiation_interval=1)  # Higher II for fadd + store timing
    # opt_sch.unroll("j2", factor = 8)

    # opt_sch.pipeline("j3")
    # opt_sch.unroll("i3", factor = 4)
    #opt_sch.unroll("j3", factor = 32)

    # opt_sch.pipeline("j4")
    # opt_sch.unroll("i4", factor = 4)
    
    ### streaming fifos ###
    # loop 1 to loop 2
    opt_sch.to(opt_sch.max_vals, "i2")

    #loop 2 to loop 3
    #opt_sch.to(opt_sch.exp_buf, "j4")

    #loop 2 to loop 4 and loop 3 to loop 4
    #opt_sch.to(opt_sch.exp_buf, "i3") # loop 2 to loop 4
    opt_sch.to(opt_sch.invs, "i3") # loop 3 to loop 4

    #opt_sch.dataflow("softmax_v1")

def schedule_softmax_v2(opt_sch):
    # opt_sch.pipeline("j1")
    # opt_sch.unroll("j1", factor = H)

    opt_sch.pipeline("j2")
 
    opt_sch.unroll("j3", factor = 32)

    #opt_sch.buffer_at(opt_sch.max_vals, "i")
    #opt_sch.buffer_at(opt_sch.exp_buf, "i")

    opt_sch.partition(opt_sch.QK_in, partition_type=partition.Cyclic, dim=2, factor = 32)
    opt_sch.partition(opt_sch.QK_out, partition_type=partition.Cyclic, dim=2, factor = 32)
    # opt_sch.partition(max_vals_buf, partition_type=partition.Cyclic, dim=1, factor = 32)
    # opt_sch.partition(exp_buf_buf, partition_type=partition.Cyclic, dim=2, factor = 32)
    opt_sch.partition(opt_sch.rows_total, partition_type=partition.Cyclic, dim=1, factor = 32)

    # opt_sch.pipeline("i")
    # opt_sch.dataflow("softmax_v2")

def schedule_softmax_v1(opt_sch):
    opt_sch.partition(opt_sch.QK_in, partition_type=partition.Cyclic, dim=2, factor = 8)

    # Fix critical path timing issue - the comparison is in j1 loop!
    opt_sch.pipeline("j1", initiation_interval=2)  # Higher initiation interval to break dependency chain
    # NO unrolling for j1 - it has loop-carried dependencies that make critical path worse!

    # Fix j2 loop critical path - floating-point accumulation dependency
    #opt_sch.unroll("j2", factor = 8)
    opt_sch.pipeline("j2", initiation_interval=2)  # Higher II for fadd + store timing
    
    # opt_sch.buffer_at(opt_sch.exp_buf, "i1")
    # opt_sch.buffer_at(opt_sch.rows_total, "i1")

    #opt_sch.pipeline("j3")
    #opt_sch.unroll("j3", factor = 32)

    # Partition max_vals completely since it's critical for the j1 comparison
    opt_sch.partition(opt_sch.max_vals, partition_type=partition.Complete, dim=1)
    opt_sch.partition(opt_sch.QK_out, partition_type=partition.Cyclic, dim=2, factor = 32)
    # Partition rows_total completely since it's critical for the j2 accumulation
    opt_sch.partition(opt_sch.rows_total, partition_type=partition.Complete, dim=1)
    
    opt_sch.to(opt_sch.exp_buf, "j3", depth = 32)
    opt_sch.to(opt_sch.invs, "i2", depth = 32)

    #opt_sch.dataflow("softmax_v1")


if __name__ == "__main__":
    #test_softmax_v2()
    #test_softmax_v1()
    test_custom_softmax()
    #test_function_equivalence()
