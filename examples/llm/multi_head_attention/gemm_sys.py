from allo import dsl, template
from allo.ir.types import int4, int8, int16, int32, index, Int, UInt
from allo.ir.utils import MockBuffer

from allo._mlir.ir import (
    InsertionPoint,
    StringAttr,
    IndexType,
    MemRefType,
    FlatSymbolRefAttr,
    AffineMap,
    AffineMapAttr,
    ShapedType,
)
from allo._mlir.ir import Type as MLIRType

from allo._mlir.dialects import (
    memref as memref_d,
    affine as affine_d,
    arith as arith_d,
    func as func_d,
)
from allo._mlir.dialects.affine import AffineExpr
import re


def systolic_tile[
    TyA, TyB, TyC, K: int32, Mt: int32, Nt: int32
](A: "TyA[Mt, K]", B: "TyB[K, Nt]", C: "TyC[Mt, Nt]"):
    A_fifo: TyA[Mt, Nt + 1, K]
    B_fifo: TyB[Nt, Mt + 1, K]
    A_drain: TyA[Mt]
    B_drain: TyB[Nt]

    for k in range(K, name="data_load"):
        # Can be fully unrolled inside this loop,
        # once A and B are correctly partitioned
        for m in range(Mt):
            A_fifo[m, 0, k] = A[m, k]
        for n in range(Nt):
            B_fifo[n, 0, k] = B[k, n]
    for i, j in dsl.grid(Mt, Nt, name="PE"):
        with template.meta_if(TyA == int8 and TyB == int16 and TyC == int32):
            PE_kernel_packed_int8xint8[K, Mt, Nt](
                A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j
            )
        with template.meta_else():
            PE_kernel[TyA, TyB, TyC, K, Mt, Nt](
                A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j
            )
    for k in range(K, name="data_drain"):
        for m in range(Mt):
            A_drain[m] = A_fifo[m, Nt, k]
        for n in range(Nt):
            B_drain[n] = B_fifo[n, Mt, k]

def schedule_systolic(s):
    if s.top_func_name == "systolic":
        assert len(s.inst_list) == 8
        tile_name = "systolic_tile"
        M0, M1 = s.inst_list[-2], s.inst_list[-1]
    elif s.top_func_name == "packed_systolic":
        assert len(s.inst_list) == 9
        tile_name = "systolic_tile"
        M0, M1 = s.inst_list[-3], s.inst_list[-2]
    elif s.top_func_name == "packed_int8xint8_systolic":
        assert len(s.inst_list) == 6
        tile_name = "systolic_tile"
        M0, M1 = s.inst_list[-3], s.inst_list[-2]
    else:
        raise ValueError(
            f"Cannot apply `schedule_systolic` to function: {s.top_func_name}"
        )
    s.partition(s.local_C, dim=0)  # required, otherwise it will fail dataflow checking
    s.partition(s.local_A, dim=1)
    s.partition(s.local_B, dim=2)
    load_A_loop = s.get_loops(s.top_func_name)["outer_tile"]["ai"]
    if str(load_A_loop.loop.attributes["upperBoundMap"]) == "affine_map<() -> (1)>":
        load_A_loop = s.get_loops(s.top_func_name)["outer_tile"]["ak"]
    s.pipeline(load_A_loop)
    load_B_loop = s.get_loops(s.top_func_name)["outer_tile"]["bj"]
    if str(load_B_loop.loop.attributes["upperBoundMap"]) == "affine_map<() -> (1)>":
        load_B_loop = s.get_loops(s.top_func_name)["outer_tile"]["bk"]
    s.pipeline(load_B_loop)
    store_C_loop = s.get_loops(s.top_func_name)["outer_tile"]["si"]
    if str(store_C_loop.loop.attributes["upperBoundMap"]) == "affine_map<() -> (1)>":
        store_C_loop = s.get_loops(s.top_func_name)["outer_tile"]["sj"]
    s.pipeline(store_C_loop)
    tile_loop = s.get_loops(s.top_func_name)["outer_tile"]["ni"]
    s.dataflow(tile_loop)
    pe = s.unfold(f"{tile_name}:PE", [0, 1])  # specify which are spatial loops
    s.to(MockBuffer(tile_name, "A_fifo"), pe, axis=1, depth=M0 + 1)
    s.to(MockBuffer(tile_name, "B_fifo"), pe, axis=0, depth=M1 + 1)
    return s

s = allo.schedule(systolic_tile)
schedule_systolic(s)
mod = s.build(target = "vitis_hls", mode = "csyn", project = "systolic_tile.prj")()
