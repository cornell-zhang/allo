# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .. import dsl, template
from ..ir.types import int4, int8, int16, int32, index, Int, UInt
from ..ir.utils import MockBuffer

from .._mlir.ir import (
    InsertionPoint,
    StringAttr,
    IndexType,
    MemRefType,
    FlatSymbolRefAttr,
    AffineMap,
    AffineMapAttr,
    ShapedType,
)
from .._mlir.ir import Type as MLIRType

from .._mlir.dialects import (
    memref as memref_d,
    affine as affine_d,
    arith as arith_d,
    func as func_d,
)
from .._mlir.dialects.affine import AffineExpr
import re


def PE_kernel[
    TyA, TyB, TyC, K: int32, Mt: int32, Nt: int32
](
    A_in: "TyA[K]",
    B_in: "TyB[K]",
    A_out: "TyA[K]",
    B_out: "TyB[K]",
    C: "TyC[Mt, Nt]",
    i: index,
    j: index,
):
    # Be careful, need to use high precision for accumulation
    v: TyC = 0
    for k in dsl.grid(K, name="reduction"):
        a: TyA = A_in[k]
        b: TyB = B_in[k]
        v += a * b
        A_out[k] = a
        B_out[k] = b
    C[i, j] = v


def PE_kernel_packed_int4xint8[
    K: int32, Mt: int32, Nt: int32
](
    A_in: "int8[K]",  # not bit-packed
    B_in: "int8[K]",  # bit-packed, each element is 4 bits
    A_out: "int8[K]",
    B_out: "int8[K]",
    C: "int16[Mt, Nt // 2]",  # bit-packed, each element is 8 bits
    i: index,
    j: index,
):
    v: int32 = 0
    for k in dsl.grid(K, name="reduction"):
        a: int8 = A_in[k]
        b_packed: int8 = B_in[k]
        b0: int4 = b_packed[0:4]
        b1: int4 = b_packed[4:8]
        s0: UInt(1) = a[7] ^ b0[3]
        s1: UInt(1) = a[7] ^ b1[3]
        au: UInt(8) = dsl.abs(a)
        b0u: UInt(4) = dsl.abs(b0)
        b1u: UInt(4) = dsl.abs(b1)
        op0: UInt(18) = 0
        op1: UInt(27) = 0
        op0[0:8] = au
        op1[0:4] = b0u
        op1[13:17] = b1u
        res: UInt(48) = op0 * op1
        res0u: UInt(12) = res[0:12]
        res1u: UInt(12) = res[13:25]
        res0: int16 = -res0u if s0 else res0u
        res1: int16 = -res1u if s1 else res1u
        v[0:16] += res0
        v[16:32] += res1
        A_out[k] = a
        B_out[k] = b_packed
    C[i, j] = v


def PE_kernel_packed_int8xint8[
    K: int32, Mt: int32, Nt: int32
](
    A_in: "int8[K]",  # not bit-packed
    B_in: "int16[K]",  # bit-packed, each element is 8 bits
    A_out: "int8[K]",
    B_out: "int16[K]",
    C: "int32[Mt, Nt]",  # bit-packed, each element is 16 bits
    i: index,
    j: index,
):
    v: int32 = 0
    for k in dsl.grid(K, name="reduction"):
        a: int8 = A_in[k]
        b_packed: int16 = B_in[k]
        b0: int8 = b_packed[0:8]
        b1: int8 = b_packed[8:16]
        s0: UInt(1) = a[7] ^ b0[7]
        s1: UInt(1) = a[7] ^ b1[7]
        au: UInt(8) = dsl.abs(a)
        b0u: UInt(8) = dsl.abs(b0)
        b1u: UInt(8) = dsl.abs(b1)
        # DSP48E1: 18x27 multiplier -> 45-bit result
        op0: UInt(18) = 0
        op1: UInt(27) = 0
        op0[0:8] = au
        op1[0:8] = b0u
        op1[17:25] = b1u
        res: UInt(45) = op0 * op1
        res0u: UInt(16) = res[0:16]
        res1u: UInt(16) = res[17:33]
        res0: int16 = -res0u if s0 else res0u
        res1: int16 = -res1u if s1 else res1u
        v[0:16] += res0
        v[16:32] += res1
        A_out[k] = a
        B_out[k] = b_packed
    C[i, j] = v


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


def systolic[
    TyA, TyB, TyC, M: int32, K: int32, N: int32, Mt: int32, Nt: int32
](A: "TyA[M, K]", B: "TyB[K, N]", C: "TyC[M, N]"):
    local_A: TyA[Mt, K]
    local_B: TyB[K, Nt]
    local_C: TyC[Mt, Nt]

    # k needs not be tiled, since it is temporal dimension
    for mi, ni in dsl.grid(M // Mt, N // Nt, name="outer_tile"):
        # reversed traversal, better for cascading systolic arrays with FIFOs
        # corresponds to the order of the previous `store_C_tile` output
        for ak, ai in dsl.grid(K, Mt, name="load_A_tile"):
            # reuse along the ni dimension
            if ni == 0:
                local_A[ai, ak] = A[mi * Mt + ai, ak]
        for bk, bj in dsl.grid(K, Nt, name="load_B_tile"):
            # reuse along the mi dimension
            # since the inner access order is different from the outer one,
            # we cannot cache as a line buffer
            local_B[bk, bj] = B[bk, ni * Nt + bj]
        systolic_tile[TyA, TyB, TyC, K, Mt, Nt](
            local_A,
            local_B,
            local_C,
        )
        # reversed traversal, better for cascading systolic arrays with FIFOs
        for sj, si in dsl.grid(Nt, Mt, name="store_C_tile"):
            C[mi * Mt + si, ni * Nt + sj] = local_C[si, sj]


def packed_systolic[
    TyA: Int,
    TyB: Int,
    TyC: Int,
    M: int32,
    K: int32,
    N: int32,
    Mt: int32,
    Nt: int32,
    P: int32,  # packing factor
](
    A: "Int(TyA.bits * P)[M // P, K]",
    B: "Int(TyB.bits * P)[K, N // P]",
    C: "Int(TyC.bits * P)[M // P, N]",
):
    local_A: TyA[Mt, K]
    local_B: TyB[K, Nt]
    local_C: TyC[Mt, Nt]

    # k needs not be tiled, since it is temporal dimension
    for mi, ni in dsl.grid(M // Mt, N // Nt, name="outer_tile"):
        # reversed traversal, better for cascading systolic arrays with FIFOs
        # corresponds to the order of the previous `store_C_tile` output
        for ak, ai in dsl.grid(K, Mt // P, name="load_A_tile"):
            # reuse along the ni dimension
            if ni == 0:
                a: Int(TyA.bits * P) = A[mi * Mt // P + ai, ak]
                for p in range(P):
                    local_A[ai * P + p, ak] = a[p * TyA.bits : (p + 1) * TyA.bits]
        for bk, bj in dsl.grid(K, Nt // P, name="load_B_tile"):
            # reuse along the mi dimension
            # since the inner access order is different from the outer one,
            # we cannot cache as a line buffer
            b: Int(TyB.bits * P) = B[bk, ni * Nt // P + bj]
            for p in range(P):
                local_B[bk, bj * P + p] = b[p * TyB.bits : (p + 1) * TyB.bits]
        systolic_tile[TyA, TyB, TyC, K, Mt, Nt](
            local_A,
            local_B,
            local_C,
        )
        # reversed traversal, better for cascading systolic arrays with FIFOs
        for sj, si in dsl.grid(Nt, Mt // P, name="store_C_tile"):
            c: Int(TyC.bits * P) = 0
            for p in range(P):
                c[p * TyC.bits : (p + 1) * TyC.bits] = local_C[si * P + p, sj]
            C[mi * Mt // P + si, ni * Nt + sj] = c


def packed_int8xint8_systolic[
    M: int32,
    K: int32,
    N: int32,
    Mt: int32,
    Nt: int32,
    P: int32,  # packing factor
](A: "Int(8 * P)[M // P, K]", B: "Int(8 * P)[K, N // P]", C: "Int(8 * P)[M // P, N]"):
    local_A: int8[Mt, K]
    local_B: int16[K, Nt // 2]
    local_C: int32[Mt, Nt // 2]

    # k needs not be tiled, since it is temporal dimension
    for mi, ni in dsl.grid(M // Mt, N // Nt, name="outer_tile"):
        # reversed traversal, better for cascading systolic arrays with FIFOs
        # corresponds to the order of the previous `store_C_tile` output
        for ak, ai in dsl.grid(K, Mt // P, name="load_A_tile"):
            # reuse along the ni dimension
            if ni == 0:
                a: Int(8 * P) = A[mi * Mt // P + ai, ak]
                for p in range(P):
                    local_A[ai * P + p, ak] = a[p * 8 : (p + 1) * 8]
        for bk, bj in dsl.grid(K, Nt // P, name="load_B_tile"):
            # reuse along the mi dimension
            # since the inner access order is different from the outer one,
            # we cannot cache as a line buffer
            b: Int(8 * P) = B[bk, ni * Nt // P + bj]
            for p in range(P // 2):
                local_B[bk, bj * P + p] = b[p * 16 : (p + 1) * 16]
        systolic_tile[int8, int16, int32, K, Mt, Nt // 2](
            local_A,
            local_B,
            local_C,
        )
        # reversed traversal, better for cascading systolic arrays with FIFOs
        for sj, si in dsl.grid(Nt // 2, Mt // P, name="store_C_tile"):
            c0: Int(8 * P) = 0
            c1: Int(8 * P) = 0
            for p in range(P):
                x: int32 = local_C[si * P + p, sj]
                c0[p * 8 : (p + 1) * 8] = x[0:16]
                c1[p * 8 : (p + 1) * 8] = x[16:32]
            C[mi * Mt // P + si, ni * Nt + 2 * sj] = c0
            C[mi * Mt // P + si, ni * Nt + 2 * sj + 1] = c1


def schedule_systolic(s):
    if s.top_func_name == "systolic":
        assert len(s.inst_list) == 8
        tile_name = "systolic_tile"
        M0, M1 = s.inst_list[-2], s.inst_list[-1]
        kernel_loop = s.get_loops(f"PE_kernel")["reduction"]["k"]
    elif s.top_func_name == "packed_systolic":
        assert len(s.inst_list) == 9
        tile_name = "systolic_tile"
        M0, M1 = s.inst_list[-3], s.inst_list[-2]
    elif s.top_func_name == "packed_int8xint8_systolic":
        assert len(s.inst_list) == 6
        tile_name = "systolic_tile"
        M0, M1 = s.inst_list[-3], s.inst_list[-2]
        kernel_loop = s.get_loops(f"PE_kernel_packed_int8xint8")["reduction"]["k"]
    else:
        raise ValueError(
            f"Cannot apply `schedule_systolic` to function: {s.top_func_name}"
        )
    s.partition(s.local_C, dim=0)  # required, otherwise it will fail dataflow checking
    s.partition(s.local_A, dim=1)
    s.partition(s.local_B, dim=2)
    load_A_loop = s.get_loops(s.top_func_name)["outer_tile"]["ak"]
    s.pipeline(load_A_loop)
    load_B_loop = s.get_loops(s.top_func_name)["outer_tile"]["bk"]
    s.pipeline(load_B_loop)
    store_C_loop = s.get_loops(s.top_func_name)["outer_tile"]["sj"]
    s.pipeline(store_C_loop)
    inner_tile_loop = s.get_loops(s.top_func_name)["outer_tile"]["ni"]
    outer_tile_loop = s.get_loops(s.top_func_name)["outer_tile"]["mi"]
    tile_loop = s.fuse(outer_tile_loop, inner_tile_loop)
    kernel_loop = None
    for kernel_name in {"PE_kernel", "PE_kernel_packed_int8xint8"}:
        try:
            kernel_loop = s.get_loops(kernel_name)["reduction"]["k"]
            break
        except RuntimeError:
            continue
    if kernel_loop is not None:
        s.pipeline(kernel_loop)
    pe = s.unfold(f"{tile_name}:PE", [0, 1])  # specify which are spatial loops
    s.to(MockBuffer(tile_name, "A_fifo"), pe, axis=1, depth=M0 + 1)
    s.to(MockBuffer(tile_name, "B_fifo"), pe, axis=0, depth=M1 + 1)
    return s


def check_systolic(sch):
    """
    This function checks if there's only one function and it has only one three-level perfect loop.
    """
    if len(sch.module.body.operations) == 1:
        if sch.module.body.operations[0].name.value == "gemm":
            gemm_func = sch.module.body.operations[0]
            if len(gemm_func.body.blocks[0].operations) == 2:
                if isinstance(
                    gemm_func.body.blocks[0].operations[0], affine_d.AffineForOp
                ):
                    affine_for_op = gemm_func.body.blocks[0].operations[0]
                    if len(affine_for_op.body.operations) == 2:
                        if isinstance(
                            affine_for_op.body.operations[0], affine_d.AffineForOp
                        ):
                            affine_for_op = affine_for_op.body.operations[0]
                            if len(affine_for_op.body.operations) == 2:
                                if isinstance(
                                    affine_for_op.body.operations[0],
                                    affine_d.AffineForOp,
                                ):
                                    return True
    return False


# pylint: disable=all
def prepare_systolic(sch, band_name):
    """
    This function outlines the k loop and builds the load/drain loops.

    Parameters
    ----------
    band_name: str
        The name of the band.

    """
    if ":" in band_name:
        func = sch._find_function(band_name.split(":")[0])
        band_name = band_name.split(":")[1]
    else:
        func = sch.top_func

    band = sch._find_band(band_name, func)
    loops = list(band)
    outer_loop = loops[0][1].loop
    middle_loop = loops[1][1].loop  # Middle loop
    inner_loop = loops[-1][1].loop  # Last/innermost loop
    i_size = int(
        re.findall(
            r"affine_map<\(\) -> \(([0-9]*)\)>",
            str(outer_loop.attributes["upperBoundMap"]),
        )[0]
    )
    j_size = int(
        re.findall(
            r"affine_map<\(\) -> \(([0-9]*)\)>",
            str(middle_loop.attributes["upperBoundMap"]),
        )[0]
    )
    k_size = int(
        re.findall(
            r"affine_map<\(\) -> \(([0-9]*)\)>",
            str(inner_loop.attributes["upperBoundMap"]),
        )[0]
    )
    # Find arithmetic operations in innermost loop
    add_ops = []
    mul_ops = []
    load_ops = []
    for op in inner_loop.body.operations:
        # Check for integer arithmetic
        if isinstance(op, (arith_d.AddIOp, arith_d.AddFOp)):
            add_ops.append(op)
        elif isinstance(op, (arith_d.MulIOp, arith_d.MulFOp)):
            mul_ops.append(op)
        elif isinstance(op, affine_d.AffineLoadOp):
            load_ops.append(op)
    assert len(add_ops) == 1
    assert len(mul_ops) == 1
    assert len(load_ops) > 1

    ### Create outlined PE Kernel Func

    # Get result type of first affine load operation
    load_type = load_ops[0].result.type
    arith_type = mul_ops[0].result.type
    # Create 1D memref type with k_size elements
    fifo_memref_type = MLIRType.parse(
        f"memref<{k_size}x{load_type}, strided<{[1]}, offset: ?>>"
    )
    res_memref_type = MemRefType.get([i_size, j_size], arith_type)
    # Create function type with four memref arguments
    func_type = func_d.FunctionType.get(
        [fifo_memref_type] * 4 + [res_memref_type] + [IndexType.get()] * 2, []
    )
    # Insert the function at the beginning of the module
    ip = InsertionPoint.at_block_begin(sch.module.body)
    pe_kernel = func_d.FuncOp("PE_kernel", func_type, ip=ip)
    pe_kernel.attributes["sym_visibility"] = StringAttr.get("private")

    # Create function body block and entry point
    entry_block = pe_kernel.add_entry_block()
    ip = InsertionPoint(entry_block)

    # Add argument names to sch.func_args
    # Assuming your arguments are named arg0, arg1, etc.
    # arg_names = [f"arg{i}" for i in range(len(pe_kernel.arguments))]
    arg_names = ["A_in", "B_in", "A_out", "B_out", "C", "i", "j"]
    sch.func_args["PE_kernel"] = arg_names

    # Create memref for accumulator
    acc_type = MemRefType.get([1], arith_type)
    acc = memref_d.AllocOp(acc_type, [], [], ip=ip).result

    # Store zero into accumulator
    zero = arith_d.ConstantOp(arith_type, 0, ip=ip).result
    zero_idx = arith_d.ConstantOp(IndexType.get(), 0, ip=ip).result
    affine_map = AffineMap.get(
        dim_count=1, symbol_count=0, exprs=[AffineExpr.get_constant(0)]
    )
    affine_attr = AffineMapAttr.get(affine_map)
    affine_d.AffineStoreOp(zero, acc, [zero_idx], affine_attr, ip=ip)

    # Create affine loop
    loop = affine_d.AffineForOp(
        lower_bound=0,
        upper_bound=k_size,
        step=1,
        iter_args=[],
        lower_bound_operands=None,
        upper_bound_operands=None,
        ip=ip,
    )
    loop.attributes["loop_name"] = StringAttr.get("k")
    loop.attributes["op_name"] = StringAttr.get("S_k_0")

    # # Create loop body
    ip = InsertionPoint(loop.body)

    # Load from first input fifo (arg 0)
    affine_map = AffineMap.get(
        dim_count=1, symbol_count=0, exprs=[AffineExpr.get_dim(0)]
    )
    affine_attr = AffineMapAttr.get(affine_map)
    a = affine_d.AffineLoadOp(
        load_type,
        pe_kernel.arguments[0],
        [loop.induction_variable],
        affine_attr,
        ip=ip,
    )

    # load from the second input fifo (arg 1)
    b = affine_d.AffineLoadOp(
        load_type,
        pe_kernel.arguments[1],
        [loop.induction_variable],
        affine_attr,
        ip=ip,
    )

    # move the cast, cast, mul tree over
    lhs_cast = mul_ops[0].operands[0].owner
    rhs_cast = mul_ops[0].operands[1].owner
    lhs_cast_new = lhs_cast.clone(ip=ip)
    lhs_cast_new.operation.replace_uses_of_with(lhs_cast.operands[0], a.result)
    rhs_cast_new = rhs_cast.clone(ip=ip)
    rhs_cast_new.operation.replace_uses_of_with(rhs_cast.operands[0], b.result)
    new_mul_op = mul_ops[0].clone(ip=ip)
    new_mul_op.operation.replace_uses_of_with(
        mul_ops[0].operands[0], lhs_cast_new.result
    )
    new_mul_op.operation.replace_uses_of_with(
        mul_ops[0].operands[1], rhs_cast_new.result
    )
    # Load from accumulator
    acc_val = affine_d.AffineLoadOp(arith_type, acc, [zero_idx], affine_attr, ip=ip)

    # Add multiplication result to accumulator value
    add_op = arith_d.AddIOp(acc_val.result, new_mul_op.result, ip=ip)

    # Store result back to accumulator
    affine_d.AffineStoreOp(add_op.result, acc, [zero_idx], affine_attr, ip=ip)

    # store a to first output fifo (arg 2)
    affine_d.AffineStoreOp(
        a.result,
        pe_kernel.arguments[2],
        [loop.induction_variable],
        affine_attr,
        ip=ip,
    )

    # store b to second output fifo (arg 3)
    affine_d.AffineStoreOp(
        b.result,
        pe_kernel.arguments[3],
        [loop.induction_variable],
        affine_attr,
        ip=ip,
    )

    # # Load from second input fifo (arg 1)
    # b = affine_d.AffineLoadOp(pe_kernel.arguments[1], [loop.induction_variable], [], ip=ip).result
    affine_d.AffineYieldOp([], ip=InsertionPoint(loop.body))

    # Load final value from accumulator
    acc_final = affine_d.AffineLoadOp(
        arith_type, acc, [zero_idx], affine_attr, ip=InsertionPoint(entry_block)
    )

    # Store accumulator value to output matrix C (arg 4) using indices i,j (args 5,6)
    affine_map = AffineMap.get(
        dim_count=2,
        symbol_count=0,
        exprs=[AffineExpr.get_dim(0), AffineExpr.get_dim(1)],
    )
    affine_attr = AffineMapAttr.get(affine_map)
    affine_d.AffineStoreOp(
        acc_final.result,
        pe_kernel.arguments[4],
        [pe_kernel.arguments[5], pe_kernel.arguments[6]],
        affine_attr,
        ip=InsertionPoint(entry_block),
    )

    func_d.ReturnOp([], ip=InsertionPoint(entry_block))

    ### Create load loop
    ip = InsertionPoint(outer_loop)

    # Create memref types for FIFOs and drains
    fifo_memref_type = MemRefType.get([i_size, j_size + 1, k_size], load_type)
    drain_memref_type = MemRefType.get([k_size], load_type)

    # Create the memrefs
    A_fifo = sch.A_fifo
    B_fifo = sch.B_fifo
    # Move A_fifo, B_fifo's op to the beginning of the function body block[0]
    A_drain = memref_d.AllocOp(drain_memref_type, [], [], ip=ip)
    B_drain = memref_d.AllocOp(drain_memref_type, [], [], ip=ip)

    # After creating the AllocOps, add name attributes
    A_drain.attributes["name"] = StringAttr.get("A_drain")
    B_drain.attributes["name"] = StringAttr.get("B_drain")

    # Then create and attach MockBuffers as you already have
    A_drain_mock_buffer = MockBuffer(func.name.value, "A_drain")
    B_drain_mock_buffer = MockBuffer(func.name.value, "B_drain")

    A_drain_mock_buffer.op = A_drain
    B_drain_mock_buffer.op = B_drain

    # setattr(sch, "A_fifo", MockBuffer(func.name.value, "A_fifo"))
    # setattr(sch, "B_fifo", MockBuffer(func.name.value, "B_fifo"))
    setattr(sch, "A_drain", MockBuffer(func.name.value, "A_drain"))
    setattr(sch, "B_drain", MockBuffer(func.name.value, "B_drain"))

    # Create data load loop nest
    # Outer k loop
    k_map = AffineMap.get(
        dim_count=0, symbol_count=0, exprs=[AffineExpr.get_constant(k_size)]
    )
    k_loop = affine_d.AffineForOp(0, k_map, 1, ip=ip)
    k_loop.attributes["loop_name"] = StringAttr.get("k")
    k_loop.attributes["op_name"] = StringAttr.get("data_load")

    # Inner i loop for loading A
    i_map = AffineMap.get(
        dim_count=0, symbol_count=0, exprs=[AffineExpr.get_constant(i_size)]
    )
    i_loop = affine_d.AffineForOp(0, i_map, 1, ip=InsertionPoint(k_loop.body))
    i_loop.attributes["loop_name"] = StringAttr.get("i")
    i_loop.attributes["op_name"] = StringAttr.get("data_load")

    # Load from A and store to A_fifo
    affine_map = AffineMap.get(
        dim_count=2,
        symbol_count=0,
        exprs=[AffineExpr.get_dim(0), AffineExpr.get_dim(1)],
    )
    affine_attr = AffineMapAttr.get(affine_map)
    a_val = affine_d.AffineLoadOp(
        load_type,
        func.arguments[0],
        [i_loop.induction_variable, k_loop.induction_variable],
        affine_attr,
        ip=InsertionPoint(i_loop.body),
    )

    # Store to A_fifo[i, 0, k]
    zero_idx = arith_d.ConstantOp(
        IndexType.get(), 0, ip=InsertionPoint.at_block_begin(func.body.blocks[0])
    ).result
    affine_map = AffineMap.get(
        dim_count=3,
        symbol_count=0,
        exprs=[
            AffineExpr.get_dim(0),
            AffineExpr.get_constant(0),
            AffineExpr.get_dim(2),
        ],
    )
    affine_attr = AffineMapAttr.get(affine_map)
    affine_d.AffineStoreOp(
        a_val.result,
        A_fifo.result,
        [i_loop.induction_variable, zero_idx, k_loop.induction_variable],
        affine_attr,
        ip=InsertionPoint(i_loop.body),
    )
    affine_d.AffineYieldOp([], ip=InsertionPoint(i_loop.body))

    # Inner j loop for loading B
    j_map = AffineMap.get(
        dim_count=0, symbol_count=0, exprs=[AffineExpr.get_constant(j_size)]
    )
    j_loop = affine_d.AffineForOp(0, j_map, 1, ip=InsertionPoint(k_loop.body))
    j_loop.attributes["loop_name"] = StringAttr.get("j")
    j_loop.attributes["op_name"] = StringAttr.get("data_load")

    # Load from B and store to B_fifo
    affine_map = AffineMap.get(
        dim_count=2,
        symbol_count=0,
        exprs=[AffineExpr.get_dim(0), AffineExpr.get_dim(1)],
    )
    affine_attr = AffineMapAttr.get(affine_map)
    b_val = affine_d.AffineLoadOp(
        load_type,
        func.arguments[1],
        [k_loop.induction_variable, j_loop.induction_variable],
        affine_attr,
        ip=InsertionPoint(j_loop.body),
    )

    # Store to B_fifo[j, 0, k]
    affine_map = AffineMap.get(
        dim_count=3,
        symbol_count=0,
        exprs=[
            AffineExpr.get_dim(0),
            AffineExpr.get_constant(0),
            AffineExpr.get_dim(2),
        ],
    )
    affine_attr = AffineMapAttr.get(affine_map)
    affine_d.AffineStoreOp(
        b_val.result,
        B_fifo.result,
        [j_loop.induction_variable, zero_idx, k_loop.induction_variable],
        affine_attr,
        ip=InsertionPoint(j_loop.body),
    )
    affine_d.AffineYieldOp([], ip=InsertionPoint(j_loop.body))
    affine_d.AffineYieldOp([], ip=InsertionPoint(k_loop.body))

    ### Create drain loop band

    k_map = AffineMap.get(
        dim_count=0, symbol_count=0, exprs=[AffineExpr.get_constant(k_size)]
    )
    k_loop = affine_d.AffineForOp(
        0, k_map, 1, ip=InsertionPoint.at_block_terminator(func.body.blocks[0])
    )
    k_loop.attributes["loop_name"] = StringAttr.get("k")
    k_loop.attributes["op_name"] = StringAttr.get("data_drain")

    # Inner i loop for draining A
    i_map = AffineMap.get(
        dim_count=0, symbol_count=0, exprs=[AffineExpr.get_constant(i_size)]
    )
    i_loop = affine_d.AffineForOp(0, i_map, 1, ip=InsertionPoint(k_loop.body))
    i_loop.attributes["loop_name"] = StringAttr.get("i")
    k_loop.attributes["op_name"] = StringAttr.get("data_drain")

    # Load from A_fifo[i, 4, k] and store to A_drain[i]
    i_size_idx = arith_d.ConstantOp(
        IndexType.get(),
        i_size,
        ip=InsertionPoint.at_block_begin(func.body.blocks[0]),
    ).result
    j_size_idx = arith_d.ConstantOp(
        IndexType.get(),
        j_size,
        ip=InsertionPoint.at_block_begin(func.body.blocks[0]),
    ).result
    affine_map = AffineMap.get(
        dim_count=3,
        symbol_count=0,
        exprs=[
            AffineExpr.get_dim(0),
            AffineExpr.get_constant(i_size),
            AffineExpr.get_dim(2),
        ],
    )
    affine_attr = AffineMapAttr.get(affine_map)
    a_val = affine_d.AffineLoadOp(
        load_type,
        A_fifo.result,
        [i_loop.induction_variable, i_size_idx, k_loop.induction_variable],
        affine_attr,
        ip=InsertionPoint(i_loop.body),
    )

    affine_map = AffineMap.get(
        dim_count=1, symbol_count=0, exprs=[AffineExpr.get_dim(0)]
    )
    affine_attr = AffineMapAttr.get(affine_map)
    affine_d.AffineStoreOp(
        a_val.result,
        A_drain.result,
        [i_loop.induction_variable],
        affine_attr,
        ip=InsertionPoint(i_loop.body),
    )
    affine_d.AffineYieldOp([], ip=InsertionPoint(i_loop.body))

    # Inner j loop for draining B
    j_map = AffineMap.get(
        dim_count=0, symbol_count=0, exprs=[AffineExpr.get_constant(j_size)]
    )
    j_loop = affine_d.AffineForOp(0, j_map, 1, ip=InsertionPoint(k_loop.body))
    j_loop.attributes["loop_name"] = StringAttr.get("j")
    j_loop.attributes["op_name"] = StringAttr.get("data_drain")

    # Load from B_fifo[j, 4, k] and store to B_drain[j]
    affine_map = AffineMap.get(
        dim_count=3,
        symbol_count=0,
        exprs=[
            AffineExpr.get_dim(0),
            AffineExpr.get_constant(j_size),
            AffineExpr.get_dim(2),
        ],
    )
    affine_attr = AffineMapAttr.get(affine_map)
    b_val = affine_d.AffineLoadOp(
        load_type,
        B_fifo.result,
        [j_loop.induction_variable, j_size_idx, k_loop.induction_variable],
        affine_attr,
        ip=InsertionPoint(j_loop.body),
    )

    affine_map = AffineMap.get(
        dim_count=1, symbol_count=0, exprs=[AffineExpr.get_dim(0)]
    )
    affine_attr = AffineMapAttr.get(affine_map)
    affine_d.AffineStoreOp(
        b_val.result,
        B_drain.result,
        [j_loop.induction_variable],
        affine_attr,
        ip=InsertionPoint(j_loop.body),
    )
    affine_d.AffineYieldOp([], ip=InsertionPoint(j_loop.body))
    affine_d.AffineYieldOp([], ip=InsertionPoint(k_loop.body))

    ### Build func call
    # first get the slice
    iv_i = outer_loop.induction_variable
    iv_j = middle_loop.induction_variable
    # Remove all ops except terminator
    ops = list(middle_loop.body.operations)
    for op in ops[:-1]:
        op.operation.erase()
    idx_one = arith_d.ConstantOp(
        IndexType.get(), 1, ip=InsertionPoint.at_block_terminator(middle_loop.body)
    )
    i_plus_one = arith_d.AddIOp(
        iv_i, idx_one, ip=InsertionPoint.at_block_terminator(middle_loop.body)
    )
    j_plus_one = arith_d.AddIOp(
        iv_j, idx_one, ip=InsertionPoint.at_block_terminator(middle_loop.body)
    )
    result = MLIRType.parse(
        f"memref<{k_size}x{A_fifo.result.type.element_type}, strided<{[1]}, offset: ?>>"
    )
    a_fifo_slice_in = memref_d.SubViewOp(
        source=A_fifo.result,
        result=result,
        static_offsets=[
            ShapedType.get_dynamic_size(),
            ShapedType.get_dynamic_size(),
            0,
        ],
        static_sizes=[1, 1, k_size],
        static_strides=[1] * 3,
        offsets=[iv_i, iv_j],
        sizes=[],
        strides=[],
        ip=InsertionPoint.at_block_terminator(middle_loop.body),
    )
    a_fifo_slice_out = memref_d.SubViewOp(
        source=A_fifo.result,
        result=result,
        static_offsets=[
            ShapedType.get_dynamic_size(),
            ShapedType.get_dynamic_size(),
            0,
        ],
        static_sizes=[1, 1, k_size],
        static_strides=[1] * 3,
        offsets=[iv_i, j_plus_one],
        sizes=[],
        strides=[],
        ip=InsertionPoint.at_block_terminator(middle_loop.body),
    )
    result = MLIRType.parse(
        f"memref<{k_size}x{B_fifo.result.type.element_type}, strided<{[1]}, offset: ?>>"
    )
    b_fifo_slice_in = memref_d.SubViewOp(
        source=B_fifo.result,
        result=result,
        static_offsets=[
            ShapedType.get_dynamic_size(),
            ShapedType.get_dynamic_size(),
            0,
        ],
        static_sizes=[1, 1, k_size],
        static_strides=[1] * 3,
        offsets=[iv_j, iv_i],
        sizes=[],
        strides=[],
        ip=InsertionPoint.at_block_terminator(middle_loop.body),
    )
    b_fifo_slice_out = memref_d.SubViewOp(
        source=B_fifo.result,
        result=result,
        static_offsets=[
            ShapedType.get_dynamic_size(),
            ShapedType.get_dynamic_size(),
            0,
        ],
        static_sizes=[1, 1, k_size],
        static_strides=[1] * 3,
        offsets=[iv_j, i_plus_one],
        sizes=[],
        strides=[],
        ip=InsertionPoint.at_block_terminator(middle_loop.body),
    )

    func_d.CallOp(
        [],
        FlatSymbolRefAttr.get("PE_kernel"),
        [
            a_fifo_slice_in.result,
            b_fifo_slice_in.result,
            a_fifo_slice_out.result,
            b_fifo_slice_out.result,
            func.arguments[2],
            outer_loop.induction_variable,
            middle_loop.induction_variable,
        ],
        ip=InsertionPoint.at_block_terminator(middle_loop.body),
    )
