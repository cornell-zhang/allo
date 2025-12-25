# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, super-init-not-called, too-many-nested-blocks, too-many-branches
# pylint: disable=consider-using-enumerate, no-value-for-parameter, too-many-function-args, redefined-variable-type

import os
from ..backend.llvm import LLVMModule
from .._mlir.ir import (
    Location,
    UnitAttr,
    InsertionPoint,
    Module,
    Context,
    Region,
    RegionSequence,
    Block,
    BlockArgument,
    BlockArgumentList,
    OpView,
    OpResult,
    OpOperandList,
    Operation,
    Value,
    TypeAttr,
    StringAttr,
    AffineMapAttr,
    AffineMap,
    AffineExpr,
    FunctionType,
    MemRefType,
    IntegerType,
    FloatType,
    IndexType,
)
from .._mlir.dialects import (
    allo as allo_d,
    func as func_d,
    memref as memref_d,
    openmp as openmp_d,
    arith as arith_d,
    index as index_d,
    affine as affine_d,
    scf as scf_d,
    llvm as llvm_d,
)
from .._mlir.passmanager import PassManager
from .._mlir.execution_engine import ExecutionEngine
from ..ir.transform import find_func_in_module
from ..passes import decompose_library_function
from ..utils import get_func_inputs_outputs


# The `walk` function
def recursive_collect_ops(
    top_op: Operation, target_op_type: tuple[type], res_list: list
):
    if isinstance(top_op, target_op_type):
        res_list.append(top_op)
    for region in top_op.regions:
        for block in region.blocks:
            for op in block:
                recursive_collect_ops(op, target_op_type, res_list)


# Useful when searching for omp operations after lowering
def recursive_collect_ops_by_name(
    top_op: Operation, target_op_name: str, res_list: list
):
    if top_op.name == target_op_name:
        res_list.append(top_op)
    for region in top_op.regions:
        for block in region.blocks:
            for op in block:
                recursive_collect_ops_by_name(op, target_op_name, res_list)


def build_dataflow_simulator(module: Module, top_func_name: str):
    with module.context, Location.unknown():
        func = find_func_in_module(module, top_func_name)
        assert isinstance(func.body, Region)
        top_func_ops = func.body.blocks[0].operations
        pe_call_define_ops: dict[func_d.CallOp, func_d.FuncOp] = {}
        stream_construct_ops: dict[str, allo_d.StreamConstructOp] = {}
        for op in top_func_ops:
            if isinstance(op, memref_d.AllocOp):
                continue
            if isinstance(op, func_d.CallOp):
                callee_name = str(op.callee)[1:]
                if not callee_name.startswith(("load_buf", "store_res")):
                    for mod_op in module.body.operations:
                        if isinstance(mod_op, func_d.FuncOp):
                            if callee_name == str(mod_op.sym_name).strip('"'):
                                pe_call_define_ops[op] = mod_op
                                break
            elif isinstance(op, allo_d.StreamConstructOp):
                stream_name = str(op.attributes["name"]).strip('"')
                stream_construct_ops[stream_name] = op

        # Construct Memref variables for pipes
        stream_struct_table: dict[str, OpResult] = {}  # stream name: stream struct
        stream_type_table: dict[str, MemRefType] = {}
        int_type = IntegerType.get_signless(32, module.context)
        memref_scalar_int_type = MemRefType.get([], int_type)
        empty_map = AffineMapAttr.get(AffineMap.get(0, 0, []))
        const_0_defined = False
        for stream_access_op in stream_construct_ops.values():
            stream_name = stream_access_op.attributes["name"]
            stream_type = allo_d.StreamType(stream_access_op.result.type)
            stream_item_type = stream_type.base_type
            stream_depth = stream_type.depth
            assert isinstance(stream_item_type, (MemRefType, IntegerType, FloatType))
            assert isinstance(stream_depth, int)
            ip = InsertionPoint(beforeOperation=stream_access_op)
            if isinstance(stream_item_type, MemRefType):
                item_element_type = stream_item_type.element_type
                if not isinstance(item_element_type, (IntegerType, FloatType)):
                    raise NotImplementedError()
                memref_stream_type = MemRefType.get(
                    shape=[stream_depth + 1] + stream_item_type.shape,
                    element_type=item_element_type,
                )
            else:
                memref_stream_type = MemRefType.get(
                    shape=[stream_depth + 1], element_type=stream_item_type
                )
            stream_memref_op = memref_d.AllocOp(memref_stream_type, [], [], ip=ip)
            stream_head_op = memref_d.AllocOp(memref_scalar_int_type, [], [], ip=ip)
            stream_tail_op = memref_d.AllocOp(memref_scalar_int_type, [], [], ip=ip)
            if not const_0_defined:
                const_zero = arith_d.ConstantOp(int_type, 0, ip=ip)
                const_0_defined = True
            memref_d.StoreOp(value=const_zero, memref=stream_head_op, indices=[], ip=ip)
            memref_d.StoreOp(value=const_zero, memref=stream_tail_op, indices=[], ip=ip)
            fifo_struct_type = allo_d.StructType.get(
                members=[
                    memref_stream_type,
                    memref_scalar_int_type,
                    memref_scalar_int_type,
                ],
                context=func.context,
            )
            fifo_struct_op = allo_d.StructConstructOp(
                output=fifo_struct_type,
                input=[stream_memref_op, stream_head_op, stream_tail_op],
                ip=ip,
            )
            fifo_struct_memref_type = MemRefType.get([], fifo_struct_type)
            stream_memref_op = memref_d.AllocOp(fifo_struct_memref_type, [], [], ip=ip)
            stream_memref_op.attributes["name"] = stream_name
            affine_d.AffineStoreOp(
                value=fifo_struct_op,
                memref=stream_memref_op,
                indices=[],
                map=empty_map,
                ip=ip,
            )
            stream_name_str = str(stream_name).strip('"')
            stream_head_op.attributes["name"] = StringAttr.get(
                f"{stream_name_str}_head"
            )
            stream_tail_op.attributes["name"] = StringAttr.get(
                f"{stream_name_str}_tail"
            )
            stream_memref_op.attributes["name"] = stream_name
            stream_struct_table[stream_name_str] = stream_memref_op.result
            stream_type_table[stream_name_str] = memref_stream_type

        # Transfrom the stream operations in function calls
        for call_op, func_def_op in pe_call_define_ops.items():
            # Get the correspondence between arguments and passed pipes
            arg_stream_table: dict[BlockArgument, str] = {}  # arg: stream name
            assert isinstance(call_op.operands_, OpOperandList)
            assert isinstance(func_def_op.arguments, BlockArgumentList)
            assert len(call_op.operands_) == len(func_def_op.arguments)
            for i in range(len(call_op.operands_)):
                arg_instance = call_op.operands_[i]
                for stream_name, stream_construct_op in stream_construct_ops.items():
                    if Value(stream_construct_op.result) == arg_instance:
                        arg_def = func_def_op.arguments[i]
                        arg_stream_table[arg_def] = stream_name
            # Collect and replace `stream_get`s and `stream_put`s
            func_stream_ops = []
            recursive_collect_ops(
                func_def_op, (allo_d.StreamGetOp, allo_d.StreamPutOp), func_stream_ops
            )
            for stream_access_op in func_stream_ops:
                assert isinstance(
                    stream_access_op, (allo_d.StreamGetOp, allo_d.StreamPutOp)
                )
                replace_ip = InsertionPoint(beforeOperation=stream_access_op)
                # Have to leverage weak typing here
                stream = stream_access_op.stream
                stream_arg = BlockArgument(stream)
                stream_name = arg_stream_table[stream_arg]
                stream_type = stream_type_table[stream_name]
                stream_memref = stream_struct_table[stream_name]
                # Change argument definitions
                stream_arg.set_type(stream_memref.type)
                old_func_type = func_def_op.type
                new_inputs = old_func_type.inputs.copy()
                new_inputs[stream_arg.arg_number] = stream_memref.type
                new_func_type = FunctionType.get(
                    inputs=new_inputs,
                    results=old_func_type.results,
                    context=old_func_type.context,
                )
                func_def_op.attributes["function_type"] = TypeAttr.get(
                    new_func_type, module.context
                )
                call_op.operands_[stream_arg.arg_number] = stream_memref
                # FIFO access
                # Spin and wait for the FIFO to be not full
                assert isinstance(stream_memref.type, MemRefType)
                stream_struct = affine_d.AffineLoadOp(
                    result=stream_memref.type.element_type,
                    memref=stream_arg,
                    indices=[],
                    map=empty_map,
                    ip=replace_ip,
                )
                head_ptr = allo_d.StructGetOp(
                    output=memref_scalar_int_type,
                    input=stream_struct,
                    index=1,
                    ip=replace_ip,
                )
                tail_ptr = allo_d.StructGetOp(
                    output=memref_scalar_int_type,
                    input=stream_struct,
                    index=2,
                    ip=replace_ip,
                )
                fifo_ptr = allo_d.StructGetOp(
                    output=stream_type, input=stream_struct, index=0, ip=replace_ip
                )
                const_one = arith_d.ConstantOp(int_type, 1, ip=replace_ip)
                const_fifo_depth = arith_d.ConstantOp(
                    int_type, stream_type.get_dim_size(0), ip=replace_ip
                )
                if isinstance(stream_access_op, allo_d.StreamPutOp):
                    tail_val_op = memref_d.LoadOp(
                        memref=tail_ptr, indices=[], ip=replace_ip
                    )
                    tail_inc_op = arith_d.AddIOp(
                        lhs=tail_val_op.result, rhs=const_one.result, ip=replace_ip
                    )
                    tail_next_op = arith_d.RemUIOp(
                        lhs=tail_inc_op.result,
                        rhs=const_fifo_depth.result,
                        ip=replace_ip,
                    )
                else:
                    assert isinstance(stream_access_op, allo_d.StreamGetOp)
                    head_val_op = memref_d.LoadOp(
                        memref=head_ptr, indices=[], ip=replace_ip
                    )
                    head_inc_op = arith_d.AddIOp(
                        lhs=head_val_op.result, rhs=const_one.result, ip=replace_ip
                    )
                    head_next_op = arith_d.RemUIOp(
                        lhs=head_inc_op.result,
                        rhs=const_fifo_depth.result,
                        ip=replace_ip,
                    )
                spin_while_op = scf_d.WhileOp(results_=[], inits=[], ip=replace_ip)
                assert isinstance(spin_while_op.before, Region)
                assert isinstance(spin_while_op.after, Region)
                before_block = Block.create_at_start(
                    parent=spin_while_op.before, arg_types=[]
                )
                before_ip = InsertionPoint(before_block)
                openmp_d.FlushOp([], ip=before_ip)
                after_block = Block.create_at_start(
                    parent=spin_while_op.after, arg_types=[]
                )
                after_ip = InsertionPoint(after_block)
                openmp_d.TaskyieldOp(ip=after_ip)
                scf_d.YieldOp(results_=[], ip=after_ip)
                if isinstance(stream_access_op, allo_d.StreamPutOp):
                    head_val_op = memref_d.LoadOp(
                        memref=head_ptr, indices=[], ip=before_ip
                    )
                    cmp_op = arith_d.CmpIOp(
                        predicate=0, lhs=head_val_op, rhs=tail_next_op, ip=before_ip
                    )
                    scf_d.ConditionOp(condition=cmp_op, args=[], ip=before_ip)
                    data = stream_access_op.data
                    assert isinstance(data, Value)  # Vector or scalar
                    tail_index_op = index_d.CastUOp(
                        output=IndexType.get(module.context),
                        input=tail_val_op,
                        ip=replace_ip,
                    )
                    if isinstance(data.type, MemRefType):  # Vector
                        # Data is an `alloc` pointer and should be loaded first
                        element_type = data.type.element_type
                        if not isinstance(element_type, (IntegerType, FloatType)):
                            # May get StructType involved in the future
                            raise NotImplementedError()
                        rank = data.type.rank
                        for_ip = replace_ip
                        for_induction_vars = []
                        for_ips: list[InsertionPoint] = (
                            []
                        )  # Reserved to insert affine.yield ops later
                        for i in range(rank):
                            dim_size = data.type.get_dim_size(i)
                            for_loop_op = affine_d.AffineForOp(0, dim_size, ip=for_ip)
                            for_induction_vars.append(for_loop_op.induction_variable)
                            for_ip = InsertionPoint(for_loop_op.body)
                            for_ips.append(for_ip)
                        element_dim_map = AffineMap.get(
                            dim_count=rank,
                            symbol_count=0,
                            exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                            context=module.context,
                        )
                        element_load_op = affine_d.AffineLoadOp(
                            result=element_type,
                            memref=data,
                            indices=for_induction_vars,
                            map=AffineMapAttr.get(element_dim_map),
                            ip=for_ip,
                        )  # Fetch the element
                        memref_d.StoreOp(
                            value=element_load_op,
                            memref=fifo_ptr,
                            indices=[tail_index_op] + for_induction_vars,
                            ip=for_ip,
                        )  # Put the element to the stream
                        for ip in for_ips:
                            affine_d.AffineYieldOp([], ip=ip)
                    else:  # Scalar
                        # Ensure data type matches the memref element type
                        fifo_element_type = stream_type.element_type
                        store_value = data
                        if data.type != fifo_element_type:
                            # Cast the data to match the expected element type
                            if isinstance(data.type, IntegerType) and isinstance(
                                fifo_element_type, IntegerType
                            ):
                                if data.type.width > fifo_element_type.width:
                                    store_value = arith_d.TruncIOp(
                                        fifo_element_type, data, ip=replace_ip
                                    )
                                elif data.type.width < fifo_element_type.width:
                                    if data.type.is_signed:
                                        store_value = arith_d.ExtSIOp(
                                            fifo_element_type, data, ip=replace_ip
                                        )
                                    else:
                                        store_value = arith_d.ExtUIOp(
                                            fifo_element_type, data, ip=replace_ip
                                        )
                        memref_d.StoreOp(
                            value=store_value,
                            memref=fifo_ptr,
                            indices=[tail_index_op],
                            ip=replace_ip,
                        )
                    # Atomic update of tail
                    critical_op = openmp_d.CriticalOp(ip=replace_ip)
                    critical_ip = InsertionPoint(
                        Block.create_at_start(critical_op.region)
                    )
                    memref_d.StoreOp(tail_next_op, tail_ptr, [], ip=critical_ip)
                    openmp_d.TerminatorOp(ip=critical_ip)
                else:
                    assert isinstance(stream_access_op, allo_d.StreamGetOp)
                    tail_val_op = memref_d.LoadOp(
                        memref=tail_ptr, indices=[], ip=before_ip
                    )
                    cmp_op = arith_d.CmpIOp(
                        0, lhs=head_val_op, rhs=tail_val_op, ip=before_ip
                    )
                    scf_d.ConditionOp(condition=cmp_op, args=[], ip=before_ip)
                    orig_got_val = stream_access_op.res
                    assert isinstance(orig_got_val, OpResult)
                    head_index_op = index_d.CastUOp(
                        output=IndexType.get(module.context),
                        input=head_val_op,
                        ip=replace_ip,
                    )
                    if isinstance(orig_got_val.type, MemRefType):
                        element_type = orig_got_val.type.element_type
                        if not isinstance(element_type, (IntegerType, FloatType)):
                            raise NotImplementedError()
                        rank = orig_got_val.type.rank
                        assert rank > 0
                        # Create a memref for the loaded element
                        element_alloc_op = memref_d.AllocOp(
                            memref=orig_got_val.type,
                            dynamicSizes=[],
                            symbolOperands=[],
                            ip=replace_ip,
                        )
                        orig_got_val.replace_all_uses_with(element_alloc_op.result)
                        # Create the element load/store loop
                        for_ip = replace_ip
                        for_induction_vars = []
                        for_ips: list[InsertionPoint] = []
                        for i in range(rank):
                            for_loop_op = affine_d.AffineForOp(
                                0,
                                orig_got_val.type.get_dim_size(i),
                                ip=for_ip,
                            )
                            for_induction_vars.append(for_loop_op.induction_variable)
                            for_ip = InsertionPoint(for_loop_op.body)
                            for_ips.append(for_ip)
                        element_dim_map = AffineMap.get(
                            dim_count=rank,
                            symbol_count=0,
                            exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                            context=module.context,
                        )
                        element_load_op = memref_d.LoadOp(
                            memref=fifo_ptr,
                            indices=[head_index_op] + for_induction_vars,
                            ip=for_ip,  # The innermost Loop body
                        )
                        affine_d.AffineStoreOp(
                            value=element_load_op,
                            memref=element_alloc_op,
                            indices=for_induction_vars,
                            map=AffineMapAttr.get(element_dim_map),
                            ip=for_ip,
                        )
                        for ip in for_ips:
                            affine_d.AffineYieldOp([], ip=ip)
                    else:  # Scalar
                        new_get_op = memref_d.LoadOp(
                            memref=fifo_ptr, indices=[head_index_op], ip=replace_ip
                        )
                        # Ensure loaded type matches the expected result type
                        loaded_value = new_get_op.result
                        expected_type = orig_got_val.type
                        if loaded_value.type != expected_type:
                            # Cast the loaded value to match the expected type
                            if isinstance(
                                loaded_value.type, IntegerType
                            ) and isinstance(expected_type, IntegerType):
                                if loaded_value.type.width < expected_type.width:
                                    if loaded_value.type.is_signed:
                                        loaded_value = arith_d.ExtSIOp(
                                            expected_type, loaded_value, ip=replace_ip
                                        )
                                    else:
                                        loaded_value = arith_d.ExtUIOp(
                                            expected_type, loaded_value, ip=replace_ip
                                        )
                                elif loaded_value.type.width > expected_type.width:
                                    loaded_value = arith_d.TruncIOp(
                                        expected_type, loaded_value, ip=replace_ip
                                    )
                        orig_got_val.replace_all_uses_with(loaded_value)
                    critical_op = openmp_d.CriticalOp(ip=replace_ip)
                    critical_ip = InsertionPoint(
                        Block.create_at_start(critical_op.region)
                    )
                    memref_d.StoreOp(head_next_op, head_ptr, [], ip=critical_ip)
                    openmp_d.TerminatorOp(ip=critical_ip)
                stream_access_op.operation.erase()

        for op in stream_construct_ops.values():
            op.operation.erase()

        # Add the outmost `omp.parallel`
        assert len(pe_call_define_ops) > 0
        omp_ip = InsertionPoint(beforeOperation=list(pe_call_define_ops.keys())[0])
        omp_parallel_op = openmp_d.ParallelOp([], [], [], [], ip=omp_ip)
        assert isinstance(omp_parallel_op.region, Region)
        omp_parallel_block = Block.create_at_start(omp_parallel_op.region, [])

        # Add `omp.sections`
        ip_omp_parallel = InsertionPoint(omp_parallel_block)
        omp_sections_op = openmp_d.SectionsOp([], [], [], [], ip=ip_omp_parallel)
        omp_sections_block = Block.create_at_start(omp_sections_op.region, [])
        openmp_d.TerminatorOp(ip=ip_omp_parallel)

        # Add `omp.section`s for PE calls
        ip_omp_sections = InsertionPoint(omp_sections_block)
        for call_op in pe_call_define_ops:
            assert isinstance(call_op, OpView)
            omp_section_op = openmp_d.SectionOp(ip=ip_omp_sections)
            omp_section_block = Block.create_at_start(omp_section_op.region, [])
            ip_omp_section = InsertionPoint(omp_section_block)
            omp_term_op = openmp_d.TerminatorOp(ip=ip_omp_section)
            call_op.operation.move_before(omp_term_op.operation)
        openmp_d.TerminatorOp(ip=ip_omp_sections)


# This pass is only meant to run on fully lowered MLIR code
# Note: OpenMP operations in lowered IR are not the original operation types anymore
def convert_critical_write_to_atomic_write(module: Module):
    with module.context, Location.unknown():
        omp_critical_ops = []
        for op in module.body:
            if not isinstance(op, llvm_d.LLVMFuncOp):
                continue
            recursive_collect_ops_by_name(op, "omp.critical", omp_critical_ops)
        for critical_op in omp_critical_ops:
            # Transform a critical area with only the store op and omp.terminator
            assert isinstance(critical_op.regions, RegionSequence)
            if len(critical_op.regions) != 1:
                continue
            region = critical_op.regions[0]
            if len(region.blocks) != 1:
                continue
            block = region.blocks[0]
            if len(block.operations) != 2:
                continue
            if (
                not isinstance(block.operations[0], llvm_d.StoreOp)
                or block.operations[1].name != "omp.terminator"
            ):
                continue
            store_op = block.operations[0]
            assert isinstance(store_op, llvm_d.StoreOp)
            store_ip = InsertionPoint(critical_op)
            openmp_d.AtomicWriteOp(x=store_op.addr, expr=store_op.value, ip=store_ip)
            critical_op.operation.erase()


class LLVMOMPModule(LLVMModule):
    def __init__(self, mod: Module, top_func_name: str, ext_libs=None):
        with Context() as ctx:
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.top_func_name = top_func_name
            func = find_func_in_module(self.module, top_func_name)
            ext_libs = [] if ext_libs is None else ext_libs
            # Get input/output types
            self.in_types, self.out_types = get_func_inputs_outputs(func)
            self.module = decompose_library_function(self.module)

            build_dataflow_simulator(self.module, self.top_func_name)
            # Attach necessary attributes
            func = find_func_in_module(self.module, top_func_name)
            if func is None:
                raise RuntimeError(
                    "No top-level function found in the built MLIR module"
                )
            func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            func.attributes["top"] = UnitAttr.get()

            # Start lowering
            # Lower linalg for AIE
            pm = PassManager.parse(
                "builtin.module("
                "one-shot-bufferize,"
                "expand-strided-metadata,"
                "func.func(convert-linalg-to-affine-loops)"
                ")"
            )
            pm.run(self.module.operation)
            # Lower StructType
            allo_d.lower_composite_type(self.module)
            # Lower bit ops
            allo_d.lower_bit_ops(self.module)
            # Reference: https://discourse.llvm.org/t/help-lowering-affine-loop-to-openmp/72441/9
            pm = PassManager.parse(
                "builtin.module("
                "lower-affine,"
                "convert-scf-to-cf,"
                "finalize-memref-to-llvm,"
                "convert-func-to-llvm,"
                "convert-index-to-llvm,"
                "convert-cf-to-llvm,"
                "convert-openmp-to-llvm,"
                "canonicalize"
                ")"
            )
            pm.run(self.module.operation)
            convert_critical_write_to_atomic_write(self.module)

            assert os.getenv("LLVM_BUILD_DIR") is not None, "LLVM_BUILD_DIR is not set"
            shared_libs = [
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_runner_utils.so"
                ),
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_c_runner_utils.so"
                ),
                os.path.join(os.getenv("LLVM_BUILD_DIR"), "lib", "libomp.so"),
            ]
            shared_libs += [lib.compile_shared_lib() for lib in ext_libs]
            self.execution_engine = ExecutionEngine(
                self.module, opt_level=2, shared_libs=shared_libs
            )
