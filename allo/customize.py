# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, too-many-nested-blocks, too-many-instance-attributes

import re
import inspect
import textwrap
import copy
from dataclasses import dataclass
from functools import wraps
from types import FunctionType as PyFunctionType
from typing import Union
from collections.abc import Callable

from ._mlir.ir import (
    Context,
    Location,
    InsertionPoint,
    StringAttr,
    UnitAttr,
    IndexType,
    IntegerType,
    IntegerAttr,
    TypeAttr,
    F32Type,
    MemRefType,
    FlatSymbolRefAttr,
    FunctionType,
    AffineMap,
    AffineMapAttr,
    BlockArgument,
)
from ._mlir.dialects import (
    allo as allo_d,
    memref as memref_d,
    affine as affine_d,
    scf as scf_d,
    arith as arith_d,
    func as func_d,
)
from ._mlir.dialects.affine import (
    AffineExpr,
    AffineDimExpr,
)
from ._mlir.exceptions import (
    AlloValueError,
)

from . import primitives as prim
from .ir.visitor import ASTContext
from .ir.utils import MockArg, MockBuffer, parse_ast, get_global_vars
from .ir.builder import ASTTransformer
from .ir.infer import TypeInferer
from .ir.transform import (
    get_affine_loop_nests,
    find_loop_in_bands,
    find_buffer,
    find_func_in_module,
    LoopWrapper,
)
from .passes import (
    _mlir_lower_pipeline,
    lower_linalg_and_attach_names,
    analyze_use_def,
)
from .utils import freeze_list
from .backend.llvm import LLVMModule
from .backend.hls import HLSModule
from .backend.xls import XLSCCModule
from .library import KERNEL2SCHEDULE
from .library.systolic import check_systolic, prepare_systolic


def getsourcefile(obj):
    ret = inspect.getsourcefile(obj)
    if ret is None:
        ret = inspect.getfile(obj)
    return ret


def wrapped_apply(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        sch = args[0]
        with sch.module.context, Location.unknown():
            res = fn(*args, **kwargs)
        _mlir_lower_pipeline(sch.module)
        # Update top function in the current context
        for op in sch.module.body.operations:
            if isinstance(op, func_d.FuncOp) and op.name.value == sch.top_func_name:
                sch.top_func = op
                break
        else:
            raise RuntimeError("Top function not found")
        # Update insertion point
        sch.ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        # Record primitive sequences
        if fn.__name__ != "compose":
            sch.primitive_sequences.append((fn.__name__, list(args[1:]), kwargs))
        return res

    return wrapper


@dataclass
class Partition:
    Complete = 0
    Block = 1
    Cyclic = 2


class Schedule:
    def __init__(
        self,
        module,
        top_func,
        func_args,
        ip,
        ext_libs=None,
        inst_list=None,
        func_instances=None,
    ):
        self.module = module
        self.top_func = top_func
        self.top_func_name = top_func.name.value
        # func_args are dtensors
        self.func_args = func_args
        self.ip = ip
        self.primitive_sequences = []
        if ext_libs is None:
            ext_libs = []
        self.ext_libs = ext_libs
        self.partitioned_arrays = {}
        self.inst_list = inst_list if inst_list is not None else []
        if func_args:
            for func_name, _ in func_args.items():
                if func_name not in self.func_args:
                    self.func_args[func_name] = []
        self.func_instances = func_instances
        self.systolic = check_systolic(self)

    def get_loops(self, func=None):
        if isinstance(func, str):
            func = self._find_function(func)
        if func is None:
            func = self.top_func
        return get_affine_loop_nests(func)

    def _find_band(self, band_name, func=None):
        loops = self.get_loops(func)
        if band_name in loops.loops:
            return loops[band_name]
        raise RuntimeError(f"Band {band_name} not found")

    def _find_function(self, name, error=True):
        for func in self.module.body.operations:
            if isinstance(func, func_d.FuncOp) and func.name.value == name:
                return func
        if error:
            raise RuntimeError(f"Function {name} not found")
        return None

    def _get_func_and_axis(self, axis):
        if isinstance(axis, LoopWrapper):
            func = self._find_function(axis.func)
            return func, axis
        if ":" in axis:
            func_name, axis = axis.split(":")
        else:
            func_name = self.top_func_name
        func = self._find_function(func_name)
        return func, axis

    @wrapped_apply
    def split(self, axis, factor):
        """
        `split` will find the loop with loop index `axis` and tile it with each tile size `factor`
        The new inner loop will be named `axis.inner` and the outer loop will be named `axis.outer`

        Parameters
        ----------
        axis: str
            The name of an index in the kernel.

        factor: int
            The size of each tile, e.g. the size of the inner nested loop.
        """
        func, axis = self._get_func_and_axis(axis)
        band_name, axis = find_loop_in_bands(func, axis)
        ip = InsertionPoint.at_block_terminator(func.entry_block)
        op_hdl = allo_d.CreateOpHandleOp(band_name, ip=ip)
        loop_hdl = allo_d.CreateLoopHandleOp(op_hdl.result, StringAttr.get(axis), ip=ip)
        i32 = IntegerType.get_unsigned(32)
        factor = IntegerAttr.get(i32, factor)
        allo_d.SplitOp(loop_hdl.result, factor, ip=ip)

    @wrapped_apply
    def reorder(self, *args):
        """
        Reorders nested loops with indices listed in `args` such that the outermost loop is the first
        index listed in `args`, the second is the second outermost, and so on.

        This function is vardic, accepting each index as a separate argument.
        """
        func, axis = self._get_func_and_axis(args[0])
        band_name, _ = find_loop_in_bands(func, axis)
        ip = InsertionPoint.at_block_terminator(func.entry_block)
        op_hdl = allo_d.CreateOpHandleOp(band_name, ip=ip)
        loop_hdls = []
        for arg in args:
            func, axis = self._get_func_and_axis(arg)
            band_name, axis = find_loop_in_bands(func, axis)
            loop_hdls.append(
                allo_d.CreateLoopHandleOp(op_hdl.result, StringAttr.get(axis), ip=ip)
            )
        arg_results = [arg.result for arg in loop_hdls]
        allo_d.ReorderOp(arg_results, ip=ip)

    @wrapped_apply
    def unroll(self, axis, factor=0):
        """
        Unrolls a loop with loop index `axis` by `factor`.

        Parameters
        ----------
        axis: str
            The name of an index in the kernel.

        factor: int
            The factor to unroll by, for example a factor of 2 will cause the body to be duplicated once.
        """

        func, axis = self._get_func_and_axis(axis)
        band_name, axis = find_loop_in_bands(func, axis)
        ip = InsertionPoint.at_block_terminator(func.entry_block)
        op_hdl = allo_d.CreateOpHandleOp(band_name, ip=ip)
        loop_hdl = allo_d.CreateLoopHandleOp(op_hdl.result, StringAttr.get(axis), ip=ip)
        i32 = IntegerType.get_unsigned(32)
        factor = IntegerAttr.get(i32, factor)
        allo_d.UnrollOp(loop_hdl.result, factor=factor, ip=ip)

    @wrapped_apply
    def fuse(self, *args):
        """
        Combines loops with indices listed in `args` into a single loop over a single index.

        This function is vardic, accepting each index as a separate argument.
        """
        func, axis = self._get_func_and_axis(args[0])
        band_name, _ = find_loop_in_bands(func, args[0])
        ip = InsertionPoint.at_block_terminator(func.entry_block)
        op_hdl = allo_d.CreateOpHandleOp(band_name, ip=ip)
        loop_hdls = []
        axes = []
        for arg in args:
            func, axis = self._get_func_and_axis(args)
            band_name, axis = find_loop_in_bands(func, arg)
            loop_hdls.append(
                allo_d.CreateLoopHandleOp(op_hdl.result, StringAttr.get(axis), ip=ip)
            )
            axes.append(axis)
        arg_results = [arg.result for arg in loop_hdls]
        allo_d.FuseOp(arg_results, ip=ip)
        if isinstance(args[0], LoopWrapper):
            name = "_".join(axes) + "_fused"
            return LoopWrapper(f"{args[0].func}:{band_name}.{name}", None)
        return LoopWrapper(f"{func.name.value}:{band_name}", None)

    @wrapped_apply
    def partition(self, target, partition_type=Partition.Complete, dim=0, factor=0):
        """
        Partitions a given array and propagates to all callers and callees.

        Parameters
        ----------
        target: allo.ir.utils.MockBuffer | str
            The array to partition.
        partition_type: allo.customize.Partition
            Complete, Block, or Cyclic partition type.
        factor: int
            The number of arrays created by a block or cyclic partition.
        dim: int
            The dimension to partition. If dim=0, all dimensions are partitioned.
        """
        # Validate inputs
        if partition_type not in (
            Partition.Complete,
            Partition.Block,
            Partition.Cyclic,
        ):
            raise AlloValueError("Invalid partition type")
        if dim < 0:
            raise AlloValueError("Invalid dimension")
        if factor < 0:
            raise AlloValueError("Invalid factor")

        # Convert partition type to integer
        partition_type_int = {
            Partition.Complete: 0,
            Partition.Block: 1,
            Partition.Cyclic: 2,
        }[partition_type]

        # Normalize target to MockBuffer
        if isinstance(target, str):
            func_name, buf_name = target.split(":")
            target = MockBuffer(func_name, buf_name)

        # Check for duplicate partitioning
        target_key = f"{target.func}:{target.name}"
        if target_key in self.partitioned_arrays:
            for item in self.partitioned_arrays[target_key]:
                if item[0] == Partition.Complete and item[1] == 0:
                    return  # Already completely partitioned
                raise AlloValueError(
                    f"Cannot partition the same array twice: {target_key}"
                )

        # Collect all buffers that need partitioning via propagation
        buffers_to_partition = self._collect_partition_targets(target)

        # Apply partitioning to all collected buffers
        i32 = IntegerType.get_signless(32)
        ui32 = IntegerType.get_unsigned(32)

        for buf in buffers_to_partition:
            buf_key = f"{buf.func}:{buf.name}"
            func, _, mlir_target = find_buffer(self.module, buf, self.func_args)

            # Record partition info
            if buf_key not in self.partitioned_arrays:
                self.partitioned_arrays[buf_key] = []
            self.partitioned_arrays[buf_key].append((partition_type_int, dim, factor))

            # Create partition operation
            allo_d.PartitionOp(
                mlir_target.result,
                partition_kind=IntegerAttr.get(i32, partition_type_int),
                dim=IntegerAttr.get(ui32, dim),
                factor=IntegerAttr.get(ui32, factor),
                ip=InsertionPoint.at_block_terminator(func.entry_block),
            )

            # Update types for function calls
            if isinstance(mlir_target, func_d.CallOp):
                self._update_call_types(mlir_target, partition_type_int, dim, factor)

            # If this buffer is returned by a function, update the function's return type
            # and all call sites to that function
            self._propagate_return_type_to_callers(
                func, mlir_target, partition_type_int, dim, factor
            )

        # Update global memory references
        self._update_global_types(buffers_to_partition, partition_type_int, dim, factor)

    def _propagate_return_type_to_callers(
        self, func, mlir_target, partition_type, dim, factor
    ):
        """If a partitioned buffer is returned, update function signature and all call sites."""
        if not isinstance(func, func_d.FuncOp):
            return

        # Check if this buffer is returned by the function
        for op in func.entry_block.operations:
            if isinstance(op, func_d.ReturnOp) and op.operands:
                returned_value = op.operands[0]

                # Check if the partitioned buffer is what's being returned
                if (
                    hasattr(mlir_target, "result")
                    and returned_value == mlir_target.result
                ):

                    # Compute new return type with partition layout
                    shape = mlir_target.result.type.shape
                    layout_attr = self._compute_partition_layout(
                        shape, partition_type, dim, factor
                    )

                    old_type = func.type.results[0]
                    new_return_type = MemRefType.get(
                        old_type.shape,
                        old_type.element_type,
                        layout_attr,
                        old_type.memory_space,
                    )

                    # Update function signature
                    new_func_type = FunctionType.get(
                        list(func.type.inputs), [new_return_type]
                    )
                    func.attributes["function_type"] = TypeAttr.get(new_func_type)

                    # Update ALL call sites to this function
                    func_name = func.attributes["sym_name"].value
                    self._update_all_call_sites(func_name, new_return_type)

                break

    def _collect_partition_targets(self, target):
        """Collect all buffers that need partitioning by traversing call graph."""
        visited = set()
        to_partition = []
        worklist = [target]

        while worklist:
            buf = worklist.pop()
            buf_key = f"{buf.func}:{buf.name}"

            if buf_key in visited:
                continue
            visited.add(buf_key)
            to_partition.append(buf)

            _, _, mlir_target = find_buffer(self.module, buf, self.func_args)

            # Add equivalent variables (aliases)
            for equiv in self._get_equivalent_buffers(buf):
                if f"{equiv.func}:{equiv.name}" not in visited:
                    worklist.append(equiv)

            # Propagate through function calls (callees)
            worklist.extend(self._get_callee_buffers(mlir_target))

            # Propagate to callers
            worklist.extend(self._get_caller_buffers(mlir_target))

        return to_partition

    def _get_equivalent_buffers(self, buf):
        """Get all buffers that are aliases of the given buffer."""
        result = []
        arg_names = [
            dtensor.name if hasattr(dtensor, "name") else dtensor
            for dtensor in self.func_args.get(buf.func, [])
        ]

        # Convert to argument index if it's a function argument
        if buf.name in arg_names:
            idx = arg_names.index(buf.name)
            lookup_key = f"{buf.func}:{idx}"
        else:
            lookup_key = f"{buf.func}:{buf.name}"

        for equiv_key in self.get_equivalent_variables(lookup_key):
            path, name = equiv_key.split(":")
            if name.isdigit():
                # Convert argument index back to name
                arg = self.func_args[path][int(name)]
                name = arg.name if hasattr(arg, "name") else arg
            result.append(MockBuffer(path, name))

        return result

    def _get_callee_buffers(self, mlir_target):
        """Get buffers in called functions that correspond to this buffer."""
        result = []

        # If this is a call result, partition the returned buffer in callee
        if isinstance(mlir_target, func_d.CallOp):
            callee_name = FlatSymbolRefAttr(mlir_target.attributes["callee"]).value
            callee_func = self._find_function(callee_name, error=False)

            if callee_func:
                # Find the returned buffer
                for op in callee_func.entry_block.operations:
                    if isinstance(op, func_d.ReturnOp) and op.operands:
                        returned = op.operands[0]
                        if hasattr(returned, "owner") and returned.owner:
                            owner = returned.owner
                            op_name = getattr(
                                owner,
                                "name",
                                getattr(getattr(owner, "operation", None), "name", ""),
                            )
                            if op_name == "memref.alloc" and "name" in owner.attributes:
                                buf_name = StringAttr(owner.attributes["name"]).value
                                result.append(MockBuffer(callee_name, buf_name))
                        break

        # If value is passed to a call, partition corresponding parameter
        if hasattr(mlir_target, "result") and mlir_target.result:
            for use in mlir_target.result.uses:
                if isinstance(use.owner, func_d.CallOp):
                    call_op = use.owner
                    callee_name = FlatSymbolRefAttr(call_op.attributes["callee"]).value

                    for i, operand in enumerate(call_op.operands):
                        if operand == mlir_target.result:
                            param = self._get_param_buffer(callee_name, i)
                            if param:
                                result.append(param)
                            break

        # Handle BlockArgument (function parameter) passed to calls
        if isinstance(mlir_target, BlockArgument):
            for use in mlir_target.uses:
                if isinstance(use.owner, func_d.CallOp):
                    call_op = use.owner
                    callee_name = FlatSymbolRefAttr(call_op.attributes["callee"]).value

                    for i, operand in enumerate(call_op.operands):
                        if operand == mlir_target:
                            param = self._get_param_buffer(callee_name, i)
                            if param:
                                result.append(param)
                            break

        return result

    def _get_caller_buffers(self, mlir_target):
        """Get buffers in calling functions that correspond to this buffer."""
        result = []

        # Find calls to the same function and propagate
        if isinstance(mlir_target, func_d.CallOp):
            callee_attr = mlir_target.attributes["callee"]
            for func in self.module.body.operations:
                if isinstance(func, func_d.FuncOp):
                    for op in func.entry_block.operations:
                        if (
                            isinstance(op, func_d.CallOp)
                            and op.attributes["callee"] == callee_attr
                            and op != mlir_target
                            and "name" in op.attributes
                        ):
                            func_name = func.attributes["sym_name"].value
                            call_name = op.attributes["name"].value
                            result.append(MockBuffer(func_name, call_name))

        return result

    def _get_param_buffer(self, func_name, param_idx):
        """Get MockBuffer for a function parameter by index."""
        args = self.func_args.get(func_name, [])
        if param_idx < len(args):
            arg = args[param_idx]
            name = arg.name if hasattr(arg, "name") else arg
            return MockBuffer(func_name, name)
        return None

    def _compute_partition_layout(self, shape, partition_type, dim, factor):
        """Compute the affine map for partitioned memory layout."""
        partition_idx = []
        address_idx = []

        for i, size in enumerate(shape):
            applies_to_dim = (dim == 0) or (i == dim - 1)

            if applies_to_dim:
                if partition_type == Partition.Cyclic:
                    partition_idx.append(AffineDimExpr.get(i) % factor)
                    address_idx.append(
                        AffineExpr.get_floor_div(AffineDimExpr.get(i), factor)
                    )
                elif partition_type == Partition.Block:
                    block_size = (size + factor - 1) // factor
                    partition_idx.append(
                        AffineExpr.get_floor_div(AffineDimExpr.get(i), block_size)
                    )
                    address_idx.append(AffineDimExpr.get(i) % block_size)
                else:  # Complete
                    partition_idx.append(AffineDimExpr.get(i))
                    address_idx.append(AffineExpr.get_constant(0))
            else:
                partition_idx.append(AffineExpr.get_constant(0))
                address_idx.append(AffineDimExpr.get(i))

        affine_map = AffineMap.get(
            dim_count=len(shape), symbol_count=0, exprs=partition_idx + address_idx
        )
        return AffineMapAttr.get(affine_map)

    def _update_call_types(self, call_op, partition_type, dim, factor):
        """Update function signature types after partitioning a call result."""
        callee_name = FlatSymbolRefAttr(call_op.attributes["callee"]).value
        callee_func = self._find_function(callee_name, error=False)
        if not callee_func:
            return

        shape = call_op.result.type.shape
        layout_attr = self._compute_partition_layout(shape, partition_type, dim, factor)

        # Create new return type with partition layout
        old_type = callee_func.type.results[0]
        new_return_type = MemRefType.get(
            old_type.shape, old_type.element_type, layout_attr, old_type.memory_space
        )

        # Update input types if operands are partitioned
        new_input_types = list(callee_func.type.inputs)
        for i, operand in enumerate(call_op.operands):
            if hasattr(operand.type, "layout"):
                if operand.type.layout != callee_func.type.inputs[i].layout:
                    new_input_types[i] = operand.type
                    callee_func.arguments[i].set_type(operand.type)

        # Update function type
        new_func_type = FunctionType.get(new_input_types, [new_return_type])
        callee_func.attributes["function_type"] = TypeAttr.get(new_func_type)

        # CRITICAL: Update ALL call sites to this function with the new return type
        self._update_all_call_sites(callee_name, new_return_type)

        # Update downstream function parameters that use this result
        for use in call_op.result.uses:
            if isinstance(use.owner, func_d.CallOp):
                self._update_downstream_param_type(
                    use.owner, call_op.result, new_return_type
                )

    def _update_all_call_sites(self, callee_name, new_return_type):
        """Update all call sites to a function when its signature changes."""
        for func in self.module.body.operations:
            if not isinstance(func, func_d.FuncOp):
                continue
            for op in func.entry_block.operations:
                if isinstance(op, func_d.CallOp):
                    call_callee = FlatSymbolRefAttr(op.attributes["callee"]).value
                    if call_callee == callee_name:
                        # Update the call operation's result type
                        op.result.set_type(new_return_type)

                        # If this call's result is used by another function,
                        # we need to update that function's parameter types too
                        for use in op.result.uses:
                            user_op = use.owner
                            if isinstance(user_op, func_d.CallOp):
                                self._update_downstream_param_type(
                                    user_op, op.result, new_return_type
                                )

    def _update_downstream_param_type(self, call_op, partitioned_value, new_type):
        """Update parameter type in downstream function call."""
        callee_name = FlatSymbolRefAttr(call_op.attributes["callee"]).value
        callee_func = self._find_function(callee_name, error=False)
        if not callee_func:
            return

        for i, operand in enumerate(call_op.operands):
            if operand == partitioned_value:
                callee_func.arguments[i].set_type(new_type)

                new_inputs = list(callee_func.type.inputs)
                new_inputs[i] = new_type
                new_func_type = FunctionType.get(new_inputs, callee_func.type.results)
                callee_func.attributes["function_type"] = TypeAttr.get(new_func_type)
                break

    def _update_global_types(self, buffers, partition_type, dim, factor):
        """Update global memory reference types."""
        buffer_names = {buf.name for buf in buffers}

        for op in self.module.body.operations:
            if isinstance(op, memref_d.GlobalOp):
                if op.attributes["sym_name"].value in buffer_names:
                    old_type = op.attributes["type"].value
                    layout_attr = self._compute_partition_layout(
                        old_type.shape, partition_type, dim, factor
                    )
                    new_type = MemRefType.get(
                        old_type.shape,
                        old_type.element_type,
                        layout_attr,
                        old_type.memory_space,
                    )
                    op.attributes["type"] = TypeAttr.get(new_type)

    # @wrapped_apply
    def buffer_at_regular(self, target, axis):
        """
        Creates a chip buffer to hold the values of `target` written to in loop with index `axis`
        instead of immediately writing them to memory.

        Parameters
        ----------
        target: allo.ir.utils.MockBuffer
            An array written to in a loop.

        axis: str
            The loop index whose body contains writes to target
        """

        _, _, target = find_buffer(self.module, target, self.func_args)
        func, axis = self._get_func_and_axis(axis)
        band_name, axis = find_loop_in_bands(func, axis)
        ip = InsertionPoint.at_block_terminator(func.entry_block)
        op_hdl = allo_d.CreateOpHandleOp(band_name, ip=ip)
        loop_hdl = allo_d.CreateLoopHandleOp(op_hdl.result, StringAttr.get(axis), ip=ip)
        memref_type = MemRefType.get((1,), F32Type.get())
        allo_d.BufferAtOp(memref_type, target.result, loop_hdl.result, ip=ip)

    def buffer_at(self, target, axis):
        """
        Creates a chip buffer to hold the values of `target` written to in loop with index `axis`
        instead of immediately writing them to memory.

        Parameters
        ----------
        target: allo.ir.utils.MockBuffer
            An array written to in a loop.

        axis: str
            The loop index whose body contains writes to target
        """
        if self.systolic:
            return self.buffer_at_systolic(target, axis)

        with self.module.context, Location.unknown():
            self.buffer_at_regular(target, axis)
        _mlir_lower_pipeline(self.module)
        # Update top function in the current context
        for op in self.module.body.operations:
            if isinstance(op, func_d.FuncOp) and op.name.value == self.top_func_name:
                self.top_func = op
                break
        else:
            raise RuntimeError("Top function not found")
        # Update insertion point
        self.ip = InsertionPoint.at_block_terminator(self.top_func.entry_block)
        # Record primitive sequences
        self.primitive_sequences.append(("buffer_at", [target, axis], {}))

    def buffer_at_systolic(self, target, axis):
        """
        Creates a chip buffer to hold the values of `target` written to in loop with index `axis`
        instead of immediately writing them to memory in a systolic array.

        Parameters
        ----------
        target: allo.ir.utils.MockBuffer
            An array written to in a loop.

        axis: str
            The loop index whose body contains writes to target
        """
        buff_name = target.name
        _, _, target = find_buffer(self.module, target, self.func_args)
        func, axis = self._get_func_and_axis(axis)
        band_name, axis = find_loop_in_bands(func, axis)
        band = self._find_band(band_name, func)
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
        load_type = MemRefType(target.result.type).element_type
        with self.module.context, Location.unknown():
            ip = InsertionPoint.at_block_begin(func.body.blocks[0])
            fifo_memref_type = MemRefType.get([i_size, j_size + 1, k_size], load_type)
            fifo_memref = memref_d.AllocOp(fifo_memref_type, [], [], ip=ip)
            fifo_memref.attributes["name"] = StringAttr.get(f"{buff_name}_fifo")
        fifo_mock_buffer = MockBuffer(func.name.value, f"{buff_name}_fifo")
        fifo_mock_buffer.result = fifo_memref.result
        setattr(self, f"{buff_name}_fifo", fifo_mock_buffer)
        return fifo_mock_buffer

    @wrapped_apply
    def reshape(self, target, shape):
        """
        Takes an array in the kernel, `target`, for example if the array is `B`, then would be `target` would be `<schedule>.B`, and reshapes it to tuple `shape`. As an example, if the desired shape is 32 by 4 by 8, the `<shape>` would be `(32, 4, 8)`.

        Parameters
        ----------
        target: allo.ir.utils.MockBuffer
            The array, represented by a memory, to reshape.

        shape: tuple
            The new shape of the memory.
        """

        _, _, target = find_buffer(self.module, target, self.func_args)
        eletype = MemRefType(target.result.type).element_type
        memref_type = MemRefType.get(shape, eletype)
        allo_d.ReshapeOp(memref_type, target.result, ip=self.ip)

    @wrapped_apply
    def pipeline(self, axis, initiation_interval=1, rewind=False):
        """
        Pipelines a loop with index `axis` into `initiation_interval` stages.

        Parameters
        ----------
        axis: str
            The index of the loop to pipeline.

        initiation_interval: int
            The initiation_interval to be used when pipelining.

        rewind: bool
            If true, rewinding is allowed, allowing continuous loop pipelining.
            This is only effective for perfect loop nests inside a top level function.
        """

        i32 = IntegerType.get_unsigned(32)
        ii = IntegerAttr.get(i32, initiation_interval)
        func, axis = self._get_func_and_axis(axis)
        band_name, axis = find_loop_in_bands(func, axis)
        if rewind:
            self.get_loops(func)[band_name][axis].loop.attributes[
                "rewind"
            ] = UnitAttr.get()
        self.get_loops(func)[band_name][axis].loop.attributes["pipeline_ii"] = ii

    @wrapped_apply
    def parallel(self, axis):
        """
        Instantiates a loop with index `axis` to be computed in parallel with the loops it is nested with.

        Parameters
        ----------
        axis: str
            The index of the loop to be computed in parallel.
        """

        func, axis = self._get_func_and_axis(axis)
        band_name, axis = find_loop_in_bands(func, axis)
        ip = InsertionPoint.at_block_terminator(func.entry_block)
        op_hdl = allo_d.CreateOpHandleOp(band_name, ip=ip)
        loop_hdl = allo_d.CreateLoopHandleOp(op_hdl.result, StringAttr.get(axis), ip=ip)
        allo_d.ParallelOp(loop_hdl.result, ip=ip)

    @wrapped_apply
    def inline(self, axis=None):
        """
        Inlines a function `axis`.

        Parameters
        ----------
        axis: str
            The function to inline.
        """

        assert axis is None or isinstance(axis, str), "Function name must be a string"
        if axis is None:
            axis = self.top_func_name
        func = self._find_function(axis)
        func.attributes["inline"] = UnitAttr.get()

    @wrapped_apply
    def dataflow(self, axis):
        """
        Applies a "dataflow" attribute to function `axis`. This allows for parallelism if the given function uses streams or the `to` schedule.

        Parameters
        ----------
        axis: str | allo.ir.LoopWrapper
            The function to add the attribute to.
        """

        if isinstance(axis, str):
            # function
            func = self._find_function(axis)
            func.attributes["dataflow"] = UnitAttr.get()
            return
        func, _ = self._get_func_and_axis(axis)
        band_name, loop_name = axis.name.split(".", 1)
        band_name = band_name.split(":")[1]
        cnt = 0

        def locate_loop(op):
            nonlocal cnt
            for ope in op.body.operations:
                if isinstance(ope, (scf_d.ForOp, affine_d.AffineForOp)):
                    locate_loop(ope)
            if (
                "loop_name" in op.attributes
                and op.attributes["loop_name"].value == loop_name
            ):
                cnt += 1
                op.attributes["dataflow"] = UnitAttr.get()

        for op in func.entry_block.operations:
            if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
                if (
                    "op_name" in op.attributes
                    and op.attributes["op_name"].value == band_name
                ):
                    locate_loop(op)

        if cnt == 0:
            raise RuntimeError(f"Dataflow loop {band_name}.{loop_name} not found")

    @wrapped_apply
    def compute_at(self, from_loop, target_loop):
        """
        If `from_loop` and `target_loop` are indices over the same range, `<schedule>.compute_at(from_loop, target_loop)` merges the two loops, taking
        the body of `from_loop` and appending it to the body of `target_loop`.

        Parameters
        ----------
        from_loop: str
            The loop whose body is being moved.

        target_loop: str
            The loop whose body is being appended to.
        """

        from_band, _ = find_loop_in_bands(self.top_func, from_loop)
        target_band, target_axis = find_loop_in_bands(self.top_func, target_loop)
        from_hdl = allo_d.CreateOpHandleOp(from_band, ip=self.ip)
        target_hdl = allo_d.CreateOpHandleOp(target_band, ip=self.ip)
        loop_hdl = allo_d.CreateLoopHandleOp(
            target_hdl.result, StringAttr.get(target_axis), ip=self.ip
        )
        allo_d.ComputeAtOp(
            from_hdl.result, target_hdl.result, loop_hdl.result, ip=self.ip
        )

    @wrapped_apply
    def reuse_at(self, target, axis):
        """
        Takes an array in a kernel, for example if the array is `B`, this would be `<schedule>.B`, accessed by index `axis` and creates a reuse buffer
        to reuse values from `target` which are accessed in a sequentially moving window.

        Parameters
        ----------
        target: allo.ir.utils.MockBuffer
            The array being accessed.

        axis: str
            The loop index used to access values in `target`
        """

        _, _, target = find_buffer(self.module, target, self.func_args)
        func, axis = self._get_func_and_axis(axis)
        band_name, axis = find_loop_in_bands(func, axis)
        ip = InsertionPoint.at_block_terminator(func.entry_block)
        op_hdl = allo_d.CreateOpHandleOp(band_name, ip=ip)
        loop_hdl = allo_d.CreateLoopHandleOp(op_hdl.result, StringAttr.get(axis), ip=ip)
        memref_type = MemRefType.get((1,), F32Type.get())

        def find_reuse_buffers(res):
            for func in self.module.body.operations:
                if isinstance(func, func_d.FuncOp):
                    for op in func.entry_block.operations:
                        if (
                            isinstance(op, memref_d.AllocOp)
                            and "name" in op.attributes
                            and band_name + "_reuse"
                            in StringAttr(op.attributes["name"]).value
                        ):
                            res.append(op)

        prev_reuse_buffers = []
        find_reuse_buffers(prev_reuse_buffers)
        allo_d.ReuseAtOp(memref_type, target.result, loop_hdl.result, ip=ip)
        _mlir_lower_pipeline(self.module)
        new_reuse_buffers = []
        find_reuse_buffers(new_reuse_buffers)
        new_reuse_buffers = [
            buf for buf in new_reuse_buffers if buf not in prev_reuse_buffers
        ]
        if len(new_reuse_buffers) != 1:
            raise RuntimeError("Reuse buffer not found")
        return MockBuffer(
            self.top_func_name,
            StringAttr(new_reuse_buffers[-1].attributes["name"]).value,
        )

    @wrapped_apply
    def to(self, target, dst, axis=None, depth=-1):
        """
        Takes an array in the kernel, `target`, for example if the array is `B`, this would be `target` would be `<schedule>.B`,
        and converts it into a stream. `dst` is the name of the array any value of `target` is written to.
        For example if `C[i, j] = B[i, j]`, `dst` would be specified as `"C"`. If values of `<target>` get written to multiple arrays.
        Multiple calls to `<schedule>.to(...)` may be needed.

        Parameters
        ----------
        target: allo.ir.utils.MockBuffer
            The array to convert to a stream.

        dst: str
            An array which a value of `target` is written to.

        axis: str
            Move axis-th loop body to xcel scope.

        depth: int
            The streaming channel depth.
        """

        return prim.to(
            self.module, target, dst, axis, depth, self.func_args, self.top_func_name
        )

    @wrapped_apply
    def unfold(self, band_name, axes):
        """
        Finds a set of nested loops with name `band_name` and for every `<i>` in list `axes`.
        The `<i>th` nested loop is unfolded into a constant number of copies of it's loop body.

        Parameters
        ----------
        band_name: str
            The set of nested loops to unroll.

        axes: list[int]
            A list of the axes to unroll.
        """

        if self.systolic:
            prepare_systolic(self, band_name)

        assert isinstance(axes, list), "Axes must be a list"
        axes.sort()
        assert axes == list(
            range(axes[0], axes[0] + len(axes))
        ), "Axes must be consecutive"
        # start from the inner most loop
        if ":" in band_name:
            func = self._find_function(band_name.split(":")[0])
            band_name = band_name.split(":")[1]
        else:
            func = self.top_func
        for axis in axes[::-1]:
            # Need to recompute the loop nests due to the MLIR bug:
            # https://reviews.llvm.org/D101422
            # Otherwise, it may hit invalid operations
            band = self._find_band(band_name, func)
            target_outer = band.get_outer_most()
            loops = list(band)
            op_to_remove = []
            _, loop_wrapper = loops[axis]
            loop = loop_wrapper.loop
            lower_bound = loop.attributes["lowerBoundMap"]
            assert str(lower_bound) == "affine_map<() -> (0)>", "Lower bound must be 0"
            upper_bound = loop.attributes["upperBoundMap"]
            upper_bound = int(
                re.findall(r"affine_map<\(\) -> \(([0-9]*)\)>", str(upper_bound))[0]
            )
            if axis > 0:
                ip = InsertionPoint.at_block_terminator(loops[axis - 1][1].loop.body)
            else:
                ip = InsertionPoint(target_outer)
            for op in loop.body.operations:
                if isinstance(op, affine_d.AffineYieldOp):
                    break

            def update_operand(op, old, new):
                if isinstance(op, affine_d.AffineForOp):
                    # pylint: disable=cell-var-from-loop
                    for in_op in op.body.operations:
                        update_operand(in_op, old, new)
                else:
                    op.operation.replace_uses_of_with(old, new)

            # unfold the body `upper_bound` times
            for idx in range(upper_bound):
                # pylint: disable=too-many-function-args
                cst_op = arith_d.ConstantOp(IndexType.get(), idx, ip=ip)
                # Directly duplicate the loop itself
                # (to preserve a scope for replacing the induction variable),
                # and replace the induction variable with the constant
                new_loop = loop.operation.clone(ip)
                for op in new_loop.body.operations:
                    if isinstance(op, affine_d.AffineYieldOp):
                        break
                    update_operand(op, new_loop.induction_variable, cst_op.result)
                    op.move_before(new_loop)
                    if isinstance(op, affine_d.AffineForOp):
                        new_name = (
                            f"{band_name}_{idx}"
                            if "op_name" not in op.attributes
                            else f"{op.attributes['op_name'].value}_{idx}"
                        )
                        op.attributes["op_name"] = StringAttr.get(new_name)
                    if isinstance(op, func_d.CallOp):
                        # Also need to duplicate the function outside the top function
                        old_func = self._find_function(
                            FlatSymbolRefAttr(op.attributes["callee"]).value
                        )
                        dup_func = old_func.operation.clone(InsertionPoint(func))
                        new_name = (
                            f"{FlatSymbolRefAttr(op.attributes['callee']).value}_{idx}"
                        )
                        dup_func.attributes["sym_name"] = StringAttr.get(new_name)
                        # extend self.func_args
                        self.func_args[new_name] = self.func_args[
                            FlatSymbolRefAttr(op.attributes["callee"]).value
                        ]
                        op.attributes["callee"] = FlatSymbolRefAttr.get(new_name)
                        if old_func not in op_to_remove:
                            op_to_remove.append(old_func)
                op_to_remove.append(new_loop)
            # need to erase at the end
            for op in op_to_remove:
                op.operation.erase()
            loop.operation.erase()
        # TODO: use a class to wrap the results
        return axes

    # pylint: disable=redefined-builtin
    @wrapped_apply
    def compose(self, schs: list, id=None, instantiate=None):
        """
        Uses `schs`, a schedule for a kernel called in this kernel, in this kernel.

        A kernel, `<k1>`, may call another kernel, `<k2>`. This means the output of `<k1>.customize()` will contain the MLIR for the compiled `<k2>`, `<s2'>`. `<s2'>` will not have any custom schedule.
        To use a custom schedule, `<s2>`, the compiled `<k2>` with some schedule can be created.
        This is inserted into the schedule for this kernel through `self.compose(<s2>)`.

        Parameters
        ----------
        schs: allo.customize.Schedule
            The schedule of a kernel used in `self`.

        id: str
            Identifies the schedule to replace contained in `self`.
            This schedule in `self` must be annotated if `id` is specified.

        instantiate: list
            This is a list of objects used to instantiate types `schs` is generic over.
        """

        def get_name(arg):
            if isinstance(arg, (LoopWrapper, MockBuffer)):
                arg = copy.copy(arg)
                orig_func_name = arg.func if arg.func is not None else sch.top_func_name
                func_name = (
                    orig_func_name if id is None else orig_func_name + "_" + str(id)
                )
                if self._find_function(func_name, error=False) is None:
                    func_name = orig_func_name + "_0"
                arg.func = func_name
                return arg
            orig_func_name = arg.split(":")[0] if ":" in arg else sch.top_func_name
            arg = arg.split(":")[1] if ":" in arg else arg
            func_name = orig_func_name if id is None else orig_func_name + "_" + str(id)
            if self._find_function(func_name, error=False) is None:
                func_name = orig_func_name + "_0"
            return f"{func_name}:{arg}"

        if not isinstance(schs, list):
            schs = [schs]
        for sch in schs:
            if isinstance(sch, PyFunctionType):
                schedule = customize(sch, instantiate=instantiate)
                if sch not in KERNEL2SCHEDULE:
                    raise RuntimeError(
                        f"Cannot find schedule for kernel {sch.__name__}"
                    )
                sch = KERNEL2SCHEDULE[sch](schedule)
            if not isinstance(sch, Schedule):
                raise TypeError("The first argument must be a Schedule object")

            if hasattr(sch, "stateful_var_map") and sch.stateful_var_map:
                stateful_seen = getattr(self, "_stateful_seen", {})
                self._stateful_seen = stateful_seen

                func_name = f"{sch.top_func_name}_{id}" if id else sch.top_func_name
                func = self._find_function(func_name)

                for _, (global_name, _) in sch.stateful_var_map.items():
                    seen = stateful_seen.setdefault(global_name, [])

                    suffix = id if id else f"inst{len(seen)}"
                    new_name = f"{global_name}_{suffix}"

                    original_global = next(
                        (
                            op
                            for op in self.module.body.operations
                            if isinstance(op, memref_d.GlobalOp)
                            and op.attributes["sym_name"].value == global_name
                        ),
                        None,
                    )
                    if original_global is None:
                        raise RuntimeError(f"Stateful global {global_name} not found")

                    target = (
                        original_global
                        if not seen
                        else original_global.operation.clone(
                            InsertionPoint(original_global)
                        )
                    )
                    target.attributes["sym_name"] = StringAttr.get(new_name)
                    seen.append(new_name)

                    for op in func.entry_block.operations:
                        if isinstance(op, memref_d.GetGlobalOp) and (
                            FlatSymbolRefAttr(op.attributes["name"]).value
                            == global_name
                        ):
                            op.attributes["name"] = FlatSymbolRefAttr.get(new_name)

            for primitive in sch.primitive_sequences:
                args, kwargs = primitive[1:]
                # Avoid changing the original schedule
                args = args.copy()
                kwargs = kwargs.copy()
                # Update axes
                if primitive[0] in {"reorder", "fuse"}:
                    args = [get_name(arg) for arg in args]
                elif primitive[0] in {
                    "split",
                    "unroll",
                    "pipeline",
                    "parallel",
                    "dataflow",
                }:
                    if "axis" in kwargs:
                        kwargs["axis"] = get_name(kwargs["axis"])
                    else:
                        args[0] = get_name(args[0])
                elif primitive[0] in {"buffer_at", "reuse_at"}:
                    if "axis" in kwargs:
                        kwargs["axis"] = get_name(kwargs["axis"])
                    else:
                        args[1] = get_name(args[1])
                elif primitive[0] == "unfold":
                    if "band_name" in kwargs:
                        kwargs["band_name"] = get_name(kwargs["band_name"])
                    else:
                        args[0] = get_name(args[0])
                # Update target buffers
                if primitive[0] in {
                    "partition",
                    "to",
                    "buffer_at",
                    "reuse_at",
                    "reshape",
                }:
                    if "target" in kwargs:
                        kwargs["target"] = get_name(kwargs["target"])
                    else:
                        args[0] = get_name(args[0])
                with self.module.context, Location.unknown():
                    primitive_func = getattr(self, primitive[0])
                    # directly apply primitives to new functions
                    primitive_func(*args, **kwargs)

    def get_equivalent_variables(self, name):
        use_def = analyze_use_def(self.module)
        for ele in use_def:
            if name in ele:
                return ele
        return []

    def build(
        self,
        target=None,
        mode=None,
        project=None,
        configs=None,
        wrap_io=True,
        use_memory=False,
    ):
        if target is None or target == "llvm":
            target = "llvm"
            return LLVMModule(
                self.module,
                top_func_name=self.top_func_name,
                ext_libs=self.ext_libs,
            )
        # TODO Add XLS DSLX Backend
        if target in {"xls", "xlscc"}:
            return XLSCCModule(
                self.module,
                top_func_name=self.top_func_name,
                project=project,
                use_memory=use_memory,
                mode=mode,
            )
        if target in {"vhls", "vivado_hls", "vitis_hls", "pynq", "tapa", "ihls"}:
            match target:
                case "vitis_hls":
                    platform = "vitis_hls"
                case "tapa":
                    platform = "tapa"
                case "ihls":
                    platform = "intel_hls"
                case "pynq":
                    platform = "pynq"
                case _:
                    platform = "vivado_hls"
            return HLSModule(
                self.module,
                top_func_name=self.top_func_name,
                platform=platform,
                mode=mode,
                project=project,
                ext_libs=self.ext_libs,
                configs=configs,
                func_args=self.func_args,
                wrap_io=wrap_io,
            )
        raise NotImplementedError(f"Target {target} is not supported")


def customize(
    fn: Union[Callable, str],
    verbose: bool = False,
    enable_tensor: bool = False,
    lower_linalg: bool = False,
    global_vars: dict = None,
    instantiate: list = None,
    context: Context = None,
    typing_rule_set: str = "default",
    unroll: bool = True,
) -> Schedule:
    """
    Args:
        - typing_rule_set (str): Identifier of the typing rule set used during IR building.
            This controls implicit type casting behavior.
            Currently supported values include `"default"`, which is primarily intended for HLS backends, and
            `"cpp-style"`, which follows C++-like typing rules and is used for the AIE backend.
            Defaults to `"default"`.
    """
    # Get Python AST
    if isinstance(fn, str):
        src, starting_line_no = fn, 1
        file_name = None
    else:
        src, starting_line_no = inspect.getsourcelines(fn)
        src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
        src = textwrap.dedent("\n".join(src))
        file_name = inspect.getfile(fn)
    tree = parse_ast(src, starting_line_no=starting_line_no, verbose=verbose)
    if instantiate is None:
        instantiate = []
    if global_vars is None:
        global_vars = get_global_vars(fn)
    # Type construction
    ctx_type_inf = ASTContext(
        tree=tree,
        global_vars=global_vars.copy(),
        mlir_ctx=Context() if context is None else context,
        inst=instantiate,
        unroll=unroll,
        enable_tensor=enable_tensor,
        typing_rule_set=typing_rule_set,
        verbose=verbose,
    )
    tree = TypeInferer()(ctx_type_inf, tree)
    # Start building IR
    ctx = ASTContext(
        tree=tree,
        global_vars=global_vars,
        mlir_ctx=Context() if context is None else context,
        inst=instantiate,
        func_predicate_tags=ctx_type_inf.func_predicate_tags,
        unroll=unroll,
        meta_fors_to_unroll=ctx_type_inf.meta_fors_to_unroll,
        enable_tensor=enable_tensor,
        verbose=verbose,
    )
    module = ASTTransformer()(ctx, tree, file_name)
    func_instances = {
        orig_name: {
            dim: f"{orig_name}_{str(freeze_list(predicate_tag))}"
            for dim, predicate_tag in kernel_instance_info.items()
        }
        for orig_name, kernel_instance_info in ctx.func_predicate_tags.items()
    }
    if lower_linalg:
        lower_linalg_and_attach_names(module)
        ctx.top_func = find_func_in_module(module, fn.__name__)
    sch = Schedule(
        module,
        ctx.top_func,
        ctx.func_args,
        InsertionPoint.at_block_terminator(ctx.top_func.entry_block),
        ext_libs=ctx.ext_libs,
        inst_list=instantiate,
        func_instances=func_instances,
    )
    sch.stateful_var_map = getattr(ctx, "stateful_var_map", {})
    # Attach buffers to schedule:
    # The reason why we do not attach buffers to function is that
    # we may have multiple schedules referring to the same function,
    # which will cause conflicts of different buffers in different contexts.
    if isinstance(fn, Callable):
        for name, buffer in ctx.buffers.items():
            if isinstance(buffer, MockArg):  # Function arguments
                setattr(
                    sch,
                    name,
                    MockBuffer(fn.__name__, name, buffer.idx),
                )
            elif isinstance(
                buffer, (memref_d.AllocOp, func_d.CallOp, memref_d.GetGlobalOp)
            ):  # Intermediate buffers
                setattr(sch, name, MockBuffer(fn.__name__, name))
    # Check if there are memory leaks
    # All live operations = {top_func} + {top_func_ip}
    buffer = None
    ctx.buffers = None
    global_vars = {}
    # Functions are stored in ctx.global_vars, which should also be removed
    ctx = None
    # assert module.context._get_live_operation_count() == 2, (
    #     "All live operations = 1 (top_func) + 1 (top_func_ip), "
    #     f"expected 2, but got {module.context._get_live_operation_count()}"
    # )
    return sch


def build_spmw(
    fn: Union[Callable, str],
    verbose: bool = False,
    global_vars: dict = None,
    context: Context = None,
    unroll: bool = True,
) -> Schedule:
    """
    Args:
        - typing_rule_set (str): Identifier of the typing rule set used during IR building.
            This controls implicit type casting behavior.
            Currently supported values include `"default"`, which is primarily intended for HLS backends, and
            `"cpp-style"`, which follows C++-like typing rules and is used for the AIE backend.
            Defaults to `"default"`.
    """
    from .ir.spmw_builder import ASTTransformer as SPMW_Builder
    from .ir.spmw_infer import TypeInferer as SPMW_TypeInferer

    # Get Python AST
    if isinstance(fn, str):
        src, starting_line_no = fn, 1
        file_name = None
    else:
        src, starting_line_no = inspect.getsourcelines(fn)
        src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
        src = textwrap.dedent("\n".join(src))
        file_name = inspect.getfile(fn)
    tree = parse_ast(src, starting_line_no=starting_line_no, verbose=verbose)
    instantiate = []
    if global_vars is None:
        global_vars = get_global_vars(fn)
    # Type construction
    ctx_type_inf = ASTContext(
        tree=tree,
        global_vars=global_vars.copy(),
        mlir_ctx=Context() if context is None else context,
        inst=instantiate,
        unroll=unroll,
        enable_tensor=False,
        typing_rule_set="cpp-style",
        verbose=verbose,
    )
    tree = SPMW_TypeInferer()(ctx_type_inf, tree)
    # Start building IR
    ctx = ASTContext(
        tree=tree,
        global_vars=global_vars,
        mlir_ctx=Context() if context is None else context,
        inst=instantiate,
        func_predicate_tags=ctx_type_inf.func_predicate_tags,
        unroll=unroll,
        meta_fors_to_unroll=ctx_type_inf.meta_fors_to_unroll,
        enable_tensor=False,
        verbose=verbose,
    )
    module = SPMW_Builder()(ctx, tree, file_name)
    func_instances = {
        orig_name: {
            dim: f"{orig_name}_{str(freeze_list(predicate_tag))}"
            for dim, predicate_tag in kernel_instance_info.items()
        }
        for orig_name, kernel_instance_info in ctx.func_predicate_tags.items()
    }
    sch = Schedule(
        module,
        ctx.top_func,
        ctx.func_args,
        InsertionPoint.at_block_terminator(ctx.top_func.entry_block),
        ext_libs=ctx.ext_libs,
        inst_list=instantiate,
        func_instances=func_instances,
    )
    sch.stateful_var_map = getattr(ctx, "stateful_var_map", {})
    # Attach buffers to schedule:
    # The reason why we do not attach buffers to function is that
    # we may have multiple schedules referring to the same function,
    # which will cause conflicts of different buffers in different contexts.
    if isinstance(fn, Callable):
        for name, buffer in ctx.buffers.items():
            if isinstance(buffer, MockArg):  # Function arguments
                setattr(
                    sch,
                    name,
                    MockBuffer(fn.__name__, name, buffer.idx),
                )
            elif isinstance(
                buffer, (memref_d.AllocOp, func_d.CallOp, memref_d.GetGlobalOp)
            ):  # Intermediate buffers
                setattr(sch, name, MockBuffer(fn.__name__, name))
    # Check if there are memory leaks
    # All live operations = {top_func} + {top_func_ip}
    buffer = None
    ctx.buffers = None
    global_vars = {}
    # Functions are stored in ctx.global_vars, which should also be removed
    ctx = None
    return sch
