# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

import re
import inspect
import textwrap
import ast
from dataclasses import dataclass
from functools import wraps
from typing import Union
from collections.abc import Callable
import numpy as np

from hcl_mlir.ir import (
    Module,
    Context,
    Location,
    InsertionPoint,
    StringAttr,
    UnitAttr,
    IndexType,
    IntegerType,
    IntegerAttr,
    TypeAttr,
    FunctionType,
    F32Type,
    MemRefType,
    FlatSymbolRefAttr,
)
from hcl_mlir.dialects import (
    hcl as hcl_d,
    memref as memref_d,
    affine as affine_d,
    arith as arith_d,
    func as func_d,
)
from hcl_mlir.exceptions import (
    HCLValueError,
)

from .ir.visitor import ASTContext
from .ir.utils import MockArg, MockBuffer
from .ir.builder import ASTTransformer
from .ir.infer import TypeInferer
from .ir.transform import get_affine_loop_nests, find_loop_in_bands
from .ir.types import AlloType
from .ir.use_def import UseDefChain
from .passes import _mlir_lower_pipeline, lower_linalg_and_attach_names
from .backend.llvm import LLVMModule
from .backend.hls import HLSModule


def getsourcefile(obj):
    ret = inspect.getsourcefile(obj)
    if ret is None:
        ret = inspect.getfile(obj)
    return ret


def getsourcelines(obj):
    return inspect.getsourcelines(obj)


def _get_global_vars(_func):
    if isinstance(_func, Callable):
        # Discussions: https://github.com/taichi-dev/taichi/issues/282
        global_vars = _func.__globals__.copy()
    else:
        global_vars = {}

    # Get back to the outer-most scope (user-defined function)
    # Mainly used to get the annotation definitions (shape and type),
    # which are probably not defined in __globals__
    for name, var in inspect.stack()[2][0].f_locals.items():
        if isinstance(var, (int, float, AlloType)) or inspect.isfunction(var):
            global_vars[name] = var

    if isinstance(_func, Callable):
        freevar_names = _func.__code__.co_freevars
        closure = _func.__closure__
        if closure:
            freevar_values = [x.cell_contents for x in closure]
            for name, value in zip(freevar_names, freevar_values):
                global_vars[name] = value
    return global_vars


def wrapped_apply(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        sch = args[0]
        with sch.module.context, Location.unknown():
            res = fn(*args, **kwargs)
        _mlir_lower_pipeline(sch.module)
        # Remove previous Python-C++ references
        sch.module.context._clear_live_operations()
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
        sch.primitive_sequences.append((fn.__name__, args[1:], kwargs))
        return res

    return wrapper


@dataclass
class Partition:
    Complete = 0
    Block = 1
    Cyclic = 2


class Schedule:
    def __init__(
        self, module, top_func, func_args, ip, ext_libs=None, use_def_chain=None
    ):
        self.module = module
        self.top_func = top_func
        self.top_func_name = top_func.name.value
        self.func_args = func_args  # only store names here
        self.ip = ip
        self.primitive_sequences = []
        if ext_libs is None:
            ext_libs = []
        self.ext_libs = ext_libs
        self.use_def_chain = use_def_chain

    def get_loops(self):
        return get_affine_loop_nests(self.top_func)

    def _find_band(self, band):
        loops = self.get_loops()
        if band in loops.loops:
            return loops[band]
        raise RuntimeError(f"Band {band} not found")

    def _find_function(self, name):
        for func in self.module.body.operations:
            if isinstance(func, func_d.FuncOp) and func.name.value == name:
                return func
        raise RuntimeError(f"Function {name} not found")

    @wrapped_apply
    def split(self, axis, factor):
        band_name, axis = find_loop_in_bands(self.top_func, axis)
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            op_hdl.result, StringAttr.get(axis), ip=self.ip
        )
        i32 = IntegerType.get_unsigned(32)
        factor = IntegerAttr.get(i32, factor)
        hcl_d.SplitOp(loop_hdl.result, factor, ip=self.ip)

    @wrapped_apply
    def reorder(self, *args):
        band_name, _ = find_loop_in_bands(self.top_func, args[0])
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdls = []
        for arg in args:
            band_name, axis = find_loop_in_bands(self.top_func, arg)
            loop_hdls.append(
                hcl_d.CreateLoopHandleOp(
                    op_hdl.result, StringAttr.get(axis), ip=self.ip
                )
            )
        arg_results = [arg.result for arg in loop_hdls]
        hcl_d.ReorderOp(arg_results, ip=self.ip)

    @wrapped_apply
    def unroll(self, axis, factor=0):
        band_name, axis = find_loop_in_bands(self.top_func, axis)
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            op_hdl.result, StringAttr.get(axis), ip=self.ip
        )
        i32 = IntegerType.get_unsigned(32)
        factor = IntegerAttr.get(i32, factor)
        hcl_d.UnrollOp(loop_hdl.result, factor=factor, ip=self.ip)

    @wrapped_apply
    def fuse(self, *args):
        band_name, _ = find_loop_in_bands(self.top_func, args[0])
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdls = []
        for arg in args:
            band_name, axis = find_loop_in_bands(self.top_func, arg)
            loop_hdls.append(
                hcl_d.CreateLoopHandleOp(
                    op_hdl.result, StringAttr.get(axis), ip=self.ip
                )
            )
        arg_results = [arg.result for arg in loop_hdls]
        hcl_d.FuseOp(arg_results, ip=self.ip)

    def _find_target(self, target):
        assert isinstance(target, MockBuffer), "Target must be a buffer"
        if target.op is not None:
            return -1, target.op
        func_name, target_name = target.path.rsplit(".", 1)
        # Find arguments
        for idx, (name, op) in enumerate(
            zip(self.func_args[func_name], self.top_func.arguments)
        ):
            if name == target_name:
                return idx, MockArg(op)
        # Find inner intermediate buffers
        for op in self.top_func.entry_block.operations:
            if (
                isinstance(op, memref_d.AllocOp)
                and "name" in op.attributes
                and StringAttr(op.attributes["name"]).value == target_name
            ):
                # verify if it is a return tensor
                return_op = list(self.top_func.entry_block.operations)[-1]
                if len(return_op.operands) > 0 and return_op.operands[0] == op.result:
                    idx = len(self.top_func.arguments)
                else:
                    idx = -1
                return idx, op
            if (
                isinstance(op, func_d.CallOp)
                and "name" in op.attributes
                and StringAttr(op.attributes["name"]).value == target_name
            ):
                return -1, op
        raise RuntimeError(f"Target {target} not found")

    @wrapped_apply
    def partition(self, target, partition_type=Partition.Complete, dim=0, factor=0):
        # TODO: (1) test whether partition the same array
        #       (2) whether the partition has conflicts for different functions
        _, target = self._find_target(target)
        if partition_type > 2:
            raise HCLValueError("Invalid partition type")
        if dim < 0:
            raise HCLValueError("Invalid dimension")
        if factor < 0:
            raise HCLValueError("Invalid factor")
        if partition_type == Partition.Complete:
            partition_type = 0
        elif partition_type == Partition.Block:
            partition_type = 1
        elif partition_type == Partition.Cyclic:
            partition_type = 2
        else:
            raise HCLValueError("Not supported partition type")
        i32 = IntegerType.get_signless(32)
        ui32 = IntegerType.get_unsigned(32)
        partition_type = IntegerAttr.get(i32, partition_type)
        dim = IntegerAttr.get(ui32, dim)
        factor = IntegerAttr.get(ui32, factor)
        hcl_d.PartitionOp(
            target.result,
            partition_kind=partition_type,
            dim=dim,
            factor=factor,
            ip=self.ip,
        )
        # calling the same function
        partitioned_arrays = [target.result]
        if isinstance(target, func_d.CallOp):
            for call_op in self.top_func.entry_block.operations:
                if (
                    isinstance(call_op, func_d.CallOp)
                    and target.attributes["callee"] == call_op.attributes["callee"]
                ):
                    hcl_d.PartitionOp(
                        call_op.results[0],
                        partition_kind=partition_type,
                        dim=dim,
                        factor=factor,
                        ip=InsertionPoint.at_block_terminator(
                            self.top_func.entry_block
                        ),
                    )
                    partitioned_arrays.append(call_op.results[0])
        # TODO: Deep nested functions may still have chance to meet errors,
        #       since this process is not recursive.
        # Make sure the arguments of subfunctions are also partitioned
        func_to_partitioned = []
        for call_op in self.top_func.entry_block.operations:
            if isinstance(call_op, func_d.CallOp):
                # test arguments
                for idx, arg in enumerate(call_op.operands):
                    if arg in partitioned_arrays:
                        func_to_partitioned.append(
                            (FlatSymbolRefAttr(call_op.attributes["callee"]).value, idx)
                        )
                # test results
                for idx, res in enumerate(call_op.results):
                    if res in partitioned_arrays:
                        func_to_partitioned.append(
                            (
                                FlatSymbolRefAttr(call_op.attributes["callee"]).value,
                                len(call_op.operands) + idx,
                            )
                        )
        # Add partition ops to subfunctions
        # pylint: disable=too-many-nested-blocks
        for (
            func_name,
            idx,
        ) in func_to_partitioned:
            for op in self.module.body.operations:
                if (
                    isinstance(op, func_d.FuncOp)
                    and StringAttr(op.attributes["sym_name"]).value == func_name
                ):
                    if idx < len(op.arguments):
                        hcl_d.PartitionOp(
                            op.arguments[idx],
                            partition_kind=partition_type,
                            dim=dim,
                            factor=factor,
                            ip=InsertionPoint.at_block_terminator(op.entry_block),
                        )
                    else:
                        idx = idx - len(op.arguments)
                        assert (
                            idx == 0
                        ), "Can only partition one function return for now"
                        return_op = None
                        for inner_op in op.entry_block.operations:
                            if isinstance(inner_op, func_d.ReturnOp):
                                return_op = inner_op
                                break
                        else:
                            raise RuntimeError("No return op found")
                        hcl_d.PartitionOp(
                            return_op.operands[idx],
                            partition_kind=partition_type,
                            dim=dim,
                            factor=factor,
                            ip=InsertionPoint.at_block_terminator(op.entry_block),
                        )

    @wrapped_apply
    def buffer_at(self, target, axis):
        _, target = self._find_target(target)
        band_name, axis = find_loop_in_bands(self.top_func, axis)
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            op_hdl.result, StringAttr.get(axis), ip=self.ip
        )
        memref_type = MemRefType.get((1,), F32Type.get())
        hcl_d.BufferAtOp(memref_type, target.result, loop_hdl.result, ip=self.ip)

    @wrapped_apply
    def reshape(self, target, shape):
        _, target = self._find_target(target)
        eletype = MemRefType(target.result.type).element_type
        memref_type = MemRefType.get(shape, eletype)
        hcl_d.ReshapeOp(memref_type, target.result, ip=self.ip)

    @wrapped_apply
    def pipeline(self, axis, initiation_interval=1):
        i32 = IntegerType.get_unsigned(32)
        ii = IntegerAttr.get(i32, initiation_interval)
        band_name, axis = find_loop_in_bands(self.top_func, axis)
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            op_hdl.result, StringAttr.get(axis), ip=self.ip
        )
        hcl_d.PipelineOp(loop_hdl.result, ii=ii, ip=self.ip)

    @wrapped_apply
    def parallel(self, axis):
        band_name, axis = find_loop_in_bands(self.top_func, axis)
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            op_hdl.result, StringAttr.get(axis), ip=self.ip
        )
        hcl_d.ParallelOp(loop_hdl.result, ip=self.ip)

    @wrapped_apply
    def compute_at(self, from_loop, target_loop):
        from_band, _ = find_loop_in_bands(self.top_func, from_loop)
        target_band, target_axis = find_loop_in_bands(self.top_func, target_loop)
        from_hdl = hcl_d.CreateOpHandleOp(from_band, ip=self.ip)
        target_hdl = hcl_d.CreateOpHandleOp(target_band, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            target_hdl.result, StringAttr.get(target_axis), ip=self.ip
        )
        hcl_d.ComputeAtOp(
            from_hdl.result, target_hdl.result, loop_hdl.result, ip=self.ip
        )

    @wrapped_apply
    def reuse_at(self, target, axis):
        _, target = self._find_target(target)
        band_name, axis = find_loop_in_bands(self.top_func, axis)
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            op_hdl.result, StringAttr.get(axis), ip=self.ip
        )
        memref_type = MemRefType.get((1,), F32Type.get())

        def find_reuse_buffers(res):
            for op in self.top_func.entry_block.operations:
                if (
                    isinstance(op, memref_d.AllocOp)
                    and "name" in op.attributes
                    and StringAttr(band_name).value + "_reuse"
                    in StringAttr(op.attributes["name"]).value
                ):
                    res.append(op)

        prev_reuse_buffers = []
        find_reuse_buffers(prev_reuse_buffers)
        hcl_d.ReuseAtOp(memref_type, target.result, loop_hdl.result, ip=self.ip)
        _mlir_lower_pipeline(self.module)
        new_reuse_buffers = []
        find_reuse_buffers(new_reuse_buffers)
        new_reuse_buffers = [
            buf for buf in new_reuse_buffers if buf not in prev_reuse_buffers
        ]
        if len(new_reuse_buffers) != 1:
            raise RuntimeError("Reuse buffer not found")
        return MockBuffer(
            f"{self.top_func_name}.{StringAttr(new_reuse_buffers[0].attributes['name']).value}"
        )

    @wrapped_apply
    def to(self, target, dst, fifo_depth=-1):
        _, target = self._find_target(target)
        op_hdl = hcl_d.CreateOpHandleOp(StringAttr.get(dst), ip=self.ip)
        i32 = IntegerType.get_signless(32)
        self.top_func.attributes["dataflow"] = UnitAttr.get()
        hcl_d.InterKernelToOp(
            target.result,
            op_hdl.result,
            fifo_depth=IntegerAttr.get(i32, fifo_depth),
            ip=self.ip,
        )
        # Find target in the top function
        target_arr = {}
        for op in self.top_func.entry_block.operations:
            if isinstance(op, func_d.CallOp):
                for idx, arg in enumerate(op.operands):
                    if arg.owner == target:
                        target_arr[
                            FlatSymbolRefAttr(op.attributes["callee"]).value
                        ] = idx
        for func in self.module.body.operations:
            if isinstance(func, func_d.FuncOp) and func.name.value in target_arr:
                in_types = func.attributes["function_type"].value.inputs
                out_types = func.attributes["function_type"].value.results
                idx = target_arr[func.name.value]
                arg = func.arguments[idx]
                memref = MemRefType(arg.type)
                if fifo_depth == -1:
                    fifo_depth = int(np.prod(memref.shape))
                new_memref = MemRefType.get(
                    memref.shape,
                    memref.element_type,
                    memref.layout,
                    StringAttr.get(f"stream:{fifo_depth}"),
                )
                arg.set_type(new_memref)
                new_in_types = []
                for i, in_type in enumerate(in_types):
                    new_in_types.append(new_memref if i == idx else in_type)
                func_type = FunctionType.get(new_in_types, out_types)
                func.attributes["function_type"] = TypeAttr.get(func_type)

    @wrapped_apply
    def unfold(self, band_name, axes):
        assert isinstance(axes, list), "Axes must be a list"
        axes.sort()
        assert axes == list(
            range(axes[0], axes[0] + len(axes))
        ), "Axes must be consecutive"
        # start from the inner most loop
        for axis in axes[::-1]:
            # Need to recompute the loop nests due to the MLIR bug:
            # https://reviews.llvm.org/D101422
            # Otherwise, it may hit invalid operations
            band = self._find_band(band_name)
            target_outer = band.get_outer_most()
            loops = list(band)
            op_to_remove = []
            _, loop_wrapper = loops[axis]
            loop = loop_wrapper.loop
            lower_bound = loop.attributes["lower_bound"]
            assert str(lower_bound) == "affine_map<() -> (0)>", "Lower bound must be 0"
            upper_bound = loop.attributes["upper_bound"]
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
                        dup_func = old_func.operation.clone(
                            InsertionPoint(self.top_func)
                        )
                        new_name = (
                            f"{FlatSymbolRefAttr(op.attributes['callee']).value}_{idx}"
                        )
                        dup_func.attributes["sym_name"] = StringAttr.get(new_name)
                        op.attributes["callee"] = FlatSymbolRefAttr.get(new_name)
                        if old_func not in op_to_remove:
                            op_to_remove.append(old_func)
                op_to_remove.append(new_loop)
            # need to erase at the end
            for op in op_to_remove:
                op.operation.erase()
            loop.operation.erase()

    @wrapped_apply
    def compose(self, *schs):
        # pylint: disable=too-many-nested-blocks
        for sch in schs:
            if not isinstance(sch, Schedule):
                raise TypeError("The first argument must be a Schedule object")
            funcs_to_replace = []
            # Create a new module in the current context
            new_mod = Module.parse(str(sch.module), self.module.context)
            for func in new_mod.body.operations:
                # Move all the functions to the front of the current module
                if isinstance(func, func_d.FuncOp):
                    for target in self.module.body.operations:
                        if (
                            isinstance(target, func_d.FuncOp)
                            and func.name.value == target.name.value
                        ):
                            func.move_before(target)
                            target.operation.erase()
                            funcs_to_replace.append(func.name.value)
                            break
                    else:
                        raise RuntimeError(
                            f"Target function {func.name.value} not found"
                        )
            func = None
            new_mod = None
            # Need to update CallOp arguments since some of them may be partitioned
            # We simply replay all the primitives and find the `partition`
            for primitive in sch.primitive_sequences:
                if primitive[0] == "partition":
                    args, kwargs = primitive[1:]
                    # =================================
                    # The context is sch.module.context
                    if len(args) != 0:
                        target = args[0]
                    else:
                        target = kwargs["target"]
                    arg_idx, _ = sch._find_target(target)
                    if arg_idx == -1:
                        # The partitioned array is inside the subfunction,
                        # so we don't need to update the CallOp in the top level
                        continue
                    # ==================================
                    # The context is self.module.context
                    # Update top-level function call interface
                    for op in self.top_func.entry_block.operations:
                        if (
                            isinstance(op, func_d.CallOp)
                            and FlatSymbolRefAttr(op.attributes["callee"]).value
                            in funcs_to_replace
                        ):
                            # After all the transformations are added, we lower the module
                            # Otherwise, the MLIR verifier will complain
                            if arg_idx >= len(op.operands):
                                target = MockBuffer(
                                    f"{self.top_func_name}.{FlatSymbolRefAttr(op.attributes['callee']).value}",
                                    op=op,
                                )
                            else:
                                target = MockBuffer(
                                    f"{self.top_func_name}.{FlatSymbolRefAttr(op.attributes['callee']).value}",
                                    op=MockArg(op.operands[arg_idx]),
                                )
                            with self.module.context, Location.unknown():
                                self.partition.__wrapped__(
                                    self, target, *args[1:], **kwargs
                                )
                                # Still need to append the current partition primitive to the sequence
                                # This is used for deep nested composition
                                # e.g.
                                #   s1.partition(...)
                                #   s2.compose(s1)
                                #   s3.compose(s2)
                                # If not appended, s3 will not know the partition primitive of s1
                                self.primitive_sequences.append(
                                    ("partition", [target] + list(args[1:]), kwargs)
                                )

    def build(self, target=None, mode=None, project=None):
        if target is None or target == "llvm":
            target = "llvm"
            return LLVMModule(
                self.module,
                top_func_name=self.top_func_name,
                ext_libs=self.ext_libs,
            )
        if target in {"vhls", "vivado_hls", "vitis_hls"}:
            return HLSModule(
                self.module,
                top_func_name=self.top_func_name,
                platform="vivado_hls" if target != "vitis_hls" else "vitis_hls",
                mode=mode,
                project=project,
                ext_libs=self.ext_libs,
            )
        raise NotImplementedError(f"Target {target} is not supported")


def customize(
    fn: Union[Callable, str],
    verbose: bool = False,
    enable_tensor: bool = False,
    lower_linalg: bool = False,
    global_vars: dict = None,
    instantiate: dict = None,
):
    # Get Python AST
    if isinstance(fn, str):
        src = fn
    else:
        src, _ = getsourcelines(fn)
        src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
        src = textwrap.dedent("\n".join(src))
    if verbose:
        print(src)
    tree = ast.parse(src)
    if verbose:
        try:
            import astpretty

            astpretty.pprint(tree, indent=2, show_offsets=False)
        except ImportError:
            print(ast.dump(tree))
    if instantiate is None:
        instantiate = {}
    if global_vars is None:
        global_vars = _get_global_vars(fn)
    for typevar in instantiate:
        if typevar not in global_vars:
            raise RuntimeError(f"Type variable {typevar} not found")
        # Checking
        global_vars[typevar] = global_vars[typevar].instantiate(instantiate[typevar])
    # Use-def chain analysis
    use_def_chain = UseDefChain(global_vars.copy())
    use_def_chain.visit(tree)
    # Type construction
    ctx_type_inf = ASTContext(
        global_vars=global_vars,
        mlir_ctx=Context(),
        enable_tensor=enable_tensor,
        verbose=verbose,
    )
    tree = TypeInferer()(ctx_type_inf, tree)
    ctx_type_inf = None
    # Start building IR
    ctx = ASTContext(
        global_vars=global_vars,
        mlir_ctx=Context(),
        enable_tensor=enable_tensor,
        verbose=verbose,
    )
    module = ASTTransformer()(ctx, tree)
    if lower_linalg:
        lower_linalg_and_attach_names(module)
    sch = Schedule(
        module,
        ctx.top_func,
        ctx.func_args,
        InsertionPoint.at_block_terminator(ctx.top_func.entry_block),
        ext_libs=ctx.ext_libs,
        use_def_chain=use_def_chain,
    )
    # Attach buffers to schedule:
    # The reason why we do not attach buffers to function is that
    # we may have multiple schedules referring to the same function,
    # which will cause conflicts of different buffers in different contexts.
    if isinstance(fn, Callable):
        for name, buffer in ctx.buffers.items():
            if isinstance(buffer, (memref_d.AllocOp, MockArg, func_d.CallOp)):
                # Intermediate buffers and function arguments
                setattr(sch, name, MockBuffer(f"{fn.__name__}.{name}"))
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
