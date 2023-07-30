# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

import inspect
import textwrap
import ast
from dataclasses import dataclass
from functools import wraps

from hcl_mlir.ir import (
    Module,
    Context,
    Location,
    InsertionPoint,
    StringAttr,
    IntegerType,
    IntegerAttr,
    F32Type,
    MemRefType,
    FlatSymbolRefAttr,
)
from hcl_mlir.dialects import (
    hcl as hcl_d,
    memref as memref_d,
    func as func_d,
    affine as affine_d,
)
from hcl_mlir.exceptions import (
    HCLValueError,
)

from .ir.builder import ASTTransformer, ASTContext
from .ir.transform import get_affine_loop_nests, find_loop_in_bands
from .build_module import _mlir_lower_pipeline
from .module import LLVMModule, HLSModule


def getsourcefile(obj):
    ret = inspect.getsourcefile(obj)
    if ret is None:
        ret = inspect.getfile(obj)
    return ret


def getsourcelines(obj):
    return inspect.getsourcelines(obj)


def _get_global_vars(_func):
    global_vars = {}

    # Get back to the outer-most scope (user-defined function)
    for name, var in inspect.stack()[2][0].f_locals.items():
        if isinstance(var, (int, float)) or inspect.isfunction(var):
            global_vars[name] = var

    # Discussions: https://github.com/taichi-dev/taichi/issues/282
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
        with args[0].module.context, Location.unknown():
            res = fn(*args, **kwargs)
        _mlir_lower_pipeline(args[0].module)
        args[0].primitive_sequences.append((fn.__name__, args[1:], kwargs))
        return res

    return wrapper


@dataclass
class Partition:
    Complete = 0
    Block = 1
    Cyclic = 2


class Schedule:
    def __init__(self, module, top_func, ip):
        self.module = module
        self.top_func = top_func
        self.ip = ip
        self.primitive_sequences = []

    def get_loops(self):
        return get_affine_loop_nests(self.top_func)

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
        for name in args:
            if isinstance(name, affine_d.AffineForOp):
                name = StringAttr(name.attributes["loop_name"]).value
            loop_hdls.append(
                hcl_d.CreateLoopHandleOp(
                    op_hdl.result, StringAttr.get(name), ip=self.ip
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
        for name in args:
            if isinstance(name, affine_d.AffineForOp):
                name = StringAttr(name.attributes["loop_name"]).value
            loop_hdls.append(
                hcl_d.CreateLoopHandleOp(
                    op_hdl.result, StringAttr.get(name), ip=self.ip
                )
            )
        arg_results = [arg.result for arg in loop_hdls]
        hcl_d.FuseOp(arg_results, ip=self.ip)

    @wrapped_apply
    def partition(self, target, partition_type=Partition.Complete, dim=0, factor=0):
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
        # TODO: Deep nested functions may still have chance to meet errors,
        #       since this process is not recursive.
        # Make sure the arguments of subfunctions are also partitioned
        func_to_partitioned = []
        for call_op in self.top_func.entry_block.operations:
            if isinstance(call_op, func_d.CallOp):
                # test arguments
                for idx, arg in enumerate(call_op.operands):
                    if arg == target.result:
                        func_to_partitioned.append(
                            (FlatSymbolRefAttr(call_op.attributes["callee"]).value, idx)
                        )
                # test results
                for idx, res in enumerate(call_op.results):
                    if res == target.result:
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
        band_name, axis = find_loop_in_bands(self.top_func, axis)
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            op_hdl.result, StringAttr.get(axis), ip=self.ip
        )
        memref_type = MemRefType.get((1,), F32Type.get())
        hcl_d.BufferAtOp(memref_type, target.result, loop_hdl.result, ip=self.ip)

    @wrapped_apply
    def reshape(self, target, shape):
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
        return new_reuse_buffers[0]

    @wrapped_apply
    def compose(self, *schs):
        for sch in schs:
            if not isinstance(sch, Schedule):
                raise TypeError("The first argument must be a Schedule object")
            funcs_to_replace = []
            new_mod = Module.parse(str(sch.module))
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
            # Need to update CallOp arguments since some of them may be partitioned
            # We simply replay all the primitives and find the `partition`
            for primitive in sch.primitive_sequences:
                if primitive[0] == "partition":
                    args, kwargs = primitive[1:]
                    if len(args) != 0:
                        target = args[0]
                    else:
                        target = kwargs["target"]
                    arg_idx = -1
                    for idx, arg in enumerate(sch.top_func.arguments):
                        if arg == target.result:
                            arg_idx = idx
                            break
                    else:
                        raise RuntimeError("Target not found")
                    for op in self.top_func.entry_block.operations:
                        if (
                            isinstance(op, func_d.CallOp)
                            and FlatSymbolRefAttr(op.attributes["callee"]).value
                            in funcs_to_replace
                        ):
                            from .ir.builder import MockArg

                            self.partition.__wrapped__(
                                self, MockArg(op.operands[arg_idx]), *args[1:], **kwargs
                            )
                            break

    def build(self, target=None, mode=None, project=None):
        if target is None or target == "llvm":
            target = "llvm"
            _mlir_lower_pipeline(self.module, lower_linalg=True)
            return LLVMModule(self.module, top_func_name=self.top_func.name.value)
        if target == "vhls":
            # FIXME: Handle linalg.fill
            _mlir_lower_pipeline(self.module, lower_linalg=True)
            return HLSModule(
                self.module,
                top_func_name=self.top_func.name.value,
                mode=mode,
                project=project,
            )
        raise NotImplementedError(f"Target {target} is not supported")


def customize(fn, verbose=False):
    # Get Python AST
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
    # Create MLIR module
    with Context() as mlir_ctx, Location.unknown():
        hcl_d.register_dialect(mlir_ctx)
        module = Module.create()
    # Start building IR
    global_vars = _get_global_vars(fn)
    ctx = ASTContext(global_vars=global_vars, mlir_ctx=mlir_ctx)
    ctx.set_ip(module.body)
    ASTTransformer()(ctx, tree)
    # Attach buffers to function
    for name, buffer in ctx.buffers.items():
        setattr(fn, name, buffer)
    return Schedule(
        module,
        ctx.top_func,
        InsertionPoint.at_block_terminator(ctx.top_func.entry_block),
    )
