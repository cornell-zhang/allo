# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unexpected-keyword-arg, no-value-for-parameter

import numpy as np
from .._mlir.ir import (
    Location,
    UnitAttr,
    StringAttr,
    InsertionPoint,
    MemRefType,
    AffineMapAttr,
    IntegerAttr,
    IntegerType,
    FlatSymbolRefAttr,
    FunctionType,
    TypeAttr,
)
from .._mlir.dialects import (
    memref as memref_d,
    affine as affine_d,
    scf as scf_d,
    func as func_d,
)
from .utils import MockArg, MockBuffer, get_extra_type_hints


class LoopWrapper:
    def __init__(self, name, loop):
        self.name = name
        self.loop = loop
        self.func = self.name.split(":")[0]
        self.band = self.name.split(":")[1].split(".")[0]

    def __repr__(self):
        return f"LoopWrapper({self.name})"


class LoopBand:
    def __init__(self):
        """
        Loops will be directly attached to this class as an attribute
        Naming convention: `func:band.loop`
        """
        self.loops = {}

    def add_loop(self, func, band, name, loop):
        full_name = f"{func}:{band}.{name}"
        if not isinstance(loop, LoopBand):
            loop = LoopWrapper(full_name, loop)
        self.loops[name] = loop
        setattr(self, name, loop)

    def get_outer_most(self):
        return next(self.loops.values().__iter__()).loop

    def __repr__(self):
        return f"LoopBand({list(self.loops.keys())})"

    def __iter__(self):
        return self.loops.items().__iter__()

    def __getitem__(self, name):
        if name in self.loops:
            return self.loops[name]
        raise AttributeError(f"No such loop {name}")


def get_loop_band_names(func):
    results = []
    for op in func.entry_block.operations:
        if isinstance(op, affine_d.AffineForOp):
            results.append(op.attributes["op_name"])
    return results


def find_buffer(module, target, func_args):
    assert isinstance(target, MockBuffer), "Target must be a buffer"
    if target.op is not None:
        return None, -1, target.op
    func_name, target_name = target.func, target.name
    target_func = None
    for op in module.body.operations:
        if (
            isinstance(op, func_d.FuncOp)
            and StringAttr(op.attributes["sym_name"]).value == func_name
        ):
            target_func = op
            break
    if target_func is None:
        raise RuntimeError(f"Target function {func_name} not found")
    # Find arguments
    for idx, (name, op) in enumerate(zip(func_args[func_name], target_func.arguments)):
        if name == target_name:
            return target_func, idx, MockArg(op)
    # Find inner intermediate buffers
    for op in target_func.entry_block.operations:
        if (
            isinstance(op, memref_d.AllocOp)
            and "name" in op.attributes
            and StringAttr(op.attributes["name"]).value == target_name
        ):
            # verify if it is a return tensor
            return_op = list(target_func.entry_block.operations)[-1]
            if len(return_op.operands) > 0 and return_op.operands[0] == op.result:
                idx = len(target_func.arguments)
            else:
                idx = -1
            return target_func, idx, op
        if (
            isinstance(op, func_d.CallOp)
            and "name" in op.attributes
            and StringAttr(op.attributes["name"]).value == target_name
        ):
            return target_func, -1, op
        if isinstance(op, memref_d.GetGlobalOp) and op.name.value == target_name:
            return target_func, -1, op
    raise RuntimeError(f"Target {target} not found")


def find_loop_in_bands(func, axis):
    """
    Parameters
    ----------
    func: _mlir.ir.func.FuncOp
        The function to search for the loop
    axis: str or LoopWrapper
        The name of the loop or the LoopWrapper object

    Returns
    -------
    band_name, str
        The name of the band containing the loop
    axis_name: str
        The name of the loop
    """
    results = []
    bands = get_affine_loop_nests(func)
    # pylint: disable=no-else-return
    if isinstance(axis, LoopWrapper):
        return axis.band, axis.name.split(".", 1)[1]
    elif isinstance(axis, affine_d.AffineForOp):
        axis_name = StringAttr(axis.attributes["loop_name"]).value
        for _, band in bands:
            op_name = None
            for i, (_, for_loop) in enumerate(band):
                if i == 0:
                    op_name = StringAttr(for_loop.attributes["op_name"]).value
                if for_loop == axis:
                    return op_name, axis_name
        raise RuntimeError(f"Cannot find the band of loop {axis_name}")
    else:  # axis is a string
        axis_name = axis
        for _, band in bands:
            op_name = None
            for i, (name, for_loop) in enumerate(band):
                if i == 0:
                    op_name = StringAttr(for_loop.loop.attributes["op_name"]).value
                if name == axis_name:
                    results.append(op_name)
        if len(results) == 0:
            raise RuntimeError(
                f"Cannot find the band of loop {axis_name} in function {func.sym_name}"
            )
        if len(results) > 1:
            raise RuntimeError(f"Find multiple bands containing loop {axis_name}")
        return results[0], axis_name


def get_affine_loop_nests(func):
    cnt_unnamed = 0

    def DFS(operations, band):
        nonlocal cnt_unnamed
        for op in operations:
            if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
                if "loop_name" not in op.attributes:
                    name = f"L_{cnt_unnamed}"
                    cnt_unnamed += 1
                else:
                    name = StringAttr(op.attributes["loop_name"]).value
                band.add_loop(func_name, band_name, name, op)
                DFS(op.body.operations, band)
            elif isinstance(op, affine_d.AffineIfOp):
                DFS(op.thenRegion.blocks[0].operations, band)
                try:
                    DFS(op.elseRegion.blocks[0].operations, band)
                except IndexError:
                    pass
            elif isinstance(op, scf_d.IfOp):
                DFS(op.then_block.operations, band)
                try:
                    DFS(op.else_block.operations, band)
                except IndexError:
                    pass

    results = LoopBand()
    # get function name
    func_name = func.attributes["sym_name"].value
    for op in func.entry_block.operations:
        if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):  # outer-most
            band_name = StringAttr(op.attributes["op_name"]).value
            band = LoopBand()
            band.add_loop(
                func_name, band_name, StringAttr(op.attributes["loop_name"]).value, op
            )
            DFS(op.body.operations, band)
            results.add_loop(func_name, "", band_name, band)
    return results


def annotate(op, name):
    op.attributes[name] = UnitAttr.get()


def build_for_loops(grid, ip, name="loop", stage_name=None):
    for_loops = []
    if isinstance(name, str):
        names = [name + f"_l_{i}" for i in range(len(grid))]
        if stage_name is None:
            stage_name = "S_" + name
    else:  # list
        names = name
        if stage_name is None:
            stage_name = "S_" + "_".join(names)
    assert len(grid) >= 1

    def recursive_for(for_handle, idx):
        if idx == len(grid):
            return
        with InsertionPoint(for_handle.body.operations[0]) as ip:
            new_for = affine_d.AffineForOp(
                lower_bound=0,
                upper_bound=grid[idx],
                step=1,
                iter_args=[],
                lower_bound_operands=None,
                upper_bound_operands=None,
                ip=ip,
            )
            new_for.attributes["loop_name"] = StringAttr.get(names[idx])
            affine_d.AffineYieldOp([], ip=InsertionPoint(new_for.body))
            for_loops.append(new_for)
            recursive_for(new_for, idx + 1)

    if not isinstance(ip, InsertionPoint):
        ip = InsertionPoint(ip)
    with ip:
        for_handle = affine_d.AffineForOp(
            lower_bound=0,
            upper_bound=grid[0],
            step=1,
            iter_args=[],
            lower_bound_operands=None,
            upper_bound_operands=None,
            ip=ip,
        )
        affine_d.AffineYieldOp([], ip=InsertionPoint(for_handle.body))
        for_handle.attributes["loop_name"] = StringAttr.get(names[0])
        for_handle.attributes["op_name"] = StringAttr.get(stage_name)
    for_loops.append(for_handle)
    recursive_for(for_handle, 1)
    return for_loops


def update_streaming_interface(module, target, depth=-1):
    # Find target in the top function
    target_arr = {}
    # pylint: disable=too-many-nested-blocks
    for func in module.body.operations:
        if isinstance(func, func_d.FuncOp):
            for op in func.entry_block.operations:
                if isinstance(op, func_d.CallOp):
                    for idx, arg in enumerate(op.operands):
                        if arg.owner == target:
                            target_arr[
                                FlatSymbolRefAttr(op.attributes["callee"]).value
                            ] = idx
    # update function arguments
    for func in module.body.operations:
        if isinstance(func, func_d.FuncOp) and func.name.value in target_arr:
            in_types = func.attributes["function_type"].value.inputs
            out_types = func.attributes["function_type"].value.results
            idx = target_arr[func.name.value]
            arg = func.arguments[idx]
            memref = MemRefType(arg.type)
            if depth == -1:
                depth = int(np.prod(memref.shape))
            new_memref = MemRefType.get(
                memref.shape,
                memref.element_type,
                memref.layout,
                StringAttr.get(f"stream:{depth}"),
            )
            arg.set_type(new_memref)
            new_in_types = []
            for i, in_type in enumerate(in_types):
                new_in_types.append(new_memref if i == idx else in_type)
            func_type = FunctionType.get(new_in_types, out_types)
            func.attributes["function_type"] = TypeAttr.get(func_type)


def create_data_movement(
    tensors, name, ip, from_memory=True, flatten=False, mapping=None
):

    if len(tensors) != 2:
        raise IndexError("One source and one destination ONLY!")

    src_tensor = tensors[0]
    des_tensor = tensors[1]

    if from_memory:
        shape = MemRefType(des_tensor.type).shape
    else:
        shape = MemRefType(src_tensor.type).shape

    if mapping is not None:
        loop_bounds, src_pattern, dst_pattern = mapping
    else:
        loop_bounds, src_pattern, dst_pattern = shape, None, None

    for_loops = build_for_loops(loop_bounds, ip, name)
    for_loops[-1].attributes["pipeline_ii"] = IntegerAttr.get(
        IntegerType.get_unsigned(32), 1
    )
    for_loops[-1].attributes["rewind"] = UnitAttr.get()
    induction_vars = [for_loop.induction_variable for for_loop in for_loops]

    with InsertionPoint(for_loops[-1].body.operations[0]):
        if not flatten:
            var_str = ", ".join([f"d{i}" for i in range(len(loop_bounds))])
            if dst_pattern is None:
                dst_pattern = var_str
            affine_attr = AffineMapAttr.parse(
                f"affine_map<({var_str})->({dst_pattern})>"
            )
            load = affine_d.AffineLoadOp(
                MemRefType(src_tensor.type).element_type,
                src_tensor,
                induction_vars,
                affine_attr,
            )
            affine_d.AffineStoreOp(
                load.result,
                des_tensor,
                induction_vars,
                affine_attr,
            )
        else:
            # Build affine attribution
            out_str = ""
            reversed_shape = list(shape)[::-1]
            for i in range(len(shape)):
                s_str = " * ".join([str(s) for s in reversed_shape[:i]])
                if s_str != "":
                    out_str = s_str + f" * d{len(shape) - i - 1}" + out_str
                else:
                    out_str = f" d{len(shape) - i - 1}" + out_str
                if i != len(shape) - 1:
                    out_str = " + " + out_str

            in_str = ", ".join([f"d{i}" for i in range(len(loop_bounds))])

            # Build LoadOp
            if src_pattern is not None:
                load_str = src_pattern
            else:
                load_str = out_str if from_memory else in_str

            affine_attr = AffineMapAttr.parse(f"affine_map<({in_str})->({load_str})>")
            load = affine_d.AffineLoadOp(
                MemRefType(src_tensor.type).element_type,
                src_tensor,
                induction_vars,
                affine_attr,
            )

            # Build StoreOp
            if dst_pattern is not None:
                store_str = dst_pattern
            else:
                store_str = in_str if from_memory else out_str
            affine_attr = AffineMapAttr.parse(f"affine_map<({in_str})->({store_str})>")
            affine_d.AffineStoreOp(
                load.result,
                des_tensor,
                induction_vars,
                affine_attr,
            )


def wrap_data_movement(arg, ip, func_name, from_memory, flatten, mapping):
    # Build input types
    shape = MemRefType(arg.type).shape

    if not flatten:
        type_flatten = MemRefType.get(shape, MemRefType(arg.type).element_type)
    else:
        type_flatten = MemRefType.get(
            (np.prod(shape),), MemRefType(arg.type).element_type
        )

    type_buf = MemRefType(arg.type)
    input_types = [type_flatten, type_buf] if from_memory else [type_buf, type_flatten]

    # Build Function
    func_type = FunctionType.get(input_types, [])
    func_op = func_d.FuncOp(name=func_name, type=func_type, ip=ip)

    # Attach type hints
    if hasattr(arg, "dtype"):
        typehints = [get_extra_type_hints(arg.dtype)] * 2
        func_op.attributes["itypes"] = StringAttr.get("".join(typehints))

    # Set context
    func_op.add_entry_block()

    # Build ForOp for movement inside
    with func_op.context, Location.unknown():
        ip_move = InsertionPoint(func_op.entry_block)

        create_data_movement(
            func_op.arguments,
            func_name,
            ip=ip_move,
            from_memory=from_memory,
            flatten=flatten,
            mapping=mapping,
        )

    func_d.ReturnOp([], ip=InsertionPoint(func_op.entry_block))


def find_func_in_module(module, func_name):
    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp) and op.name.value == func_name:
            return op
    return None


def find_func_and_axis(self, axis):
    if isinstance(axis, LoopWrapper):
        func = (
            self._find_function(axis.func) if axis.func is not None else self.top_func
        )
        return func, axis
    if ":" in axis:
        func_name, axis = axis.split(":")
    else:
        func_name = self.top_func_name
    func = self._find_function(func_name)
    return func, axis
