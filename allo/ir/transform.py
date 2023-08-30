# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

import hcl_mlir
from hcl_mlir import UnitAttr, StringAttr, InsertionPoint, MemRefType
from hcl_mlir.dialects import (
    memref as memref_d,
    affine as affine_d,
    scf as scf_d,
    func as func_d,
)


class LoopWrapper:
    def __init__(self, name, loop):
        self.name = name
        self.loop = loop

    def __repr__(self):
        return f"LoopWrapper({self.name})"


class LoopBand:
    def __init__(self):
        """
        Loops will be directly attached to this class as an attribute
        """
        self.loops = {}

    def add_loop(self, path, name, loop):
        if path != "":
            full_name = f"{path}.{name}"
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


def find_loop_in_bands(func, axis):
    results = []
    bands = get_affine_loop_nests(func)
    # pylint: disable=no-else-return
    if isinstance(axis, LoopWrapper):
        assert "." in axis.name
        path, axis = axis.name.split(".", 1)
        return path, axis
    elif isinstance(axis, affine_d.AffineForOp):
        axis_name = StringAttr(axis.attributes["loop_name"]).value
        for _, band in bands:
            op_name = None
            for i, (_, for_loop) in enumerate(band):
                if i == 0:
                    op_name = for_loop.attributes["op_name"]
                if for_loop == axis:
                    return op_name, axis_name
        raise RuntimeError(f"Cannot find the band of loop {axis_name}")
    else:  # axis is a string
        axis_name = axis
        for _, band in bands:
            op_name = None
            for i, (name, for_loop) in enumerate(band):
                if i == 0:
                    op_name = for_loop.loop.attributes["op_name"]
                if name == axis_name:
                    results.append(op_name)
        if len(results) == 0:
            raise RuntimeError(f"Cannot find the band of loop {axis_name}")
        if len(results) > 1:
            raise RuntimeError(f"Find multiple bands containing loop {axis_name}")
        return results[0], axis_name


def get_affine_loop_nests(func):
    cnt_unnamed = 0

    def DFS(operations, band, path=""):
        nonlocal cnt_unnamed
        for op in operations:
            if isinstance(op, affine_d.AffineForOp):
                if "loop_name" not in op.attributes:
                    name = f"L_{cnt_unnamed}"
                    cnt_unnamed += 1
                else:
                    name = StringAttr(op.attributes["loop_name"]).value
                band.add_loop(path, name, op)
                DFS(op.body.operations, band, path)
            elif isinstance(op, (affine_d.AffineIfOp, scf_d.IfOp)):
                DFS(op.then_block.operations, band, path)
                try:
                    DFS(op.else_block.operations, band, path)
                except IndexError:
                    pass

    results = LoopBand()
    for op in func.entry_block.operations:
        if isinstance(op, affine_d.AffineForOp):  # outer-most
            band_name = StringAttr(op.attributes["op_name"]).value
            band = LoopBand()
            band.add_loop(band_name, StringAttr(op.attributes["loop_name"]).value, op)
            DFS(op.body.operations, band, path=band_name)
            results.add_loop("", band_name, band)
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
        with InsertionPoint(for_handle.body.operations[0]):
            new_for = hcl_mlir.make_for(0, grid[idx], name=names[idx])
            for_loops.append(new_for)
            recursive_for(new_for, idx + 1)

    if not isinstance(ip, InsertionPoint):
        ip = InsertionPoint(ip)
    with ip:
        for_handle = hcl_mlir.make_for(0, grid[0], name=names[0], stage=stage_name)
    for_loops.append(for_handle)
    recursive_for(for_handle, 1)
    return for_loops


def create_buffer(tensor, name, ip):
    with InsertionPoint(ip):
        alloc_op = memref_d.AllocOp(tensor.type, [], [])
        alloc_op.attributes["name"] = StringAttr.get(name)
        shape = MemRefType(tensor.type).shape
    for_loops = build_for_loops(shape, ip, name)
    induction_vars = [for_loop.induction_variable for for_loop in for_loops]
    with InsertionPoint(for_loops[-1].body.operations[0]):
        load = memref_d.LoadOp(tensor, induction_vars)
        memref_d.StoreOp(
            load.result,
            alloc_op.result,
            induction_vars,
        )
    # TODO: Upgrade LLVM version and use the following code
    # tensor.replace_all_uses_with(alloc_op.result)


def find_func_in_module(module, func_name):
    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp) and op.name.value == func_name:
            return op
    return None
