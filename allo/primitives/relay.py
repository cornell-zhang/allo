# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

from hcl_mlir.ir import (
    InsertionPoint,
    StringAttr,
    UnitAttr,
    IntegerType,
    IntegerAttr,
    TypeAttr,
    FunctionType,
    MemRefType,
)
from hcl_mlir.dialects import (
    hcl as hcl_d,
    memref as memref_d,
    func as func_d,
)
from hcl_mlir.ir import Type as MLIRType
from ..ir.transform import update_streaming_interface, find_buffer, find_func_in_module


def _to(module, target, dst, axis=None, depth=-1, func_args=None, top_func_name=None):
    assert func_args is not None, "Need to specify func_args"
    assert top_func_name is not None, "Need to specify top_func_name"
    func, _, target = find_buffer(module, target, func_args)
    func.attributes["dataflow"] = UnitAttr.get()
    top_func = find_func_in_module(module, top_func_name)
    ip = InsertionPoint.at_block_terminator(top_func.entry_block)
    # pylint: disable=too-many-nested-blocks
    if axis is None:
        op_hdl = hcl_d.CreateOpHandleOp(StringAttr.get(dst), ip=ip)
        i32 = IntegerType.get_signless(32)
        hcl_d.InterKernelToOp(
            target.result,
            op_hdl.result,
            fifo_depth=IntegerAttr.get(i32, depth),
            ip=ip,
        )
        update_streaming_interface(module, target, depth=depth)
    else:
        assert dst is not None, "Need to specify dst"
        # TODO: fix the ad-hoc logic here
        memref = MemRefType(target.result.type)
        space_time_label = ["T"] * len(memref.shape)
        for idx in dst:
            space_time_label[idx] = "S"
        memory_space = f"stream:{depth};{''.join(space_time_label)}"
        new_memref = MemRefType.get(
            memref.shape,
            memref.element_type,
            memref.layout,
            StringAttr.get(memory_space),
        )
        new_alloc = memref_d.AllocOp(new_memref, [], [], ip=InsertionPoint(target))
        new_alloc.attributes["name"] = target.attributes["name"]
        target.result.replace_all_uses_with(new_alloc.result)
        for use in new_alloc.result.uses:
            op = use.owner
            if isinstance(op, memref_d.SubViewOp):
                subview_memref = MLIRType.parse(
                    str(op.result.type)[:-1] + f', "{memory_space}">'
                )
                subview = memref_d.SubViewOp(
                    source=op.source,
                    result=subview_memref,
                    static_offsets=op.static_offsets,
                    static_sizes=op.static_sizes,
                    static_strides=op.static_strides,
                    offsets=op.offsets,
                    sizes=[],
                    strides=[],
                    ip=InsertionPoint(op),
                )
                op.result.replace_all_uses_with(subview.result)
                op.operation.erase()
        target.operation.erase()
        # Find target in the top function
        for op in func.entry_block.operations:
            if isinstance(op, func_d.CallOp):
                for func in module.body.operations:
                    if (
                        isinstance(func, func_d.FuncOp)
                        and func.name.value == op.attributes["callee"].value
                    ):
                        out_types = func.attributes["function_type"].value.results
                        new_in_types = []
                        for i, operand in enumerate(op.operands):
                            new_in_types.append(operand.type)
                        func_type = FunctionType.get(new_in_types, out_types)
                        func.attributes["function_type"] = TypeAttr.get(func_type)
                        for i, arg in enumerate(func.arguments):
                            arg.set_type(new_in_types[i])
