# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from typing import Optional, Sequence, Union
    from ..ir import *
    from ._ods_common import (
        get_op_result_or_value as _get_op_result_or_value,
        get_op_results_or_values as _get_op_results_or_values,
        get_default_loc_context as _ods_get_default_loc_context,
    )
    from ._ods_common import _cext as _ods_cext

    _ods_ir = _ods_cext.ir
    from .._mlir_libs._mlirDialectsLinalg import fill_builtin_region
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


def isa(cls: Type, ty: Type):
    try:
        cls(ty)
        return True
    except ValueError:
        return False


class StructuredOpMixin:
    """All structured ops use the same mixin class."""

    def __init__(self, inputs, outputs=(), results=(), loc=None, ip=None):
        super().__init__(
            self.build_generic(
                results=list(results),
                operands=[list(inputs), list(outputs)],
                loc=loc,
                ip=ip,
            )
        )


def select_opview_mixin(parent_opview_cls):
    # TODO: This shouldn't be a heuristic: we should have a way to annotate
    # the OpView to note that it is a structured op.
    if (
        "__init__" not in parent_opview_cls.__dict__
        and hasattr(parent_opview_cls, "inputs")
        and hasattr(parent_opview_cls, "outputs")
        and hasattr(parent_opview_cls, "result_tensors")
    ):
        return StructuredOpMixin


def get_element_type(dtype):
    if MemRefType.isinstance(dtype):
        return MemRefType(dtype).element_type
    return RankedTensorType(dtype).element_type


class BroadcastOp:
    def __init__(self, inputs, outputs, dimensions, *, loc=None, ip=None):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.extend(_get_op_results_or_values(inputs))
        operands.extend(_get_op_results_or_values(outputs))
        _ods_context = _ods_get_default_loc_context(loc)
        attributes["dimensions"] = (
            dimensions
            if (
                issubclass(type(dimensions), _ods_ir.Attribute)
                or not _ods_ir.AttrBuilder.contains("DenseI64ArrayAttr")
            )
            else _ods_ir.AttrBuilder.get("DenseI64ArrayAttr")(
                dimensions, context=_ods_context
            )
        )
        for output in outputs:
            if isinstance(output.type, RankedTensorType):
                results.append(output.type)
        _ods_successors = None
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=operands,
                successors=_ods_successors,
                regions=regions,
                loc=loc,
                ip=ip,
            )
        )
        types = [get_element_type(inp.type) for inp in inputs] + [
            get_element_type(out.type) for out in outputs
        ]
        self.regions[0].blocks.append(*types)
        from ._linalg_ops_gen import YieldOp

        YieldOp(
            [self.regions[0].blocks[0].arguments[0]],
            ip=InsertionPoint(self.regions[0].blocks[0]),
        )

    @property
    def result(self):
        return self.operation.results[0]


class TransposeOp:
    def __init__(self, inputs, outputs, permutation, *, loc=None, ip=None):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.extend(_get_op_results_or_values(inputs))
        operands.extend(_get_op_results_or_values(outputs))
        _ods_context = _ods_get_default_loc_context(loc)
        attributes["permutation"] = (
            permutation
            if (
                issubclass(type(permutation), _ods_ir.Attribute)
                or not _ods_ir.AttrBuilder.contains("DenseI64ArrayAttr")
            )
            else _ods_ir.AttrBuilder.get("DenseI64ArrayAttr")(
                permutation, context=_ods_context
            )
        )
        for output in outputs:
            if isinstance(output.type, RankedTensorType):
                results.append(output.type)
        _ods_successors = None
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=operands,
                successors=_ods_successors,
                regions=regions,
                loc=loc,
                ip=ip,
            )
        )
        types = [get_element_type(inp.type) for inp in inputs] + [
            get_element_type(out.type) for out in outputs
        ]
        self.regions[0].blocks.append(*types)
        from ._linalg_ops_gen import YieldOp

        YieldOp(
            [self.regions[0].blocks[0].arguments[0]],
            ip=InsertionPoint(self.regions[0].blocks[0]),
        )

    @property
    def result(self):
        return self.operation.results[0]
