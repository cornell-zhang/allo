# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from ..ir import *
    from ._ods_common import (
        get_op_result_or_value as _get_op_result_or_value,
        get_op_results_or_values as _get_op_results_or_values,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Any, Sequence


class AffineForOp:
    """Specialization for the Affine for op class."""

    def __init__(
        self,
        lower_bound,
        upper_bound,
        step,
        lowerBoundMap,
        upperBoundMap,
        reduction=None,
        iter_args: Sequence[Any] = [],
        name="",
        stage="",
        *,
        loc=None,
        ip=None
    ):
        """Creates an Affine `for` operation.
        operation   ::= `affine.for` ssa-id `=` lower-bound `to` upper-bound
                        (`step` integer-literal)? `{` op* `}`

        lower-bound ::= `max`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
        upper-bound ::= `min`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
        shorthand-bound ::= ssa-id | `-`? integer-literal

        - `lower_bound` is the value to use as lower bound of the loop.
        - `upper_bound` is the value to use as upper bound of the loop.
        - `step` is the value to use as loop step.
        - `iter_args` is a list of additional loop-carried arguments.
        """
        results = [arg.type for arg in iter_args]
        attributes = {}
        attributes["step"] = step
        attributes["lower_bound"] = lowerBoundMap
        attributes["upper_bound"] = upperBoundMap
        attributes["loop_name"] = name
        if stage != "":
            attributes["op_name"] = stage
        if reduction:
            attributes["reduction"] = reduction
        if lower_bound == None and upper_bound == None:
            operands = list(iter_args)
        elif lower_bound != None and upper_bound == None:
            operands = [lower_bound] + list(iter_args)
        elif upper_bound != None and lower_bound == None:
            operands = [upper_bound] + list(iter_args)
        else:
            operands = [lower_bound, upper_bound] + list(iter_args)
        super().__init__(
            self.build_generic(
                regions=1,
                results=results,
                operands=operands,
                attributes=attributes,
                loc=loc,
                ip=ip,
            )
        )
        self.regions[0].blocks.append(IndexType.get(), *results)

    @property
    def body(self):
        """Returns the body (block) of the loop."""
        return self.regions[0].blocks[0]

    @property
    def induction_variable(self):
        """Returns the induction variable of the loop."""
        return self.body.arguments[0]

    @property
    def inner_iter_args(self):
        """Returns the loop-carried arguments usable within the loop.
        To obtain the loop-carried operands, use `iter_args`.
        """
        return self.body.arguments[1:]


class AffineLoadOp:
    """Specialization for the MemRef load operation."""

    def __init__(self, memref, indices, map=None, *, loc=None, ip=None):
        memref_resolved = _get_op_result_or_value(memref)
        indices_resolved = [] if indices is None else _get_op_results_or_values(indices)
        return_type = MemRefType(memref_resolved.type).element_type
        super().__init__(return_type, memref, indices_resolved, map, loc=loc, ip=ip)


class AffineStoreOp:
    def __init__(self, value, memref, indices, affine_attr=None, *, loc=None, ip=None):
        operands = []
        results = []
        operands.append(value)
        operands.append(memref)
        operands.extend(indices)
        attributes = {}
        if affine_attr == None:
            identity_map = AffineMap.get_identity(len(indices))
            affine_attr = AffineMapAttr.get(identity_map)
        attributes["map"] = affine_attr
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=operands,
                loc=loc,
                ip=ip,
            )
        )

    @property
    def value(self):
        return self.operation.operands[0]

    @property
    def memref(self):
        return self.operation.operands[1]

    @property
    def indices(self):
        _ods_variadic_group_length = len(self.operation.operands) - 3 + 1
        return self.operation.operands[2 : 2 + _ods_variadic_group_length]


class AffineIfOp:
    """
    The affine.if operation contains two regions for the “then” and “else” clauses. affine.if may return results that are defined in its regions. The values defined are determined by which execution path is taken. Each region of the affine.if must contain a single block with no arguments, and be terminated by affine.yield. If affine.if defines no values, the affine.yield can be left out, and will be inserted implicitly. Otherwise, it must be explicit. If no values are defined, the else block may be empty (i.e. contain no blocks).
    """

    def __init__(
        self, cond, set_operands, results_=[], *, hasElse=False, loc=None, ip=None
    ):
        operands = []
        results = []
        results.extend(results_)
        operands.extend(set_operands)
        attributes = {}
        attributes["condition"] = cond
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=operands,
                loc=loc,
                ip=ip,
            )
        )
        self.regions[0].blocks.append(*[])
        if hasElse:
            self.regions[1].blocks.append(*[])

    @property
    def then_block(self):
        return self.regions[0].blocks[0]

    @property
    def else_block(self):
        return self.regions[1].blocks[0]
