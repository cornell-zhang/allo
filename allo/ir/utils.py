# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

from hcl_mlir.ir import (
    MemRefType,
    IntegerType,
    F32Type,
    IntegerAttr,
    FloatAttr,
    StringAttr,
    AffineConstantExpr,
    AffineMap,
    AffineMapAttr,
)
from hcl_mlir.dialects import (
    memref as memref_d,
    affine as affine_d,
    arith as arith_d,
)
from hcl_mlir import get_mlir_type
from .types import AlloType


def get_extra_type_hints(dtype: AlloType):
    if str(dtype).startswith("int"):
        return "s"
    if str(dtype).startswith("uint"):
        return "u"
    return "_"


def get_kwarg(kwargs, name):
    for keyword in kwargs:
        if keyword.arg == name:
            return keyword.value
    raise RuntimeError(f"Keyword argument {name} not found")


def print_node(node):
    print(node.__class__.__name__, node.dtype, node.shape)


class MockOp:
    def __init__(self):
        pass


class MockArg(MockOp):
    def __init__(self, val):
        self.val = val

    @property
    def result(self):
        return self.val


class MockBuffer(MockOp):
    def __init__(self, path, op=None):
        self.path = path
        # Normally we do not use this attribute to avoid possible context conflicts
        # only when we need to access the op directly, we set this attribute (e.g., compose)
        self.op = op

    def __repr__(self):
        return f"MockBuffer({self.path})"


class MockConstant(MockOp):
    def __init__(self, val, ctx):
        self.val = val
        self.ctx = ctx

    @property
    def result(self):
        # TODO: Support other types
        if isinstance(self.val, int):
            dtype = IntegerType.get_signless(32)
            value_attr = IntegerAttr.get(dtype, self.val)
        else:
            dtype = F32Type.get()
            value_attr = FloatAttr.get(dtype, self.val)
        # pylint: disable=too-many-function-args
        const_op = arith_d.ConstantOp(dtype, value_attr, ip=self.ctx.get_ip())
        return const_op.result


class MockScalar(MockOp):
    def __init__(self, name, dtype, ctx):
        self.name = name
        self.ctx = ctx
        shape = (1,)
        ele_type = get_mlir_type(dtype)
        memref_type = MemRefType.get(shape, ele_type)
        alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ctx.get_ip())
        alloc_op.attributes["name"] = StringAttr.get(name)
        self.op = alloc_op

    @property
    def result(self):
        affine_map = AffineMap.get(
            dim_count=0, symbol_count=0, exprs=[AffineConstantExpr.get(0)]
        )
        affine_attr = AffineMapAttr.get(affine_map)
        load = affine_d.AffineLoadOp(
            self.op.result, [], affine_attr, ip=self.ctx.get_ip()
        )
        load.attributes["from"] = StringAttr.get(self.name)
        return load.result
