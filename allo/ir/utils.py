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
    tensor as tensor_d,
)
from .types import AlloType, Int, UInt, Fixed, UFixed


def get_extra_type_hints(dtype: AlloType):
    assert isinstance(dtype, AlloType), f"Expect AlloType, got {dtype}"
    if isinstance(dtype, (Int, Fixed)):
        return "s"
    if isinstance(dtype, (UInt, UFixed)):
        return "u"
    return "_"


def get_kwarg(kwargs, name):
    for keyword in kwargs:
        if keyword.arg == name:
            return keyword.value
    raise RuntimeError(f"Keyword argument {name} not found")


class MockOp:
    def __init__(self):
        pass


class MockArg(MockOp):
    def __init__(self, val, is_affine=True, idx=None):
        self.val = val
        self.is_affine = is_affine
        # Used for identifying the location of function arguments
        # if self.idx is None, this variable is not a function argument
        self.idx = idx

    @property
    def result(self):
        return self.val

    @property
    def results(self):
        return [self.result]


class MockBuffer(MockOp):
    def __init__(self, path, name, idx=None, op=None):
        self.path = path
        self.name = name
        # Normally we do not use this attribute to avoid possible context conflicts
        # only when we need to access the op directly, we set this attribute (e.g., compose)
        self.op = op
        # Used for identifying the location of function arguments
        # if self.idx is None, this variable is not a function argument
        self.idx = idx

    def __repr__(self):
        return (
            f"MockBuffer({self.path}:{self.name})"
            if self.idx is None
            else f"MockBuffer({self.path}:{self.name}:{self.idx})"
        )


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

    @property
    def results(self):
        # DO NOT USE THIS METHOD FOR ACCESSING RESULT!
        return [None]


class MockScalar(MockOp):
    def __init__(self, name, dtype, ctx, value=None):
        self.name = name
        self.ctx = ctx
        self.value = value
        shape = (1,)
        assert isinstance(dtype, AlloType), f"Expect AlloType, got {dtype}"
        self.dtype = dtype
        if not ctx.enable_tensor:
            memref_type = MemRefType.get(shape, dtype.build())
            alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ctx.get_ip())
            alloc_op.attributes["name"] = StringAttr.get(name)
        else:
            alloc_op = tensor_d.EmptyOp(tuple(), dtype.build(), ip=ctx.get_ip())
        self.op = alloc_op

    @property
    def result(self):
        # pylint: disable=no-else-return
        if not self.ctx.enable_tensor:
            affine_map = AffineMap.get(
                dim_count=0, symbol_count=0, exprs=[AffineConstantExpr.get(0)]
            )
            affine_attr = AffineMapAttr.get(affine_map)
            load = affine_d.AffineLoadOp(
                self.op.result, [], affine_attr, ip=self.ctx.get_ip()
            )
            load.attributes["from"] = StringAttr.get(self.name)
            return load.result
        else:
            return tensor_d.ExtractOp(
                tensor=self.op.result,
                indices=[],
                ip=self.ctx.get_ip(),
            ).result

    @property
    def results(self):
        # DO NOT USE THIS METHOD FOR ACCESSING RESULT!
        return [None]
