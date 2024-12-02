# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

import ast
import inspect
from collections.abc import Callable
from types import FunctionType as PyFunctionType
from .._mlir.ir import (
    MemRefType,
    IntegerType,
    F32Type,
    IntegerAttr,
    FloatAttr,
    StringAttr,
    AffineMap,
    AffineMapAttr,
)
from .._mlir.dialects import (
    memref as memref_d,
    affine as affine_d,
    arith as arith_d,
    tensor as tensor_d,
    func as func_d,
)
from .types import AlloType, Int, UInt, Fixed, UFixed, Index
from .symbol_resolver import ASTResolver


def _get_global_vars(_func):
    if isinstance(_func, Callable):
        # Discussions: https://github.com/taichi-dev/taichi/issues/282
        global_vars = _func.__globals__.copy()
    else:
        global_vars = {}

    # Get back to the outer-most scope (user-defined function)
    # Mainly used to get the annotation definitions (shape and type),
    # which are probably not defined in __globals__
    for name, var in inspect.stack()[3][0].f_locals.items():
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


def get_global_vars(func):
    global_vars = _get_global_vars(func)
    new_global_vars = global_vars.copy()
    for var in global_vars.values():
        # import functions from other files
        if isinstance(var, PyFunctionType):
            new_global_vars.update(_get_global_vars(var))
    return new_global_vars


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


def _adjust_line_numbers(node, offset):
    for child in ast.walk(node):
        if hasattr(child, "lineno"):
            child.lineno += offset
        if hasattr(child, "end_lineno"):
            child.end_lineno += offset


def parse_ast(src, starting_line_no=1, verbose=False):
    tree = ast.parse(src)
    _adjust_line_numbers(tree, starting_line_no - 1)
    if verbose:
        print(src)
        try:
            import astpretty

            astpretty.pprint(tree, indent=2, show_offsets=False)
        except ImportError:
            print(ast.dump(tree))
    return tree


def get_func_id_from_param_types(param_types):
    for param_type in param_types:
        if isinstance(param_type, str):
            return param_type
    return None


def get_all_funcs_except_top(s):
    funcs = []
    for func in s.module.body.operations:
        if (
            isinstance(func, func_d.FuncOp)
            and func.attributes["sym_name"].value != s.top_func_name
        ):
            funcs.append(func)
    return funcs


def resolve_generic_types(global_vars, type_var, call_val):
    name = type_var.name
    if type_var.bound is None:
        return name, call_val
    constrained_types = ASTResolver.resolve_param_types(type_var.bound, global_vars)
    for ty in constrained_types:
        if (
            (isinstance(ty, AlloType) and type(ty).isinstance(call_val))
            or ty == call_val
            or (hasattr(ty, "isinstance") and ty.isinstance(call_val))
        ):
            return name, call_val
    raise RuntimeError(f"Cannot resolve type {name} with {call_val}")


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
    def __init__(self, func, name, idx=None, op=None):
        self.func = func
        self.name = name
        # Normally we do not use this attribute to avoid possible context conflicts
        # only when we need to access the op directly, we set this attribute (e.g., compose)
        self.op = op
        # Used for identifying the location of function arguments
        # if self.idx is None, this variable is not a function argument
        self.idx = idx

    def __repr__(self):
        return (
            f"MockBuffer({self.func}:{self.name})"
            if self.idx is None
            else f"MockBuffer({self.func}:{self.name}:{self.idx})"
        )


class MockConstant(MockOp):
    def __init__(self, val, ctx, dtype=None):
        self.val = val
        self.ctx = ctx
        assert dtype is None or isinstance(
            dtype, AlloType
        ), f"Expect AlloType, got {dtype}"
        self.dtype = dtype

    @property
    def result(self):
        if self.dtype is not None:
            assert isinstance(self.dtype, Index)
            dtype = self.dtype.build()
            value_attr = IntegerAttr.get(dtype, self.val)
        elif isinstance(self.val, int):
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
        return [self.result]


class MockScalar(MockOp):
    def __init__(self, name, dtype, ctx, value=None):
        self.name = name
        self.ctx = ctx
        self.value = value
        assert isinstance(dtype, AlloType), f"Expect AlloType, got {dtype}"
        self.dtype = dtype
        if not ctx.enable_tensor:
            memref_type = MemRefType.get((), dtype.build())
            alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ctx.get_ip())
            alloc_op.attributes["name"] = StringAttr.get(name)
        else:
            alloc_op = tensor_d.EmptyOp(tuple(), dtype.build(), ip=ctx.get_ip())
        self.op = alloc_op

    @property
    def result(self):
        # pylint: disable=no-else-return
        if not self.ctx.enable_tensor:
            affine_map = AffineMap.get(dim_count=0, symbol_count=0, exprs=[])
            affine_attr = AffineMapAttr.get(affine_map)
            load = affine_d.AffineLoadOp(
                self.dtype.build(),
                self.op.result,
                [],
                affine_attr,
                ip=self.ctx.get_ip(),
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
        return [self.result]
