# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument, eval-used, redefined-variable-type, cell-var-from-loop

import ast
import copy
import sys
import os
import traceback
import inspect
import textwrap
import warnings
import sympy
import numpy as np

from .visitor import ASTVisitor, ASTContext, get_symbolic_expr
from .symbol_resolver import ASTResolver
from .types import (
    AlloType,
    Int,
    UInt,
    Fixed,
    UFixed,
    Index,
    uint1,
    int32,
    float16,
    float32,
    float64,
    Struct,
    Stream,
)
from .typing_rule import get_typing_rule
from ..utils import (
    is_anywidth_int_type_and_not_np,
    get_bitwidth_from_type,
    handle_overflow,
    make_anywidth_numpy_array,
    np_supported_types,
    construct_kernel_name,
)
from ..memory import DTensor, Layout
from ..logging import print_error_message
from .utils import parse_ast, get_func_id_from_param_types, resolve_generic_types


# pylint: disable=too-many-public-methods
class TypeInferer(ASTVisitor):
    def print_verbose(self, ctx: ASTContext, node: ast.AST):
        if isinstance(node, ast.Name):
            print("Name:", node.id, node.dtype, node.shape)
        else:
            print(node.__class__.__name__, node.dtype, node.shape)

    @staticmethod
    def visit_call_type(ctx: ASTContext, node: ast.expr):
        ty_cls = ASTResolver.resolve(node.func, ctx.global_vars)
        args = node.args

        if ty_cls is Fixed or ty_cls is UFixed:
            assert len(args) == 2
            assert isinstance(args[0], ast.Constant)
            assert isinstance(args[1], ast.Constant)
            dtype = ty_cls(
                ASTResolver.resolve_constant(args[0], ctx),
                ASTResolver.resolve_constant(args[1], ctx),
            )
        else:
            assert len(args) == 1
            dtype = ty_cls(ASTResolver.resolve_constant(args[0], ctx))
        return dtype

    @staticmethod
    def visit_type_hint(ctx: ASTContext, node: ast.AST):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Call):
                # e.g., a: UInt(16)[4]
                dtype = TypeInferer.visit_call_type(ctx, node.value)
            elif isinstance(node.value, ast.Subscript):
                # e.g., pipe: Stream[Ty, 4][4]
                base_type, base_shape, _ = TypeInferer.visit_type_hint(ctx, node.value)
                assert isinstance(base_type, Stream) and len(base_shape) == 0
                elts = (
                    node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice]
                )
                shape = tuple(ASTResolver.resolve(x, ctx.global_vars) for x in elts)
                assert all(
                    isinstance(x, (int)) for x in shape
                ), "stream array shape should be a compile time constant"
                return base_type, shape, None
            else:
                dtype = ASTResolver.resolve(node.value, ctx.global_vars)
            if dtype is Stream:
                # e.g., pipe: Stream[Ty, 4]
                assert (
                    isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2
                ), "Only support `ele_type` and `depth` for now"
                base_type, base_shape, _ = TypeInferer.visit_type_hint(
                    ctx, node.slice.elts[0]
                )
                depth = ASTResolver.resolve(node.slice.elts[1], ctx.global_vars)
                stream_dtype = Stream(dtype=base_type, shape=base_shape, depth=depth)
                shape = tuple()
                return stream_dtype, shape, None
            assert dtype is not None, f"Unsupported type `{node.value.id}`"
            size = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            elts = size.elts if isinstance(size, ast.Tuple) else [size]
            shape = tuple(ASTResolver.resolve_constant(x, ctx) for x in elts)
            return dtype, shape, Layout("R" * len(shape))  # default layout
        if isinstance(node, ast.Name):
            dtype = ASTResolver.resolve(node, ctx.global_vars)
            assert dtype is not None, f"Unsupported type `{node.id}`"
            return dtype, tuple(), None
        if isinstance(node, ast.Call):
            dtype = TypeInferer.visit_call_type(ctx, node)
            return dtype, tuple(), None
        if isinstance(node, ast.Constant):
            assert isinstance(node.value, str), "Only support string type annotation"
            tree = ast.parse(node.value)
            return TypeInferer.visit_type_hint(ctx, tree.body[0].value)
        if isinstance(node, ast.Attribute):
            # e.g., allo.ir.types.float32
            dtype = ASTResolver.resolve(node, ctx.global_vars)
            return dtype, tuple(), None
        if isinstance(node, ast.BinOp):
            # memory refinement
            # e.g., A: Ty[M] @ Layout("S0")
            dtype, shape, _ = TypeInferer.visit_type_hint(ctx, node.left)
            spec = ASTResolver.resolve(node.right, ctx.global_vars)
            return dtype, shape, spec
        raise RuntimeError("Unsupported function argument type")

    @staticmethod
    def visit_Name(ctx: ASTContext, node: ast.Name):
        var = ctx.get_symbol(node.id, allow_missing=True)
        if var is not None:
            node.dtype = var.dtype
            node.shape = var.shape
            return node
        if node.id in ctx.global_vars:
            var = ctx.global_vars[node.id]
            if isinstance(var, int):
                node.dtype = int32
                node.shape = tuple()
            elif isinstance(var, float):
                node.dtype = float32
                node.shape = tuple()
            elif isinstance(var, AlloType):
                node.dtype = Index()
                node.shape = tuple()
            else:
                raise RuntimeError(f"Unsupported global variable `{node.id}`")
            return node
        raise RuntimeError(f"Unsupported Name `{node.id}`")

    @staticmethod
    def visit_Constant(ctx: ASTContext, node: ast.Constant):
        node.shape = tuple()
        if isinstance(node.value, int):
            node.dtype = int32
        elif isinstance(node.value, float):
            node.dtype = float32
        elif isinstance(node.value, str):
            node.dtype = str
        elif node.value is None:
            return ASTResolver.resolve_constant(node.value, ctx)
        else:
            raise RuntimeError("Unsupported constant type")
        return node

    @staticmethod
    def visit_Tuple(ctx: ASTContext, node: ast.Tuple):
        visit_stmts(ctx, node.elts)
        node.shape = [elt.shape for elt in node.elts]
        node.dtype = [elt.dtype for elt in node.elts]
        return node

    @staticmethod
    def visit_Dict(ctx: ASTContext, node: ast.Dict):
        # Visit all keys and values
        visit_stmts(ctx, node.keys)
        visit_stmts(ctx, node.values)

        # Dictionary type is a mapping of keys to value types
        node.dtype = Struct({k.value: v.dtype for k, v in zip(node.keys, node.values)})
        node.shape = ()  # one dict is considered as one Struct-type scalar
        return node

    @staticmethod
    def visit_Index(ctx: ASTContext, node: ast.Index):
        value = visit_stmt(ctx, node.value)
        node.shape = value.shape
        node.dtype = value.dtype
        return node

    @staticmethod
    def visit_Attribute(ctx: ASTContext, node: ast.Attribute):
        res = visit_stmt(ctx, node.value)
        if node.attr == "T":
            node.dtype = res.dtype
            node.shape = res.shape[::-1]
            return node
        if node.attr == "reverse":
            if not isinstance(res.dtype, (Int, UInt)):
                raise RuntimeError("Can only reverse integers")
            node.dtype = res.dtype
            node.shape = res.shape
            return node
        if node.attr == "copy":
            node.dtype = res.dtype
            node.shape = res.shape
            return node
        if node.attr in {"bits", "fracs"} and isinstance(res, ast.Name):
            node.dtype = res.dtype
            node.shape = res.shape
            return node
        raise RuntimeError(f"Unsupported attribute `{node.attr}`")

    @staticmethod
    def visit_all_for(ctx: ASTContext, node: ast.For):
        with ctx.block_scope_guard():
            # Set loop induction variables
            if isinstance(node.target, ast.Tuple):
                ivs = list(node.target.elts)
            else:
                ivs = [node.target]
            for iv in ivs:
                iv.shape = tuple()
                iv.dtype = Index()
                iv_ = ctx.get_symbol(iv.id, allow_missing=True)
                assert (
                    iv_ is None
                ), "Please choose a different name for the loop iterator."
                ctx.put_symbol(name=iv.id, val=iv)
            visit_stmts(ctx, node.iter.args)
            visit_stmts(ctx, node.body)
            node.shape = None
            node.dtype = None
        return node

    @staticmethod
    def visit_For(ctx: ASTContext, node: ast.For):
        if node.orelse:
            raise RuntimeError("'else' clause for 'for' not supported in Allo kernels")
        with ctx.loop_scope_guard():
            if isinstance(node.iter, ast.Call):
                obj = ASTResolver.resolve(node.iter.func, ctx.global_vars)
                if (
                    obj is None
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == "range"
                ) or (obj is not None and obj.__name__ in {"grid", "reduction"}):
                    return TypeInferer.visit_all_for(ctx, node)
            raise RuntimeError("Unsupported for loop")

    @staticmethod
    def visit_broadcast(
        ctx: ASTContext,
        lhs_shape: list[int],
        rhs_shape: list[int],
        match_lhs: bool = False,
    ):
        # See the broadcasting rules in NumPy
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        # When operating on two arrays, NumPy compares their shapes element-wise.
        # It starts with the trailing (i.e. rightmost) dimension and works its way left.
        # Two dimensions are compatible when
        # 1. they are equal, or
        # 2. one of them is 1.
        if rhs_shape is None:
            return lhs_shape, [], []
        tmp_lhs_shape = list(lhs_shape)
        tmp_rhs_shape = list(rhs_shape)
        if match_lhs and len(tmp_lhs_shape) < len(tmp_rhs_shape):
            raise RuntimeError(f"Cannot broadcast {rhs_shape} to {lhs_shape}")
        # match larger shape
        lhs_dims, rhs_dims = set(), set()
        if len(tmp_lhs_shape) < len(tmp_rhs_shape):
            padded_dim = len(tmp_rhs_shape) - len(tmp_lhs_shape)
            tmp_lhs_shape = [1] * padded_dim + tmp_lhs_shape
            lhs_dims = set(range(padded_dim))
        elif len(tmp_lhs_shape) > len(tmp_rhs_shape):
            padded_dim = len(tmp_lhs_shape) - len(tmp_rhs_shape)
            tmp_rhs_shape = [1] * padded_dim + tmp_rhs_shape
            rhs_dims = set(range(padded_dim))
        # match shape
        # pylint: disable=consider-using-enumerate
        for i in range(len(tmp_lhs_shape)):
            if tmp_lhs_shape[i] == 1:
                tmp_lhs_shape[i] = tmp_rhs_shape[i]
                if tmp_rhs_shape[i] != 1:
                    if match_lhs:
                        raise RuntimeError(
                            f"Cannot broadcast {rhs_shape} to {lhs_shape}"
                        )
                    lhs_dims.add(i)
            elif tmp_rhs_shape[i] == 1:
                tmp_rhs_shape[i] = tmp_lhs_shape[i]
                if tmp_lhs_shape[i] != 1:
                    rhs_dims.add(i)
            else:
                assert (
                    tmp_lhs_shape[i] == tmp_rhs_shape[i]
                ), f"Shape mismatch, got {lhs_shape} and {rhs_shape}, and cannot be broadcasted"
        assert tmp_lhs_shape == tmp_rhs_shape
        return tuple(tmp_lhs_shape), list(lhs_dims), list(rhs_dims)

    @staticmethod
    def visit_general_binop(
        ctx: ASTContext, node: ast.AugAssign | ast.BinOp, lhs: ast.expr, rhs: ast.expr
    ):
        typing_rule = get_typing_rule(type(node.op))
        res_type = typing_rule(lhs.dtype, rhs.dtype)
        node.dtype = res_type
        final_shape, lhs_dims, rhs_dims = TypeInferer.visit_broadcast(
            ctx, lhs.shape, rhs_shape=None if rhs is None else rhs.shape
        )
        node.shape = final_shape
        node.dims = (lhs_dims, rhs_dims)
        if ctx.verbose:
            print(
                f"Broadcasted shape {lhs.shape} x {rhs.shape} -> {node.shape} for dims: {lhs_dims} & {rhs_dims}"
            )
        return node

    @staticmethod
    def visit_UnaryOp(ctx: ASTContext, node: ast.UnaryOp):
        operand = visit_stmt(ctx, node.operand)
        node.shape = operand.shape
        if isinstance(operand.dtype, UInt):
            # need to create a corresponding Int type
            node.dtype = Int(operand.dtype.bits)
        else:
            node.dtype = operand.dtype
        return node

    @staticmethod
    def visit_BinOp(ctx: ASTContext, node: ast.BinOp):
        lhs = visit_stmt(ctx, node.left)
        rhs = visit_stmt(ctx, node.right)
        return TypeInferer.visit_general_binop(ctx, node, lhs, rhs)

    @staticmethod
    def visit_Assign(ctx: ASTContext, node: ast.Assign):
        assert len(node.targets) == 1, "chained assignment not supported"
        targets, values, rhs_dtypes, rhs_shapes = [], [], None, None
        rhs_visited = False
        if isinstance(node.targets[0], ast.Tuple):
            targets.extend(node.targets[0].elts)
            if isinstance(node.value, ast.Tuple):
                values.extend(node.value.elts)
        else:
            targets.append(node.targets[0])
            values.append(node.value)
        if isinstance(node.value, ast.Call):
            # special case: the builtin get_pid()
            if (
                isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "get_pid"
            ):
                for i, target in enumerate(targets):
                    # TODO: add target symbol for pid??
                    ctx.global_vars[ast.unparse(target)] = ctx.global_vars[f"df.p{i}"]
                    ctx.symbolic[ast.unparse(target)] = f"p{i}"
                return node
            rhs = visit_stmt(ctx, node.value)
            rhs_visited = True
            rhs_dtypes = [rhs.dtype] if len(targets) == 1 else list(rhs.dtype)
            rhs_shapes = [rhs.shape] if len(targets) == 1 else list(rhs.shape)
        assert len(targets) == len(values) or len(targets) == len(rhs_dtypes) == len(
            rhs_shapes
        )
        for idx, target in enumerate(targets):
            if isinstance(target, ast.Name):
                target_ = ctx.get_symbol(target.id, allow_missing=True)
                target_shape, target_dtype = None, None
                if target_ is not None:
                    target_shape, target_dtype = target_.shape, target_.dtype
                if not rhs_visited:
                    rhs = TypeInferer.visit_assignment_val(
                        ctx, values[idx], target_shape, target_dtype
                    )
                    rhs_shape, rhs_dtype = rhs.shape, rhs.dtype
                else:
                    rhs_shape, rhs_dtype = rhs_shapes[idx], rhs_dtypes[idx]
                if target_ is None:
                    # new def
                    ctx.put_symbol(name=target.id, val=target)
                    target.dtype = rhs_dtype
                    target.shape = rhs_shape
                else:
                    # assign
                    target.dtype, target.shape = target_dtype, target_shape
                final_shape, lhs_dims, rhs_dims = TypeInferer.visit_broadcast(
                    ctx, target.shape, rhs_shape=rhs_shape, match_lhs=True
                )
                assert (
                    final_shape == target.shape
                ), f"Shape mismatch, got {final_shape} and {target.shape}"
                node.dims = target.dims = (lhs_dims, rhs_dims)
            else:
                # assign
                if not rhs_visited:
                    rhs = visit_stmt(ctx, values[idx])
                    rhs_shape = None if rhs is None else rhs.shape
                else:
                    rhs_shape = rhs_shapes[idx]
                lhs = visit_stmt(ctx, target)
                final_shape, lhs_dims, rhs_dims = TypeInferer.visit_broadcast(
                    ctx, lhs.shape, rhs_shape=rhs_shape, match_lhs=True
                )
                assert (
                    final_shape == lhs.shape
                ), f"Shape mismatch, got {final_shape} and {lhs.shape}"
                target.dtype, target.shape = lhs.dtype, lhs.shape
                node.dims = target.dims = (lhs_dims, rhs_dims)
            node.dtype, node.shape = target.dtype, target.shape
        return node

    @staticmethod
    def visit_constant_tensor(
        ctx: ASTContext, node: ast.expr, np_values: np.array, dtype: AlloType
    ):
        dtype = str(dtype)
        if is_anywidth_int_type_and_not_np(dtype):
            bitwidth = get_bitwidth_from_type(dtype)
            if bitwidth <= 64:
                np_arr = handle_overflow(np_values, bitwidth, dtype)
                np_values = make_anywidth_numpy_array(np_arr, bitwidth)
        elif dtype in np_supported_types:
            target_np_type = np_supported_types[dtype]
            if np_values.dtype != target_np_type:
                # avoid changing the address of the original array
                np_values = np_values.astype(target_np_type)
        else:
            raise RuntimeError("Unsupported constant tensor element type")
        node.np_values = np_values
        node.shape = np_values.shape
        return node

    @staticmethod
    def visit_AugAssign(ctx: ASTContext, node: ast.AugAssign):
        # visit RHS
        rhs = visit_stmt(ctx, node.value)
        # load LHS
        if isinstance(node.target, ast.Subscript):
            lhs = visit_stmt(ctx, node.target)
        elif isinstance(node.target, ast.Name):  # scalar
            lhs = ctx.get_symbol(node.target.id)
            assert lhs is not None
            node.target.dtype, node.target.shape = lhs.dtype, lhs.shape
        else:
            raise RuntimeError("Unsupported AugAssign")
        # augment LHS
        TypeInferer.visit_general_binop(ctx, node, lhs, rhs)
        # store LHS
        node.dtype = lhs.dtype
        node.shape = lhs.shape
        return node

    @staticmethod
    def visit_symbol(ctx: ASTContext, node: ast.expr):
        if isinstance(node, ast.Name):
            return sympy.symbols(node.id)
        if isinstance(node, ast.Constant):
            return sympy.Integer(node.value)
        if isinstance(node, ast.Attribute):
            assert isinstance(node.value, ast.Name)
            var = ctx.global_vars[node.value.id]
            if node.attr == "bits":
                return sympy.Integer(var.bits)
            if node.attr == "fracs":
                return sympy.Integer(var.fracs)
        if isinstance(node, ast.BinOp):
            lhs = TypeInferer.visit_symbol(ctx, node.left)
            rhs = TypeInferer.visit_symbol(ctx, node.right)
            op = {
                ast.Add: lambda l, r: l + r,
                ast.Sub: lambda l, r: l - r,
                ast.Mult: lambda l, r: l * r,
                ast.Div: lambda l, r: l / r,
                ast.FloorDiv: lambda l, r: l // r,
                ast.Mod: lambda l, r: l % r,
                ast.Pow: lambda l, r: l**r,
                ast.LShift: lambda l, r: l << r,
                ast.RShift: lambda l, r: l >> r,
                ast.BitOr: lambda l, r: l | r,
                ast.BitXor: lambda l, r: l ^ r,
                ast.BitAnd: lambda l, r: l & r,
            }.get(type(node.op))
            return op(lhs, rhs)
        # pylint: disable=raising-bad-type
        raise None

    @staticmethod
    def visit_Subscript(ctx: ASTContext, node: ast.Subscript):
        value = visit_stmt(ctx, node.value)
        # Handle struct field access
        if len(value.shape) == 0 and isinstance(value.dtype, Struct):
            if not isinstance(node.slice, ast.Constant) or not isinstance(
                node.slice.value, str
            ):
                raise RuntimeError("Struct field access must use string literal")
            field = node.slice.value
            if field not in value.dtype.dtype_dict:
                raise RuntimeError(f"Field {field} not found in struct type")
            node.dtype = value.dtype.dtype_dict[field]
            node.shape = tuple()
            return node

        # Handle tensor subscript
        if len(value.shape) > 0:
            visit_stmt(ctx, node.slice)
            # calculate tensor slicing
            shape = []
            # e.g., A[:5, 0, 1:3] -> [(0,5,1),0,(1,3,1)]
            indices = ASTResolver.resolve_slice(node.slice, ctx)
            size = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            elts = (
                size.elts
                if isinstance(size, ast.Tuple)
                else size.dims if isinstance(size, ast.ExtSlice) else [size]
            )
            access_dim = len(elts)
            total_dim = len(value.shape)
            if access_dim < total_dim:  # only access a part of the tensor
                shape = value.shape[access_dim:]
            if isinstance(indices, tuple):  # Slice
                indices = [indices]
            if isinstance(indices, list):  # ExtSlice
                for dim, index in enumerate(indices):
                    if isinstance(index, (list, tuple)):
                        lower = index[0] if index[0] is not None else 0
                        upper = (
                            index[1]
                            if index[1] is not None
                            else ctx.get_symbol(node.value.id).shape[dim]
                        )
                        step = (
                            index[2] if (len(index) > 2 and index[2] is not None) else 1
                        )
                        size = (upper - lower) // step
                        if size > 0:
                            shape.append(size)
            node.shape = tuple(shape)
            node.dtype = ctx.get_symbol(node.value.id).dtype
        elif len(value.shape) == 0 and isinstance(
            value.dtype, (Int, UInt)
        ):  # bit operation
            if isinstance(node.slice, (ast.Index, ast.Constant, ast.Name, ast.BinOp)):
                visit_stmt(ctx, node.slice)
                node.shape = tuple()
                node.dtype = uint1
            elif isinstance(node.slice, ast.Slice):
                lower_sym = TypeInferer.visit_symbol(ctx, node.slice.lower)
                upper_sym = TypeInferer.visit_symbol(ctx, node.slice.upper)
                if (
                    lower_sym is not None
                    and upper_sym is not None
                    and isinstance(upper_sym - lower_sym, sympy.core.numbers.Integer)
                ):
                    stride = int(upper_sym - lower_sym)
                    assert stride > 0, "upper bound must be greater than lower bound"
                    node.dtype = UInt(stride)
                else:
                    warnings.warn(
                        "Cannot infer the bitwidth of the slice, use UInt(32) as default"
                    )
                    node.dtype = UInt(32)
                lower = visit_stmt(ctx, node.slice.lower)
                upper = visit_stmt(ctx, node.slice.upper)
                node.shape = tuple()
            else:
                raise RuntimeError(f"Unsupported bit operation {node.slice}")
        else:
            raise RuntimeError("Can only access bit (slice) for integers")
        return node

    @staticmethod
    def visit_ExtSlice(ctx: ASTContext, node: ast.ExtSlice):
        stmts = visit_stmts(ctx, node.dims)
        node.shape = tuple()
        node.dtype = [stmt.dtype for stmt in stmts]
        return node

    @staticmethod
    def visit_Slice(ctx: ASTContext, node: ast.Slice):
        if node.lower is not None:
            visit_stmt(ctx, node.lower)
        if node.upper is not None:
            visit_stmt(ctx, node.upper)
        if node.step is not None:
            visit_stmt(ctx, node.step)
        node.shape = tuple()
        node.dtype = (Index(), Index(), Index())
        return node

    @staticmethod
    def visit_assignment_val(
        ctx: ASTContext,
        value: ast.expr,
        target_shape: list[int],
        target_dtype: AlloType,
    ):
        if isinstance(value, ast.List):
            assert target_shape is not None and target_dtype is not None
            # e.g., arr: int32[2, 2] = [[1, 2], [3, 4]]
            values = compile(ast.Expression(value), "", "eval")
            # pylint: disable=eval-used
            values = np.array(eval(values, ctx.global_vars))
            assert (
                target_shape == values.shape
            ), f"Shape mismatch, got {target_shape} and {values.shape}"
            TypeInferer.visit_constant_tensor(ctx, value, values, dtype=target_dtype)
            value.dtype = target_dtype
        elif (
            isinstance(value, ast.Name)
            and value.id in ctx.global_vars
            and isinstance(ctx.global_vars[value.id], np.ndarray)
        ):
            assert target_shape is not None and target_dtype is not None
            assert (
                ctx.global_vars[value.id].shape == target_shape
            ), f"`{value.id}` shape mismatch, got {ctx.global_vars[value.id].shape} and {target_shape}"
            TypeInferer.visit_constant_tensor(
                ctx, value, ctx.global_vars[value.id], dtype=target_dtype
            )
            value.dtype = target_dtype
        else:
            visit_stmt(ctx, value)
        return value

    @staticmethod
    def visit_AnnAssign(ctx: ASTContext, node: ast.AnnAssign):
        target_dtype, target_shape, _ = TypeInferer.visit_type_hint(
            ctx, node.annotation
        )
        assert isinstance(
            node.target, ast.Name
        ), "target of AnnAssign must be Name, other type not supported."
        target_ = ctx.get_symbol(node.target.id, allow_missing=True)
        if target_ is not None:
            assert (
                node.value is not None
            ), "Unsupported annotated assignment without a value."
            # assignment
            assert (
                target_.dtype == target_dtype and target_.shape == target_shape
            ), f"Invalid assignment to {node.target.id}, type mismatch."
        # rhs
        rhs = TypeInferer.visit_assignment_val(
            ctx, node.value, target_shape, target_dtype
        )
        if target_ is None:
            # new def
            ctx.put_symbol(name=node.target.id, val=node.target)
        node.target.dtype = node.dtype = target_dtype
        node.target.shape = node.shape = target_shape
        final_shape, lhs_dims, rhs_dims = TypeInferer.visit_broadcast(
            ctx,
            node.target.shape,
            rhs_shape=None if rhs is None else rhs.shape,
            match_lhs=True,
        )
        assert (
            final_shape == target_shape
        ), f"Shape mismatch, got {final_shape} and {target_shape}"
        node.target.dims = node.dims = (lhs_dims, rhs_dims)
        return node

    @staticmethod
    def visit_FunctionDef(ctx: ASTContext, node: ast.FunctionDef):
        # pylint: disable=too-many-nested-blocks
        if ctx.top_func is not None:
            # Nested function def
            # Create a new context to avoid name collision
            old_ctx = ctx
            ctx = old_ctx.copy()
            ctx.scopes = old_ctx.scopes
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute):
                        if decorator.func.attr == "kernel":
                            assert len(decorator.keywords) > 0, "Missing kernel mapping"
                            mapping = eval(
                                ast.unparse(decorator.keywords[0].value),
                                ctx.global_vars,
                            )
                            old_ctx.mapping = mapping
                            orig_name = node.name
                            old_ctx.func_predicate_tags[orig_name] = {}
                            if ctx.unroll:
                                for dim in np.ndindex(*mapping):
                                    new_ctx = old_ctx.copy()
                                    new_ctx.rank = dim
                                    new_ctx.scopes = old_ctx.scopes
                                    new_ctx.global_vars = old_ctx.global_vars.copy()
                                    for axis, val in enumerate(dim):
                                        new_ctx.global_vars.update(
                                            {"df.p" + str(axis): val}
                                        )
                                    node.name = construct_kernel_name(orig_name, dim)
                                    # check on a specific df.kernel instance
                                    TypeInferer.visit_FunctionDef(new_ctx, node)
                                    node.name = orig_name
                            else:
                                # If not unroll, only visit one 'sample' to get the execution predicates

                                def get_predicate_list(predicate_raw, pid_map):
                                    """
                                    Recursively expand `predicate` based on conditions and pid_map.
                                    Returns a list of results, converted from cond_list:
                                    - If condition evaluates True, recurse into its values.
                                    - If False, put None.
                                    """
                                    _, cond_list = predicate_raw
                                    results = []
                                    for cond, val in cond_list:
                                        if eval(cond, pid_map):
                                            assert all(
                                                isinstance(v, tuple) and len(v) == 2
                                                for v in val
                                            )
                                            results.append(
                                                get_predicate_list(
                                                    ("True", val), pid_map
                                                )
                                            )
                                        else:
                                            results.append(None)
                                    return results

                                sample_dim = (0,) * len(mapping)
                                new_ctx = old_ctx.copy()
                                new_ctx.rank = sample_dim
                                new_ctx.scopes = old_ctx.scopes
                                new_ctx.global_vars = old_ctx.global_vars.copy()
                                for axis, val in enumerate(sample_dim):
                                    new_ctx.global_vars.update(
                                        {"df.p" + str(axis): val}
                                    )
                                node.name = construct_kernel_name(orig_name, sample_dim)
                                # check on a specific df.kernel instance
                                TypeInferer.visit_FunctionDef(new_ctx, node)
                                node.name = orig_name
                                for dim in np.ndindex(*mapping):
                                    pid_map = {
                                        f"p{idx}": value
                                        for idx, value in enumerate(dim)
                                    }
                                    old_ctx.func_predicate_tags[orig_name][dim] = (
                                        get_predicate_list(
                                            new_ctx.predicate_list, pid_map
                                        )
                                    )
                            return node
        else:
            old_ctx = None

        # Generic function
        if hasattr(node, "type_params") and len(node.type_params) > 0:
            assert len(ctx.inst) == len(
                node.type_params
            ), f"Type parameters mismatch, got {ctx.inst} and {node.type_params}"
            for type_var, call_val in zip(node.type_params, ctx.inst):
                name, call_val = resolve_generic_types(
                    ctx.global_vars, type_var, call_val
                )
                ctx.global_vars[name] = call_val

        with ctx.block_scope_guard():
            # Input types
            for arg in node.args.args:
                arg.dtype, arg.shape, arg.spec = TypeInferer.visit_type_hint(
                    ctx, arg.annotation
                )
                arg.dtensor = DTensor(
                    ctx.rank, ctx.mapping, arg.shape, arg.dtype, arg.spec, name=arg.arg
                )
                # update shape
                arg.shape = arg.dtensor.get_local_shape()
                assert ctx.get_symbol(name=arg.arg, allow_missing=True) is None, (
                    f"Argument name '{arg.arg}' conflicts with an existing symbol. "
                    f"Please choose a different name to avoid the conflict."
                )
                ctx.put_symbol(name=arg.arg, val=arg)

            func_name = (
                node.name if ctx.func_id is None else f"{node.name}_{ctx.func_id}"
            )
            # Return type
            if not (
                (isinstance(node.returns, ast.Constant) and node.returns.value is None)
                or node.returns is None
            ):
                if isinstance(node.returns, ast.Tuple):
                    # Multiple return values
                    node.returns.shape = []
                    node.returns.dtype = []
                    node.returns.spec = []
                    for elt in node.returns.elts:
                        elt.dtype, elt.shape, elt.spec = TypeInferer.visit_type_hint(
                            ctx, elt
                        )
                        node.returns.dtype += [elt.dtype]
                        node.returns.shape += [elt.shape]
                        node.returns.spec += [elt.spec]
                else:
                    # Single return value
                    node.returns.dtype, node.returns.shape, node.returns.spec = (
                        TypeInferer.visit_type_hint(ctx, node.returns)
                    )
                assert ctx.get_symbol(name=func_name, allow_missing=True) is None, (
                    f"Function name '{func_name}' conflicts with an existing symbol. "
                    f"Please choose a different name to avoid the conflict."
                )
                ctx.put_symbol(name=func_name, val=node)

            # set context
            ctx.top_func = node
            ctx.top_func_tree = node
            visit_stmts(ctx, node.body)
            # Note that the result type may be different from the return type
            if node.returns is None or (
                isinstance(node.returns, ast.Constant) and node.returns.value is None
            ):
                node.dtype = None
                node.shape = None
            else:
                node.dtype = node.returns.dtype
                node.shape = node.returns.shape
        # Recover the old context
        if old_ctx is not None:
            ctx = old_ctx
        # Add the visited function to global variable for later reference
        ctx.global_vars[func_name] = node
        return node

    @staticmethod
    def visit_Compare(ctx: ASTContext, node: ast.Compare):
        lhs = visit_stmt(ctx, node.left)
        assert len(node.comparators) == 1, "Only support one comparator for now"
        rhs = visit_stmt(ctx, node.comparators[0])
        typing_rule = get_typing_rule(type(node.ops[0]))
        res_type = typing_rule(lhs.dtype, rhs.dtype)[0]
        node.dtype = res_type
        node.shape = tuple()
        return node

    @staticmethod
    def visit_BoolOp(ctx: ASTContext, node: ast.BoolOp):
        visit_stmts(ctx, node.values)
        node.dtype = uint1
        node.shape = tuple()
        return node

    @staticmethod
    def visit_IfExp(ctx: ASTContext, node: ast.IfExp):
        visit_stmt(ctx, node.test)
        visit_stmt(ctx, node.body)
        visit_stmt(ctx, node.orelse)
        typing_rule = get_typing_rule(ast.IfExp)
        res_type = typing_rule(node.body.dtype, node.orelse.dtype)
        node.dtype = res_type
        node.shape = node.body.shape
        return node

    @staticmethod
    def visit_If(ctx: ASTContext, node: ast.If):
        visit_stmt(ctx, node.test)
        with ctx.block_scope_guard():
            visit_stmts(ctx, node.body)
        if len(node.orelse) > 0:
            with ctx.block_scope_guard():
                visit_stmts(ctx, node.orelse)
        node.dtype = None
        node.shape = None
        return node

    @staticmethod
    def visit_While(ctx: ASTContext, node: ast.While):
        visit_stmt(ctx, node.test)
        with ctx.block_scope_guard():
            visit_stmts(ctx, node.body)
        if len(node.orelse) > 0:
            raise RuntimeError(
                "'else' clause for 'while' not supported in Allo kernels"
            )
        node.dtype = None
        node.shape = None
        return node

    @staticmethod
    def visit_Module(ctx: ASTContext, node: ast.Module):
        for stmt in node.body:
            visit_stmt(ctx, stmt)
        node.dtype = None
        node.shape = None
        return node

    # pylint: disable=too-many-branches
    @staticmethod
    def visit_Call(ctx: ASTContext, node: ast.Call):
        original_func_id = ctx.func_id
        if isinstance(node.func, ast.Name):
            obj = ASTResolver.resolve(node.func, ctx.global_vars)
            obj_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            obj = ASTResolver.resolve(node.func, ctx.global_vars)
            obj_name = node.func.attr
        elif isinstance(node.func, ast.Subscript):
            obj = ASTResolver.resolve(node.func.value, ctx.global_vars)
            assert obj is not None, "Unsupported function call"
            obj_name = obj.__name__
            ctx.global_vars[obj_name] = obj
            ctx.inst = ASTResolver.resolve_param_types(node.func.slice, ctx.global_vars)
            if ctx.func_id is None:
                func_id = get_func_id_from_param_types(ctx.inst)
                if func_id is None:
                    func_dict = ctx.func_name2id.setdefault(obj_name, {})
                    for key, value in func_dict.items():
                        if value == tuple(ctx.inst):
                            func_id = key
                            break
                    else:
                        func_id = len(func_dict) if len(func_dict) > 0 else None
                        func_dict[func_id] = tuple(ctx.inst)
                else:
                    ctx.inst.remove(func_id)
                    func_dict = ctx.func_name2id.setdefault(obj_name, {})
                    func_dict[func_id] = tuple(ctx.inst)
                ctx.func_id = func_id
        else:
            raise RuntimeError("Unsupported function call")

        if obj is None:
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in {"T", "reverse"}:
                    # x.T or x.reverse
                    assert (
                        len(node.args) == 0
                    ), "Only support zero argument for attribute methods"
                    attr = visit_stmt(ctx, node.func)
                    node.shape = attr.shape
                    node.dtype = attr.dtype
                elif node.func.attr == "put":
                    new_args = visit_stmts(ctx, node.args)
                    node.shape = tuple()
                    node.dtype = None
                    vid = (
                        node.func.value.id
                        if isinstance(node.func.value, ast.Name)
                        else node.func.value.value.id
                    )
                    if isinstance(node.func.value, ast.Subscript):
                        _, loops_to_unroll = get_symbolic_expr(
                            copy.deepcopy(node.func.value.slice),
                            ctx.symbolic,
                            ctx.global_vars,
                            ctx.get_alive_var_names(),
                        )
                        if os.getenv("FORCE_UNROLL_INDEX") == "1":
                            ctx.meta_fors_to_unroll.update(loops_to_unroll)
                    val = ctx.get_symbol(vid)
                    node.func.value.shape = val.dtype.shape
                    node.func.value.dtype = val.dtype.dtype
                elif node.func.attr == "get":
                    vid = (
                        node.func.value.id
                        if isinstance(node.func.value, ast.Name)
                        else node.func.value.value.id
                    )
                    if isinstance(node.func.value, ast.Subscript):
                        _, loops_to_unroll = get_symbolic_expr(
                            copy.deepcopy(node.func.value.slice),
                            ctx.symbolic,
                            ctx.global_vars,
                            ctx.get_alive_var_names(),
                        )
                        if os.getenv("FORCE_UNROLL_INDEX") == "1":
                            ctx.meta_fors_to_unroll.update(loops_to_unroll)
                    # return value
                    val = ctx.get_symbol(vid)
                    node.shape = val.dtype.shape
                    node.dtype = val.dtype.dtype
                    # stream type itself
                    node.func.value.shape = tuple()
                    node.func.value.dtype = val.dtype
                elif node.func.attr == "bitcast":
                    visit_stmt(ctx, node.func.value)
                    # single-element operation
                    node.shape = tuple()
                    if isinstance(node.func.value.dtype, (UInt, Int)):
                        if node.func.value.dtype.bits == 16:
                            node.dtype = float16
                        elif node.func.value.dtype.bits == 32:
                            node.dtype = float32
                        elif node.func.value.dtype.bits == 64:
                            node.dtype = float64
                        else:
                            raise RuntimeError(
                                f"Unsupported bitwidth {node.func.value.dtype.bits}"
                            )
                    else:
                        # casting between signed and unsigned types in C/C++
                        # does not modify the underlying bit representation,
                        # but only the interpretation.
                        node.dtype = UInt(node.func.value.dtype.bits)
                else:
                    raise RuntimeError(
                        f"Unsupported function call or attribute method `.{node.func.attr}`"
                    )
            elif node.func.id in {"float", "int"}:
                # Python-Builtin functions
                assert (
                    len(node.args) == 1
                ), "Only support one argument for `float` and `int`"
                new_args = visit_stmts(ctx, node.args)
                node.shape = tuple()
                node.dtype = float32 if node.func.id == "float" else int32
            elif node.func.id in {"min", "max"}:
                # Python-Builtin functions
                assert (
                    len(node.args) == 2
                ), "Only support two arguments for `min` and `max`"
                new_args = visit_stmts(ctx, node.args)
                typing_rule = get_typing_rule("minmax")
                res_type = typing_rule(new_args[0].dtype, new_args[1].dtype)
                node.dtype = res_type
                node.shape = new_args[0].shape
            else:
                raise RuntimeError(f"Unsupported function call {node.func.id}")
            return node

        # Local imports to avoid cyclic dependencies
        from ..backend.ip import IPModule

        try:
            from ..backend.aie.vliw import VLIWKernelFunction
        except ImportError:

            class VLIWKernelFunction:
                pass

        try:
            from ..backend.aie.external_kernel import ExternalModule
        except ImportError:

            class ExternalModule:
                pass

        if isinstance(obj, VLIWKernelFunction):
            visit_stmts(ctx, node.args)
            node.shape = None
            node.dtype = None
            return node

        if (
            obj.__module__.startswith("allo")
            and not obj.__module__.startswith("allo.library")
            and not obj.__module__.startswith("allo._mlir")
        ):
            # Allo library functions
            if isinstance(obj, IPModule):
                # HLS IP, suppose it does not have return values
                # Also, it has NO side effect, which means it does not change the shape/dtype of the input
                visit_stmts(ctx, node.args)
                node.shape = None
                node.dtype = None
                return node
            if isinstance(obj, ExternalModule):
                # AIE external kernel, suppose it does not have return values
                # Also, it has NO side effect, which means it does not change the shape/dtype of the input
                visit_stmts(ctx, node.args)
                node.shape = None
                node.dtype = None
                return node
            fn_name = obj.__name__
            if fn_name == "gather":
                assert len(node.args) == 1, "Invalid `gather`"
                fifo_list = node.args[0]
                if isinstance(fifo_list, ast.List):
                    sample = None
                    for elt in fifo_list.elts:
                        ele = visit_stmt(ctx, elt)
                        if sample is None:
                            sample = ele
                        else:
                            assert (
                                sample.dtype == ele.dtype and len(ele.shape) == 0
                            ), "Invalid `gather`"
                    fifo_list.dtype = sample.dtype
                    fifo_list.shape = (len(fifo_list.elts),)
                    new_arg = fifo_list
                else:
                    new_arg = visit_stmt(ctx, fifo_list)
                    # flattened
                    new_arg.shape = (np.prod(new_arg.shape),)
                if not isinstance(new_arg.dtype, Stream):
                    raise RuntimeError(
                        f"Unsupported gather dtype {type(new_arg.dtype)}"
                    )
                node.shape = new_arg.shape + new_arg.dtype.shape
                node.dtype = new_arg.dtype.dtype
                return node
            if fn_name == "scatter":
                assert len(node.args) == 2, "Invalid `scatter`"
                buffer, fifo_list = node.args[0], node.args[1]
                buffer_arg = visit_stmt(ctx, buffer)
                if isinstance(fifo_list, ast.List):
                    sample = None
                    for elt in fifo_list.elts:
                        ele = visit_stmt(ctx, elt)
                        if sample is None:
                            sample = ele
                        else:
                            assert (
                                sample.dtype == ele.dtype and len(ele.shape) == 0
                            ), "Invalid `scatter`"
                    fifo_list.dtype = sample.dtype
                    fifo_list.shape = (len(fifo_list.elts),)
                    new_arg = fifo_list
                else:
                    new_arg = visit_stmt(ctx, fifo_list)
                    # flattened
                    new_arg.shape = (np.prod(new_arg.shape),)
                if not isinstance(new_arg.dtype, Stream):
                    raise RuntimeError(
                        f"Unsupported scatter dtype {type(new_arg.dtype)}"
                    )
                node.shape = new_arg.shape + new_arg.dtype.shape
                node.dtype = new_arg.dtype.dtype
                assert buffer_arg.dtype == node.dtype and buffer_arg.shape == node.shape
                return node
            new_args = visit_stmts(ctx, node.args)
            if len(new_args) == 0:
                # No argument
                if fn_name == "get_pid":
                    node.shape = (tuple(), tuple(), tuple())
                    node.dtype = (Index(), Index(), Index())
                else:
                    node.shape = None
                    node.dtype = None
                return node
            if all(len(arg.shape) == 0 for arg in new_args):
                # element-wise operation
                node.shape = tuple()
                node.dtype = new_args[0].dtype
                return node
            # return node
            return TypeInferer.visit_library_op(
                ctx, node=node, op_name=fn_name, new_args=new_args
            )

        # User-defined subfunction
        func = ctx.global_vars[obj_name]
        if isinstance(func, ast.FunctionDef):
            # Has already been defined in the top-level scope
            stmts = [func]
        else:
            # Visit arguments in the top-level
            visit_stmts(ctx, node.args)
            src, starting_line_no = inspect.getsourcelines(func)
            src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
            src = textwrap.dedent("\n".join(src))
            tree = parse_ast(
                src, starting_line_no=starting_line_no, verbose=ctx.verbose
            )
            # Create a new context to avoid name collision
            func_ctx = ctx.copy()
            stmts = visit_stmts(func_ctx, tree.body)
            # Attach type-inferenced tree to the top-level AST
            node.tree = tree
        visit_stmts(ctx, node.args)
        if not isinstance(stmts[-1], ast.FunctionDef):
            node.dtype = None
            node.shape = None
        else:
            node.dtype = stmts[-1].dtype
            node.shape = stmts[-1].shape
        ctx.func_id = original_func_id
        return node

    @staticmethod
    def visit_library_op(
        ctx: ASTContext, node: ast.Call, op_name: str, new_args: list[ast.AST]
    ):
        if op_name in {
            "exp",
            "softmax",
            "abs",
            "log",
            "add",
            "sub",
            "mul",
            "div",
            "relu",
            "copy",
        }:
            # Element-wise operation
            if op_name in {"add", "sub", "mul", "div"}:
                final_shape, lhs_dims, rhs_dims = TypeInferer.visit_broadcast(
                    ctx, new_args[0].shape, new_args[1].shape
                )
                node.dims = (lhs_dims, rhs_dims)
                node.shape = final_shape
            else:
                node.shape = new_args[0].shape
            node.dtype = new_args[0].dtype
            return node
        if op_name in {"matmul", "bmm", "linear", "conv2d", "sumpool", "maxpool"}:
            argAshape = new_args[0].shape
            argBshape = new_args[1].shape
            node.dtype = new_args[0].dtype
            if op_name == "conv2d":
                node.shape = (
                    argAshape[0],
                    argBshape[0],
                    argAshape[2] - argBshape[2] + 1,
                    argAshape[3] - argBshape[3] + 1,
                )
            elif op_name in {"maxpool", "sumpool"}:
                node.shape = (
                    argAshape[0],
                    argAshape[1],
                    argAshape[2] - argBshape[0] + 1,
                    argAshape[3] - argBshape[1] + 1,
                )
            elif op_name == "matmul":
                assert (
                    argAshape[-1] == argBshape[-2]
                ), f"The last dimension of the first input and the second last dimension of the second input must be the same, got {argAshape} and {argBshape}"
                node.shape = tuple(argAshape[:-1] + argBshape[-1:])
            elif op_name == "bmm":
                assert (
                    len(argAshape) == 3 and len(argBshape) == 3
                ), f"Only support batch matrix multiplication of two 3D inputs, got {len(argAshape)} and {len(argBshape)}"
                assert (
                    argAshape[2] == argBshape[1]
                ), f"The third dimension of the first input and the second dimension of the second input must be the same, got {argAshape} and {argBshape}"
                assert (
                    argAshape[0] == argBshape[0]
                ), f"The first dimension of the first input and the first dimension of the second input must be the same, got {argAshape} and {argBshape}"
                node.shape = (argAshape[0], argAshape[1], argBshape[2])
            elif op_name == "linear":
                # The weight parameter (i.e., `new_args[1]`) should be 2D, see:
                # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                assert len(argBshape) == 2
                assert argAshape[-1] == argBshape[-1]
                # bias = True
                if len(new_args) == 3 and new_args[2] is not None:
                    assert argBshape[0] == new_args[2].shape[0]
                node.shape = argAshape[:-1] + argBshape[:-1]
            return node
        if op_name in {"transpose"}:
            assert (
                len(new_args) <= 2
            ), f"Only support zero/one extra argument for {op_name}"
            if len(new_args) == 1:
                node.shape = new_args[0].shape[::-1]
                node.dtype = new_args[0].dtype
            else:
                shape = new_args[0].shape
                axes = compile(ast.Expression(new_args[1]), "", "eval")
                # pylint: disable=eval-used
                axes = eval(axes)
                assert len(shape) == len(
                    axes
                ), f"Transpose shape mismatch, should provide the same number of dimensions as the input, got {len(shape)} and {axes}"
                new_shape = [shape[new_dim] for new_dim in axes]
                node.shape = tuple(new_shape)
                node.dtype = new_args[0].dtype
            return node
        if op_name in {"view"}:
            axes = compile(ast.Expression(new_args[1]), "", "eval")
            # pylint: disable=eval-used
            axes = eval(axes)
            node.shape = axes
            node.dtype = new_args[0].dtype
            return node
        if op_name in {"layernorm", "gelu", "tril"}:
            node.shape = new_args[0].shape
            node.dtype = new_args[0].dtype
            return node
        if op_name in {"ones", "zeros"}:
            axes = compile(ast.Expression(new_args[0]), "", "eval")
            # pylint: disable=eval-used
            axes = eval(axes)
            node.shape = axes
            assert (
                node.keywords[0].arg == "dtype"
            ), f"Only support `dtype` keyword argument for {op_name}"
            dtype = node.keywords[0].value.id
            if dtype.startswith("int"):
                node.dtype = int32
            elif dtype.startswith("float"):
                node.dtype = float32
            else:
                raise RuntimeError(f"Unsupported dtype {dtype}")
            return node
        if op_name == "concat":
            axis = node.keywords[0].value.value
            if len(new_args[0].shape) != len(new_args[1].shape):
                raise RuntimeError(
                    f"Concatenation requires the same number of dimensions, got {len(new_args[0].shape)} and {len(new_args[1].shape)}"
                )

            for i, (shape1, shape2) in enumerate(
                zip(new_args[0].shape, new_args[1].shape)
            ):
                if i != axis and shape1 != shape2:
                    raise RuntimeError(
                        f"Concatenation requires the same shape except the concatenation axis {axis}, got {new_args[0].shape} and {new_args[1].shape}"
                    )
            shape = list(new_args[0].shape)
            shape[axis] += new_args[1].shape[axis]
            node.shape = tuple(shape)
            node.dtype = new_args[0].dtype
            return node
        raise RuntimeError(f"Unsupported linalg operation {op_name}")

    @staticmethod
    def visit_Return(ctx: ASTContext, node: ast.Return):
        res = visit_stmt(ctx, node.value)
        node.dtype = res.dtype if res is not None else None
        node.shape = res.shape if res is not None else None
        return node

    @staticmethod
    def visit_With(ctx: ASTContext, node: ast.With):
        """
        Generate 'predicate tag' here to classigfy kernel instances with different control flow
        """
        assert len(node.items) == 1, "Only support one context manager"
        assert isinstance(
            node.items[0].context_expr, ast.Call
        ), "Only support `with allo.meta_if/elif/else()`"
        assert isinstance(
            node.items[0].context_expr.func, ast.Attribute
        ), "Only support `with allo.meta_if/elif/else()`"
        # Compile-time comparison
        if node.items[0].context_expr.func.attr in {"meta_if", "meta_elif"}:
            cond = ASTResolver.resolve_constant(node.items[0].context_expr.args[0], ctx)
            symbolic_cond, loops_to_unroll = get_symbolic_expr(
                copy.deepcopy(node.items[0].context_expr.args[0]),
                ctx.symbolic,
                ctx.global_vars,
                ctx.get_alive_var_names(),
            )
            ctx.meta_fors_to_unroll.update(loops_to_unroll)
            if node.items[0].context_expr.func.attr == "meta_if":
                final_cond = cond
                if len(ctx.meta_if_stack) > ctx.with_scope_level:
                    ctx.meta_if_stack[ctx.with_scope_level].append(final_cond)
                    if not ctx.unroll:
                        ctx.raw_meta_if_stack[ctx.with_scope_level].append(
                            symbolic_cond
                        )
                else:
                    # create a new nested list
                    ctx.meta_if_stack.append([final_cond])
                    if not ctx.unroll:
                        ctx.raw_meta_if_stack.append([symbolic_cond])
            else:  # meta_elif
                assert (
                    len(ctx.meta_if_stack[ctx.with_scope_level]) > 0
                ), "Unmatched allo.meta_elif()"
                if not ctx.unroll:
                    prev_cond = ctx.raw_meta_if_stack[ctx.with_scope_level][-1]
                    symbolic_cond = f"(not ({prev_cond})) and {symbolic_cond}"
                    ctx.raw_meta_if_stack[ctx.with_scope_level].append(symbolic_cond)
                if ctx.meta_if_stack[ctx.with_scope_level][
                    -1
                ]:  # previous `if` has already satisfied
                    ctx.meta_if_stack[ctx.with_scope_level].pop()
                    ctx.meta_if_stack[ctx.with_scope_level].append(True)
                    final_cond = False
                else:
                    ctx.meta_if_stack[ctx.with_scope_level].pop()
                    ctx.meta_if_stack[ctx.with_scope_level].append(cond)
                    final_cond = cond
        elif node.items[0].context_expr.func.attr == "meta_else":
            assert (
                len(ctx.meta_if_stack[ctx.with_scope_level]) > 0
            ), "Unmatched allo.meta_else()"
            final_cond = not ctx.meta_if_stack[ctx.with_scope_level][-1]
            if not ctx.unroll:
                symbolic_cond = (
                    f"(not ({ctx.raw_meta_if_stack[ctx.with_scope_level][-1]}))"
                )
            ctx.meta_if_stack[ctx.with_scope_level].pop()
        elif node.items[0].context_expr.func.attr == "meta_for":
            assert (
                len(node.items[0].context_expr.args) <= 3
            ), "Only support three arguments (lower, upper bound, and step) for `allo.meta_for()`"
            rargs = [
                ASTResolver.resolve_constant(node.items[0].context_expr.args[0], ctx)
            ]
            if len(node.items[0].context_expr.args) > 1:
                rargs.append(
                    ASTResolver.resolve_constant(
                        node.items[0].context_expr.args[1], ctx
                    )
                )
            if len(node.items[0].context_expr.args) > 2:
                rargs.append(
                    ASTResolver.resolve_constant(
                        node.items[0].context_expr.args[2], ctx
                    )
                )
            var = node.items[0].optional_vars.id
            for i in range(*rargs):
                ctx.global_vars[var] = i
                ctx.symbolic[var] = (i, node)
                with ctx.block_scope_guard():
                    visit_stmts(ctx, node.body)
                ctx.global_vars.pop(var)
                ctx.symbolic.pop(var)
                if not ctx.unroll and node not in ctx.meta_fors_to_unroll:
                    break
            node.dtype = None
            node.shape = None
            return node
        else:
            raise RuntimeError("Unsupported meta function")
        if ctx.unroll and final_cond:
            ctx.with_scope_level += 1
            with ctx.block_scope_guard():
                visit_stmts(ctx, node.body)
            # clear inner context
            ctx.meta_if_stack = ctx.meta_if_stack[: ctx.with_scope_level]
            ctx.with_scope_level -= 1
        elif not ctx.unroll:
            # if not unroll, walk into every branch unconditionally
            assert ctx.predicate_stack[-1] is not None
            ctx.predicate_stack[-1].append(tuple((symbolic_cond, [])))
            ctx.with_scope_level += 1
            ctx.predicate_stack.append(ctx.predicate_stack[-1][-1][1])
            with ctx.block_scope_guard():
                visit_stmts(ctx, node.body)
            ctx.meta_if_stack = ctx.meta_if_stack[: ctx.with_scope_level]
            ctx.predicate_stack = ctx.predicate_stack[: ctx.with_scope_level]
            ctx.with_scope_level -= 1
        node.dtype = None
        node.shape = None
        return node

    @staticmethod
    def visit_Expr(ctx: ASTContext, node: ast.Expr):
        if isinstance(node.value, ast.Constant):
            # Python comments
            node.dtype = None
            node.shape = None
            return node
        if isinstance(node.value, ast.Call):
            visit_stmt(ctx, node.value)
            node.dtype = None
            node.shape = None
            return node
        raise RuntimeError(f"Unsupported expression: {node.value}")

    @staticmethod
    def visit_Pass(ctx: ASTContext, node: ast.Pass):
        node.dtype = None
        node.shape = None
        return node


visit_stmt = TypeInferer()


def visit_stmts(ctx: ASTContext, stmts: list[ast.expr]):
    results = []
    for stmt in stmts:
        try:
            results.append(visit_stmt(ctx, stmt))
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"{traceback.format_exc()}")
            print_error_message(str(e), stmt, ctx.top_func_tree)
            sys.exit(1)
    return results
