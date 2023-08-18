# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .utils import MockConstant
from .visitor import ASTVisitor
from .symbol_resolver import ASTResolver
from .types import int32, float32


class TypeInferer(ASTVisitor):
    @staticmethod
    def visit_type_hint(node, ctx):
        if isinstance(node, ast.Subscript):
            dtype = ASTResolver.resolve(node.value, ctx.global_vars)
            assert dtype is not None, f"Unsupported type {node.value.id}"
            size = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            elts = size.elts if isinstance(size, ast.Tuple) else [size]
            shape = [
                x.value if isinstance(x, ast.Constant) else ctx.global_vars[x.id]
                for x in elts
            ]
            return dtype, shape
        if isinstance(node, ast.Name):
            dtype = ASTResolver.resolve(node, ctx.global_vars)
            assert dtype is not None, f"Unsupported type {node.id}"
            return dtype, []
        raise RuntimeError("Unsupported function argument type")

    @staticmethod
    def visit_Name(ctx, node):
        if node.id in ctx.buffers:
            return ctx.buffers[node.id]
        if node.id in ctx.global_vars:
            return MockConstant(ctx.global_vars[node.id], ctx)
        raise RuntimeError("Unsupported Name")

    @staticmethod
    def visit_Constant(ctx, node):
        node.shape = []
        if isinstance(node.value, int):
            node.dtype = int32
        if isinstance(node.value, float):
            node.dtype = float32
        return node

    @staticmethod
    def visit_all_for(ctx, node):
        return visit_stmts(ctx, node.body)

    @staticmethod
    def visit_For(ctx, node):
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
    def visit_general_binop(ctx, node, lhs, rhs):
        # TODO: Add type casting
        assert lhs.shape == rhs.shape, "Shape mismatch"
        assert lhs.dtype == rhs.dtype, "Type mismatch"
        node.dtype = lhs.dtype
        node.shape = lhs.shape
        return node

    @staticmethod
    def visit_UnaryOp(ctx, node):
        node.shape = []
        if isinstance(node.operand.dtype, int):
            node.dtype = int32
        if isinstance(node.operand.dtype, float):
            node.dtype = float32
        return node

    @staticmethod
    def visit_BinOp(ctx, node):
        lhs = visit_stmt(ctx, node.left)
        rhs = visit_stmt(ctx, node.right)
        return TypeInferer.visit_general_binop(ctx, node, lhs, rhs)

    @staticmethod
    def visit_store(ctx, node, val):
        if isinstance(node, ast.Subscript):
            return ctx.buffers[node.value.id]
        if isinstance(node, ast.Name):
            return ctx.buffers[node.id]
        raise RuntimeError("Unsupported store")

    @staticmethod
    def visit_Assign(ctx, node):
        # Compute RHS
        if isinstance(node.value, ast.Name):  # scalar
            rhs = ctx.buffers[node.value.id]
        else:
            rhs = visit_stmt(ctx, node.value)
        # store LHS
        lhs = TypeInferer.visit_store(ctx, node.targets[0], rhs)
        node.dtype = lhs.dtype
        node.shape = lhs.shape
        return node

    @staticmethod
    def visit_AugAssign(ctx, node):
        # visit RHS
        rhs = visit_stmt(ctx, node.value)
        # load LHS
        if isinstance(node.target, ast.Subscript):
            lhs = visit_stmt(ctx, node.target)
        elif isinstance(node.target, ast.Name):  # scalar
            lhs = ctx.buffers[node.target.id]
        else:
            raise RuntimeError("Unsupported AugAssign")
        # augment LHS
        res = TypeInferer.visit_general_binop(ctx, node, lhs, rhs)
        # store LHS
        lhs = TypeInferer.visit_store(ctx, node.target, res)
        node.dtype = lhs.dtype
        node.shape = lhs.shape
        return node

    @staticmethod
    def visit_Subscript(ctx, node):
        # TODO: Suppose only load a single element, this is not true if tensor slicing is added
        node.shape = []
        node.dtype = ctx.buffers[node.value.id].dtype
        return node

    @staticmethod
    def visit_AnnAssign(ctx, node):
        node.dtype, node.shape = TypeInferer.visit_type_hint(node.annotation, ctx)
        ctx.buffers[node.target.id] = node
        return node

    @staticmethod
    def visit_FunctionDef(ctx, node):
        # Input types
        for arg in node.args.args:
            arg.dtype, arg.shape = TypeInferer.visit_type_hint(arg.annotation, ctx)
            ctx.buffers[arg.arg] = arg

        # Return type
        if not (
            (isinstance(node.returns, ast.Constant) and node.returns.value is None)
            or node.returns is None
        ):
            node.returns.dtype, node.returns.shape = TypeInferer.visit_type_hint(
                node.returns, ctx
            )
            ctx.buffers[node.name] = node

        visit_stmts(ctx, node.body)
        return node

    @staticmethod
    def visit_Compare(ctx, node, is_affine=False):
        pass

    @staticmethod
    def visit_If(ctx, node, is_affine=False):
        pass

    @staticmethod
    def visit_Module(ctx, node):
        for stmt in node.body:
            visit_stmt(ctx, stmt)
        return node

    @staticmethod
    def visit_Return(ctx, node):
        res = visit_stmt(ctx, node.value)
        node.dtype = res.dtype
        node.shape = res.shape
        return node

    @staticmethod
    def visit_Expr(ctx, node):
        visit_stmt(ctx, node.value)
        return node

    @staticmethod
    def visit_Pass(ctx, node):
        pass


visit_stmt = TypeInferer()


def visit_stmts(ctx, stmts):
    results = []
    for stmt in stmts:
        results.append(visit_stmt(ctx, stmt))
    return results
