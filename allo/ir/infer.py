# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .symbol_resolver import ASTResolver


class Visitor:
    def __call__(self, ctx, node):
        method = getattr(self, "visit_" + node.__class__.__name__, None)
        if method is None:
            error_msg = f'Unsupported node "{node.__class__.__name__}"'
            raise RuntimeError(error_msg)
        return method(ctx, node)


class TypeInferer(Visitor):
    @staticmethod
    def visit_Name(ctx, node):
        pass

    @staticmethod
    def visit_Constant(ctx, node):
        pass

    @staticmethod
    def visit_range_for(ctx, node):
        pass

    @staticmethod
    def visit_grid_for(ctx, node):
        pass

    @staticmethod
    def visit_For(ctx, node):
        if node.orelse:
            raise RuntimeError("'else' clause for 'for' not supported in Allo kernels")
        with ctx.loop_scope_guard():
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ):
                return TypeInferer.visit_range_for(ctx, node)
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute)
                and (node.iter.func.attr in {"grid", "reduction"})
            ):
                return TypeInferer.visit_grid_for(ctx, node)
            raise RuntimeError("Unsupported for loop")

    @staticmethod
    def visit_general_binop(ctx, node, lhs, rhs):
        pass

    @staticmethod
    def visit_UnaryOp(ctx, node):
        pass

    @staticmethod
    def visit_BinOp(ctx, node):
        pass

    @staticmethod
    def visit_Assign(ctx, node):
        pass

    @staticmethod
    def visit_AugAssign(ctx, node):
        pass

    @staticmethod
    def visit_Subscript(ctx, node):
        pass

    @staticmethod
    def visit_AnnAssign(ctx, node):
        pass

    @staticmethod
    def visit_FunctionDef(ctx, node):
        print(ctx.global_vars)

        def visit_type(arg, type_hint):
            if isinstance(type_hint, ast.Subscript):
                dtype = ASTResolver.resolve(type_hint.value, ctx.global_vars)
                assert dtype is not None, f"Unsupported type {type_hint.value.id}"
                arg.dtype = dtype
                # pylint: disable=redefined-builtin
                slice = (
                    type_hint.slice.value
                    if isinstance(type_hint.slice, ast.Index)
                    else type_hint.slice
                )
                elts = slice.elts if isinstance(slice, ast.Tuple) else [slice]
                arg.shape = [
                    x.value if isinstance(x, ast.Constant) else ctx.global_vars[x.id]
                    for x in elts
                ]
            elif isinstance(type_hint, ast.Name):
                dtype = ASTResolver.resolve(type_hint, ctx.global_vars)
                assert dtype is not None, f"Unsupported type {type_hint.id}"
                arg.dtype = dtype
                arg.shape = []
            else:
                raise RuntimeError("Unsupported function argument type")

        # Input types
        for arg in node.args.args:
            visit_type(arg, arg.annotation)
            ctx.buffers[arg.arg] = arg

        # Return type
        if not (
            (isinstance(node.returns, ast.Constant) and node.returns.value is None)
            or node.returns is None
        ):
            visit_type(node, node.returns)
            ctx.buffers[node.name] = node

        visit_stmts(ctx, node.body)

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

    @staticmethod
    def visit_Return(ctx, node):
        ret = visit_stmt(ctx, node.value)

    @staticmethod
    def visit_Expr(ctx, node):
        return visit_stmt(ctx, node.value)

    @staticmethod
    def visit_Pass(ctx, node):
        return None


visit_stmt = TypeInferer()


def visit_stmts(ctx, stmts):
    results = []
    for stmt in stmts:
        results.append(visit_stmt(ctx, stmt))
    return results
