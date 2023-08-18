# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import textwrap

from .utils import MockConstant
from .visitor import ASTVisitor, ASTContext
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
            shape = tuple(
                x.value if isinstance(x, ast.Constant) else ctx.global_vars[x.id]
                for x in elts
            )
            return dtype, shape
        if isinstance(node, ast.Name):
            dtype = ASTResolver.resolve(node, ctx.global_vars)
            assert dtype is not None, f"Unsupported type {node.id}"
            return dtype, tuple()
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
        node.shape = tuple()
        if isinstance(node.value, int):
            node.dtype = int32
        if isinstance(node.value, float):
            node.dtype = float32
        return node

    @staticmethod
    def visit_all_for(ctx, node):
        visit_stmts(ctx, node.body)
        return node

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
        assert (
            lhs.shape == rhs.shape
        ), f"Shape mismatch, got {lhs.shape} and {rhs.shape}"
        assert lhs.dtype == rhs.dtype, f"Type mismatch, got {lhs.dtype} and {rhs.dtype}"
        node.dtype = lhs.dtype
        node.shape = lhs.shape
        return node

    @staticmethod
    def visit_UnaryOp(ctx, node):
        node.shape = tuple()
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
        if len(node.targets) > 1:
            raise RuntimeError("Cannot assign to multiple targets")
        if isinstance(rhs, ast.Call):
            if len(node.targets) > 1:
                raise RuntimeError("Cannot support multiple results yet")
            ctx.buffers[node.targets[0].id] = rhs
            return rhs
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
        node.shape = tuple()
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

        stmts = visit_stmts(ctx, node.body)
        if not isinstance(stmts[-1], ast.Return):
            node.dtype = None
            node.shape = None
        else:
            node.dtype = stmts[-1].dtype
            node.shape = stmts[-1].shape
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
    def visit_Call(ctx, node):
        obj = ASTResolver.resolve(node.func, ctx.global_vars)
        if obj is None:
            # Python-Builtin functions
            assert (
                len(node.args) == 1
            ), "Only support one argument for `float` and `int`"
            if node.func.id == "float":
                node.dtype = float32
                node.shape = tuple()
            if node.func.id == "int":
                node.dtype = int32
                node.shape = tuple()
            raise RuntimeError(f"Cannot resolve function `{node.func.id}`")

        if obj.__module__.startswith("allo"):
            # Allo library functions
            new_args = [stmt for stmt in visit_stmts(ctx, node.args)]
            fn_name = obj.__name__
            if len(new_args[0].shape) == 0:
                # element-wise operation
                node.shape = tuple()
                node.dtype = new_args[0].dtype
            return TypeInferer.visit_linalg_op(
                ctx, node=node, op_name=fn_name, new_args=new_args
            )

        # Visit arguments in the top-level
        visit_stmts(ctx, node.args)
        # User-defined subfunction
        func = ctx.global_vars[node.func.id]
        src, _ = inspect.getsourcelines(func)
        src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
        src = textwrap.dedent("\n".join(src))
        tree = ast.parse(src)
        # Create a new context to avoid name collision
        func_ctx = ASTContext(
            global_vars=ctx.global_vars,
            mlir_ctx=ctx.mlir_ctx,
            enable_tensor=ctx.enable_tensor,
        )
        stmts = visit_stmts(func_ctx, tree.body)
        # Attach type-inferenced tree to the top-level AST
        node.tree = tree
        if not isinstance(stmts[-1], ast.Return):
            node.dtype = None
            node.shape = None
        else:
            node.dtype = stmts[-1].dtype
            node.shape = stmts[-1].shape
        return node

    @staticmethod
    def visit_linalg_op(ctx, node, op_name, new_args):
        if op_name in {"exp", "softmax", "abs", "log", "add", "sub", "div"}:
            # Element-wise operation
            if op_name in {"add", "sub", "div"}:
                assert (
                    new_args[0].shape == new_args[1].shape
                ), f"Only support element-wise {op_name} of two inputs with the same shape, got {new_args[0].shape} and {new_args[1].shape}"
            node.shape = new_args[0].shape
            node.dtype = new_args[0].dtype
            return node
        if op_name in {"matmul", "bmm"}:
            if op_name == "matmul":
                argAshape = new_args[0].shape
                argBshape = new_args[1].shape
                assert (
                    len(argAshape) == 2 and len(argBshape) == 2
                ), f"Only support matrix multiplication of two 2D inputs, got {len(argAshape)} and {len(argBshape)}"
                assert (
                    argAshape[1] == argBshape[0]
                ), f"The second dimension of the first input and the first dimension of the second input must be the same, got {argAshape[1]} and {argBshape[0]}"
                node.shape = (argAshape[0], argBshape[1])
                node.dtype = new_args[0].dtype
            if op_name == "bmm":
                assert (
                    len(argAshape) == 3 and len(argBshape) == 3
                ), f"Only support batch matrix multiplication of two 3D inputs, got {len(argAshape)} and {len(argBshape)}"
                assert (
                    argAshape[2] == argBshape[1]
                ), f"The third dimension of the first input and the second dimension of the second input must be the same, got {argAshape[2]} and {argBshape[1]}"
                assert (
                    argAshape[0] == argBshape[0]
                ), f"The first dimension of the first input and the first dimension of the second input must be the same, got {argAshape[0]} and {argBshape[0]}"
                node.shape = (argAshape[0], argAshape[1], argBshape[2])
                node.dtype = new_args[0].dtype
            return node
        raise RuntimeError(f"Unsupported linalg operation {op_name}")

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
