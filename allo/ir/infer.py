# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

import ast
import inspect
import textwrap
import numpy as np

from .visitor import ASTVisitor, ASTContext
from .symbol_resolver import ASTResolver
from .types import Fixed, UFixed, int1, int32, float32
from .typing_rule import get_typing_rule


# pylint: disable=too-many-public-methods
class TypeInferer(ASTVisitor):
    def print_verbose(self, ctx, node):
        print(node.__class__.__name__, node.dtype, node.shape)

    @staticmethod
    def visit_call_type(ctx, node):
        ty_cls = ASTResolver.resolve(node.func, ctx.global_vars)
        args = node.args
        if ty_cls is Fixed or ty_cls is UFixed:
            assert len(args) == 2
            assert isinstance(args[0], ast.Constant)
            assert isinstance(args[1], ast.Constant)
            dtype = ty_cls(args[0].value, args[1].value)
        else:
            assert len(args) == 1
            assert isinstance(args[0], ast.Constant)
            dtype = ty_cls(args[0].value)
        return dtype

    @staticmethod
    def visit_type_hint(ctx, node):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Call):
                dtype = TypeInferer.visit_call_type(ctx, node.value)
            else:
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
        if isinstance(node, ast.Call):
            dtype = TypeInferer.visit_call_type(ctx, node)
            return dtype, tuple()
        raise RuntimeError("Unsupported function argument type")

    @staticmethod
    def visit_Name(ctx, node):
        if node.id in ctx.buffers:
            var = ctx.buffers[node.id]
            node.dtype = var.dtype
            node.shape = var.shape
            return node
        if node.id in ctx.global_vars:
            if isinstance(ctx.global_vars[node.id], int):
                node.dtype = int32
            elif isinstance(ctx.global_vars[node.id], float):
                node.dtype = float32
            node.shape = tuple()
            return node
        raise RuntimeError("Unsupported Name")

    @staticmethod
    def visit_Constant(ctx, node):
        node.shape = tuple()
        if isinstance(node.value, int):
            node.dtype = int32
        elif isinstance(node.value, float):
            node.dtype = float32
        else:
            raise RuntimeError("Unsupported constant type")
        return node

    @staticmethod
    def visit_Attribute(ctx, node):
        if node.attr == "T":
            res = visit_stmt(ctx, node.value)
            node.dtype = res.dtype
            node.shape = res.shape[::-1]
        return node

    @staticmethod
    def visit_all_for(ctx, node):
        visit_stmts(ctx, node.body)
        node.shape = None
        node.dtype = None
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
        assert (
            lhs.shape == rhs.shape
        ), f"Shape mismatch, got {lhs.shape} and {rhs.shape}"
        typing_rule = get_typing_rule(type(node.op))
        res_type = typing_rule(lhs.dtype, rhs.dtype)
        node.dtype = res_type
        node.shape = lhs.shape
        return node

    @staticmethod
    def visit_UnaryOp(ctx, node):
        node.shape = tuple()
        if isinstance(node.operand.dtype, int):
            node.dtype = int32
        elif isinstance(node.operand.dtype, float):
            node.dtype = float32
        else:
            raise RuntimeError("Unsupported constant type")
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
        rhs = visit_stmt(ctx, node.value)
        if len(node.targets) > 1:
            raise RuntimeError("Cannot assign to multiple targets")
        if isinstance(rhs, ast.Call) or len(rhs.shape) > 0:
            if len(node.targets) > 1:
                raise RuntimeError("Cannot support multiple results yet")
            if isinstance(node.targets[0], ast.Name):
                ctx.buffers[node.targets[0].id] = rhs
                node.dtype = rhs.dtype
                node.shape = rhs.shape
                return rhs
        # store LHS
        lhs = visit_stmt(ctx, node.targets[0])
        node.dtype = lhs.dtype
        node.shape = lhs.shape
        return node

    @staticmethod
    def visit_constant_tensor(ctx, node):
        if isinstance(node.value, ast.Name):
            values = ctx.global_vars[node.value.id]
        elif isinstance(node.value, ast.List):
            values = compile(ast.Expression(node.value), "", "eval")
            # pylint: disable=eval-used
            values = eval(values)
        else:
            raise RuntimeError("Unsupported type")
        np_values = np.asarray(values)
        if np.issubdtype(np_values.dtype, np.integer):
            node.dtype = int32
            np_values = np_values.astype(np.int32)
        elif np.issubdtype(np_values.dtype, np.floating):
            node.dtype = float32
            np_values = np_values.astype(np.float32)
        else:
            raise RuntimeError("Unsupported constant tensor element type")
        node.np_values = np_values
        node.shape = np_values.shape
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
        TypeInferer.visit_general_binop(ctx, node, lhs, rhs)
        # store LHS
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
        target_dtype, target_shape = TypeInferer.visit_type_hint(ctx, node.annotation)
        if node.value is not None:
            if (
                isinstance(node.value, ast.Name) and node.value.id in ctx.buffers
            ) or isinstance(node.value, (ast.Constant, ast.Call)):
                # Examples:
                # copied: int32 = a
                # init: int32 = 0
                # call: int32 = int(1)
                rhs = visit_stmt(ctx, node.value)
            elif isinstance(node.value, (ast.List, ast.Name)):
                rhs = TypeInferer.visit_constant_tensor(ctx, node)
            else:
                raise RuntimeError("Unsupported data type")
            if not isinstance(node.value, ast.Constant):
                assert (
                    rhs.shape == target_shape
                ), f"Shape mismatch, got {rhs.shape} and {target_shape} for {node.__class__.__name__} `{node.target.id}`"
        else:
            rhs = None
        ctx.buffers[node.target.id] = node
        node.dtype = target_dtype
        node.shape = target_shape
        return node

    @staticmethod
    def visit_FunctionDef(ctx, node):
        if ctx.top_func is not None:
            # Nested function def
            # Create a new context to avoid name collision
            old_ctx = ctx
            ctx = ASTContext(
                global_vars=ctx.global_vars,
                mlir_ctx=old_ctx.mlir_ctx,
                enable_tensor=old_ctx.enable_tensor,
                verbose=old_ctx.verbose,
            )
        else:
            old_ctx = None
        # Input types
        for arg in node.args.args:
            arg.dtype, arg.shape = TypeInferer.visit_type_hint(ctx, arg.annotation)
            ctx.buffers[arg.arg] = arg

        # Return type
        if not (
            (isinstance(node.returns, ast.Constant) and node.returns.value is None)
            or node.returns is None
        ):
            node.returns.dtype, node.returns.shape = TypeInferer.visit_type_hint(
                ctx, node.returns
            )
            ctx.buffers[node.name] = node

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
        ctx.global_vars[node.name] = node
        return node

    @staticmethod
    def visit_Compare(ctx, node):
        visit_stmt(ctx, node.left)
        visit_stmt(ctx, node.comparators[0])
        node.dtype = int1
        node.shape = tuple()
        return node

    @staticmethod
    def visit_If(ctx, node):
        visit_stmts(ctx, node.body)
        if len(node.orelse) > 0:
            visit_stmts(ctx, node.orelse)
        node.dtype = None
        node.shape = None
        return node

    @staticmethod
    def visit_Module(ctx, node):
        for stmt in node.body:
            visit_stmt(ctx, stmt)
        node.dtype = None
        node.shape = None
        return node

    @staticmethod
    def visit_Call(ctx, node):
        obj = ASTResolver.resolve(node.func, ctx.global_vars)
        if obj is None:
            # Python-Builtin functions
            assert (
                len(node.args) == 1
            ), "Only support one argument for `float` and `int`"
            new_args = visit_stmts(ctx, node.args)
            if node.func.id == "float":
                node.dtype = float32
                node.shape = tuple()
            elif node.func.id == "int":
                node.dtype = int32
                node.shape = tuple()
            else:
                raise RuntimeError(f"Cannot resolve function `{node.func.id}`")
            return node

        if obj.__module__.startswith("allo"):
            # Allo library functions
            new_args = visit_stmts(ctx, node.args)
            fn_name = obj.__name__
            if len(new_args[0].shape) == 0:
                # element-wise operation
                node.shape = tuple()
                node.dtype = new_args[0].dtype
                return node
            return TypeInferer.visit_linalg_op(
                ctx, node=node, op_name=fn_name, new_args=new_args
            )

        # User-defined subfunction
        func = ctx.global_vars[node.func.id]
        if isinstance(func, ast.FunctionDef):
            # Has already been defined in the top-level scope
            stmts = [func]
        else:
            # Visit arguments in the top-level
            visit_stmts(ctx, node.args)
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
                verbose=ctx.verbose,
            )
            stmts = visit_stmts(func_ctx, tree.body)
            # Attach type-inferenced tree to the top-level AST
            node.tree = tree
        if not isinstance(stmts[-1], ast.FunctionDef):
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
            argAshape = new_args[0].shape
            argBshape = new_args[1].shape
            if op_name == "matmul":
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
        node.dtype = None
        node.shape = None
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
