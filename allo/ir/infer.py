# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

import ast
import inspect
import textwrap
import numpy as np

from .visitor import ASTVisitor, ASTContext
from .symbol_resolver import ASTResolver
from .types import Int, UInt, Fixed, UFixed, Index, uint1, int32, float32
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
            dtype = ty_cls(
                ASTResolver.resolve_constant(args[0], ctx),
                ASTResolver.resolve_constant(args[1], ctx),
            )
        else:
            assert len(args) == 1
            dtype = ty_cls(ASTResolver.resolve_constant(args[0], ctx))
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
                node.shape = tuple()
            elif isinstance(ctx.global_vars[node.id], float):
                node.dtype = float32
                node.shape = tuple()
            return node
        raise RuntimeError(f"Unsupported Name {node.id}")

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
    def visit_Tuple(ctx, node):
        visit_stmts(ctx, node.elts)
        node.shape = [elt.shape for elt in node.elts]
        node.dtype = [elt.dtype for elt in node.elts]
        return node

    @staticmethod
    def visit_Index(ctx, node):
        value = visit_stmt(ctx, node.value)
        node.shape = value.shape
        node.dtype = value.dtype
        return node

    @staticmethod
    def visit_Attribute(ctx, node):
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
        raise RuntimeError(f"Unsupported attribute `{node.attr}`")

    @staticmethod
    def visit_all_for(ctx, node):
        # Set loop induction variables
        if isinstance(node.target, ast.Tuple):
            ivs = list(node.target.elts)
        else:
            ivs = [node.target]
        for iv in ivs:
            iv.shape = tuple()
            iv.dtype = Index()
            ctx.buffers[iv.id] = iv
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
        typing_rule = get_typing_rule(type(node.op))
        res_type = typing_rule(lhs.dtype, rhs.dtype)
        node.dtype = res_type
        # See the broadcasting rules in NumPy
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        # When operating on two arrays, NumPy compares their shapes element-wise.
        # It starts with the trailing (i.e. rightmost) dimension and works its way left.
        # Two dimensions are compatible when
        # 1. they are equal, or
        # 2. one of them is 1.
        tmp_lhs_shape = list(lhs.shape)
        tmp_rhs_shape = list(rhs.shape)
        # match larger shape
        if len(tmp_lhs_shape) < len(tmp_rhs_shape):
            tmp_lhs_shape = [1] * (
                len(tmp_rhs_shape) - len(tmp_lhs_shape)
            ) + tmp_lhs_shape
        elif len(tmp_lhs_shape) > len(tmp_rhs_shape):
            tmp_rhs_shape = [1] * (
                len(tmp_lhs_shape) - len(tmp_rhs_shape)
            ) + tmp_rhs_shape
        # match shape
        lhs_dims, rhs_dims = [], []
        # pylint: disable=consider-using-enumerate
        for i in range(len(tmp_lhs_shape)):
            if tmp_lhs_shape[i] == 1:
                tmp_lhs_shape[i] = tmp_rhs_shape[i]
                if tmp_rhs_shape[i] != 1:
                    lhs_dims.append(i)
            elif tmp_rhs_shape[i] == 1:
                tmp_rhs_shape[i] = tmp_lhs_shape[i]
                if tmp_lhs_shape[i] != 1:
                    rhs_dims.append(i)
            else:
                assert (
                    tmp_lhs_shape[i] == tmp_rhs_shape[i]
                ), f"Shape mismatch, got {lhs.shape} and {rhs.shape}, and cannot be broadcasted"
        assert tmp_lhs_shape == tmp_rhs_shape
        node.shape = tuple(tmp_lhs_shape)
        node.dims = (lhs_dims, rhs_dims)
        if ctx.verbose:
            print(
                f"Broadcasted shape {lhs.shape} x {rhs.shape} -> {node.shape} for dims: {lhs_dims} & {rhs_dims}"
            )
        return node

    @staticmethod
    def visit_UnaryOp(ctx, node):
        operand = visit_stmt(ctx, node.operand)
        node.shape = operand.shape
        # A bit tricky here, since MLIR only has arith.negf op but not arith.negi
        # https://mlir.llvm.org/docs/Dialects/ArithOps/#arithnegf-arithnegfop
        node.dtype = float32
        return node

    @staticmethod
    def visit_BinOp(ctx, node):
        lhs = visit_stmt(ctx, node.left)
        rhs = visit_stmt(ctx, node.right)
        return TypeInferer.visit_general_binop(ctx, node, lhs, rhs)

    @staticmethod
    def visit_Assign(ctx, node):
        # Compute RHS
        rhs = visit_stmt(ctx, node.value)
        if len(node.targets) > 1:
            raise RuntimeError("Cannot assign to multiple targets")
        if (isinstance(rhs, ast.Call) or len(rhs.shape) > 0) and isinstance(
            node.targets[0], ast.Name
        ):
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
    def visit_constant_tensor(ctx, node, np_values):
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
        value = visit_stmt(ctx, node.value)
        if len(value.shape) > 0:
            node.shape = tuple()
            node.dtype = ctx.buffers[node.value.id].dtype
            visit_stmt(ctx, node.slice)
        elif len(value.shape) == 0 and isinstance(
            value.dtype, (Int, UInt)
        ):  # bit operation
            if isinstance(node.slice, ast.Index):
                visit_stmt(ctx, node.slice)
                node.shape = tuple()
                node.dtype = uint1
            elif isinstance(node.slice, ast.Slice):
                assert isinstance(
                    node.slice.lower, ast.Constant
                ), "lower bound of bit slicing must be constant"
                assert isinstance(
                    node.slice.upper, ast.Constant
                ), "upper bound of bit slicing must be constant"
                lower = visit_stmt(ctx, node.slice.lower)
                upper = visit_stmt(ctx, node.slice.upper)
                assert (
                    upper.value > lower.value
                ), "upper bound must be greater than lower bound"
                node.shape = tuple()
                node.dtype = UInt(upper.value - lower.value)
            else:
                raise RuntimeError("Unsupported bit operation")
        else:
            raise RuntimeError("Can only access bit (slice) for integers")
        return node

    @staticmethod
    def visit_ExtSlice(ctx, node):
        stmts = visit_stmts(ctx, node.dims)
        node.shape = tuple()
        node.dtype = [stmt.dtype for stmt in stmts]
        return node

    @staticmethod
    def visit_Slice(ctx, node):
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
    def visit_AnnAssign(ctx, node):
        target_dtype, target_shape = TypeInferer.visit_type_hint(ctx, node.annotation)
        if isinstance(node.value, ast.List):
            values = compile(ast.Expression(node.value), "", "eval")
            # pylint: disable=eval-used
            values = eval(values)
            TypeInferer.visit_constant_tensor(ctx, node, np.array(values))
        elif (
            isinstance(node.value, ast.Name)
            and node.value.id in ctx.global_vars
            and isinstance(ctx.global_vars[node.value.id], np.ndarray)
        ):
            TypeInferer.visit_constant_tensor(ctx, node, ctx.global_vars[node.value.id])
        else:
            visit_stmt(ctx, node.value)
        ctx.buffers[node.target.id] = node
        node.dtype = target_dtype
        node.shape = target_shape
        visit_stmt(ctx, node.target)
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
        lhs = visit_stmt(ctx, node.left)
        assert len(node.comparators) == 1, "Only support one comparator for now"
        rhs = visit_stmt(ctx, node.comparators[0])
        typing_rule = get_typing_rule(type(node.ops[0]))
        res_type = typing_rule(lhs.dtype, rhs.dtype)[0]
        node.dtype = res_type
        node.shape = tuple()
        return node

    @staticmethod
    def visit_BoolOp(ctx, node):
        visit_stmts(ctx, node.values)
        node.dtype = uint1
        node.shape = tuple()
        return node

    @staticmethod
    def visit_If(ctx, node):
        visit_stmt(ctx, node.test)
        visit_stmts(ctx, node.body)
        if len(node.orelse) > 0:
            visit_stmts(ctx, node.orelse)
        node.dtype = None
        node.shape = None
        return node

    @staticmethod
    def visit_While(ctx, node):
        visit_stmt(ctx, node.test)
        visit_stmts(ctx, node.body)
        if len(node.orelse) > 0:
            raise RuntimeError(
                "'else' clause for 'while' not supported in Allo kernels"
            )
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
            if isinstance(node.func, ast.Attribute):
                # x.T or x.reverse
                assert (
                    len(node.args) == 0
                ), "Only support zero argument for attribute methods"
                attr = visit_stmt(ctx, node.func)
                node.shape = attr.shape
                node.dtype = attr.dtype
            elif node.func.id in {"float", "int"}:
                # Python-Builtin functions
                assert (
                    len(node.args) == 1
                ), "Only support one argument for `float` and `int`"
                new_args = visit_stmts(ctx, node.args)
                node.shape = tuple()
                node.dtype = float32 if node.func.id == "float" else int32
            else:
                raise RuntimeError(f"Unsupported function call {node.func.id}")
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
            return TypeInferer.visit_library_op(
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
        visit_stmts(ctx, node.args)
        if not isinstance(stmts[-1], ast.FunctionDef):
            node.dtype = None
            node.shape = None
        else:
            node.dtype = stmts[-1].dtype
            node.shape = stmts[-1].shape
        return node

    @staticmethod
    def visit_library_op(ctx, node, op_name, new_args):
        if op_name in {
            "exp",
            "softmax",
            "abs",
            "log",
            "add",
            "sub",
            "div",
            "relu",
            "copy",
        }:
            # Element-wise operation
            if op_name in {"add", "sub", "div"}:
                assert (
                    new_args[0].shape == new_args[1].shape
                ), f"Only support element-wise {op_name} of two inputs with the same shape, got {new_args[0].shape} and {new_args[1].shape}"
            node.shape = new_args[0].shape
            node.dtype = new_args[0].dtype
            return node
        if op_name in {"matmul", "bmm", "linear"}:
            argAshape = new_args[0].shape
            argBshape = new_args[1].shape
            node.dtype = new_args[0].dtype
            if op_name == "matmul":
                assert (
                    len(argAshape) == 2 and len(argBshape) == 2
                ), f"Only support matrix multiplication of two 2D inputs, got {len(argAshape)} and {len(argBshape)}"
                assert (
                    argAshape[1] == argBshape[0]
                ), f"The second dimension of the first input and the first dimension of the second input must be the same, got {argAshape[1]} and {argBshape[0]}"
                node.shape = (argAshape[0], argBshape[1])
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
            if op_name == "linear":
                assert new_args[0].shape[1] == new_args[1].shape[1]
                assert new_args[1].shape[0] == new_args[2].shape[0]
                node.shape = (new_args[0].shape[0], new_args[1].shape[0])
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
                new_shape = []
                for new_dim in axes:
                    new_shape.append(shape[new_dim])
                node.shape = tuple(new_shape)
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
