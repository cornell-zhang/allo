# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, too-many-instance-attributes

from hcl_mlir import InsertionPoint
from hcl_mlir.dialects import hcl as hcl_d


class LoopScopeGuard:
    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        self.ctx.nested_loops += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx.nested_loops -= 1
        self.ctx.loop_band_count += 1


class AffineScopeGuard:
    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        self.ctx.dim_count = 0
        self.ctx.affine_vars = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx.dim_count = 0
        self.ctx.affine_vars = []


class ASTContext:
    def __init__(
        self, global_vars, mlir_ctx, func_args=None, enable_tensor=False, verbose=False
    ):
        self.ip_stack = []
        self.buffers = {}
        self.top_func = None
        self.top_func_tree = None
        self.global_vars = global_vars
        self.mlir_ctx = mlir_ctx
        hcl_d.register_dialect(mlir_ctx)
        # map from function name to function arguments
        self.func_args = {} if func_args is None else func_args
        self.func_id = None
        # instantiation of a template function
        self.inst = None
        self.func_name2id = {}
        # used for subfunction call
        self.call_args = []
        # used to count nested loops in a band
        self.nested_loops = 0
        # used to avoid loop band naming conflict
        self.loop_band_count = 0
        # used for AffineExpr dim counting
        self.dim_count = 0
        self.unnamed_linalg_op_count = 0
        self.affine_vars = []
        self.enable_tensor = enable_tensor
        self.verbose = verbose
        # libraries for external IPs
        self.ext_libs = []

    def copy(self):
        ctx = ASTContext(
            self.global_vars.copy(),
            self.mlir_ctx,
            self.func_args,
            self.enable_tensor,
            self.verbose,
        )
        ctx.func_id = self.func_id
        ctx.inst = self.inst
        ctx.func_name2id = self.func_name2id
        ctx.enable_tensor = self.enable_tensor
        ctx.verbose = self.verbose
        ctx.ext_libs = self.ext_libs
        return ctx

    def set_ip(self, ip):
        if not isinstance(ip, InsertionPoint):
            ip = InsertionPoint(ip)
        self.ip_stack.append(ip)

    def get_ip(self):
        return self.ip_stack[-1]

    def pop_ip(self):
        return self.ip_stack.pop()

    def loop_scope_guard(self):
        return LoopScopeGuard(self)

    def affine_scope_guard(self):
        return AffineScopeGuard(self)


class ASTVisitor:
    def __call__(self, ctx, node):
        if node is None:
            return None
        method = getattr(type(self), "visit_" + node.__class__.__name__, None)
        if method is None:
            error_msg = f'Unsupported node "{node.__class__.__name__}"'
            raise RuntimeError(error_msg)
        res = method(ctx, node)
        if ctx.verbose and hasattr(self, "print_verbose"):
            self.print_verbose(ctx, node)
        return res

    @staticmethod
    def visit_Name(ctx, node):
        pass

    @staticmethod
    def visit_Constant(ctx, node):
        pass

    @staticmethod
    def visit_Tuple(ctx, node):
        visit_stmts(ctx, node.elts)

    @staticmethod
    def visit_Index(ctx, node):
        visit_stmt(ctx, node.value)

    @staticmethod
    def visit_Attribute(ctx, node):
        visit_stmt(ctx, node.value)

    @staticmethod
    def visit_For(ctx, node):
        visit_stmt(ctx, node.target)
        visit_stmt(ctx, node.iter)
        visit_stmts(ctx, node.body)

    @staticmethod
    def visit_UnaryOp(ctx, node):
        visit_stmt(ctx, node.operand)

    @staticmethod
    def visit_BinOp(ctx, node):
        visit_stmt(ctx, node.left)
        visit_stmt(ctx, node.right)

    @staticmethod
    def visit_Assign(ctx, node):
        # Compute RHS
        visit_stmt(ctx, node.value)
        if len(node.targets) > 1:
            raise RuntimeError("Cannot assign to multiple targets")
        # Compute LHS
        visit_stmt(ctx, node.targets[0])

    @staticmethod
    def visit_AugAssign(ctx, node):
        visit_stmt(ctx, node.value)
        visit_stmt(ctx, node.target)

    @staticmethod
    def visit_Subscript(ctx, node):
        visit_stmt(ctx, node.value)
        visit_stmt(ctx, node.slice)

    @staticmethod
    def visit_ExtSlice(ctx, node):
        visit_stmts(ctx, node.dims)

    @staticmethod
    def visit_Slice(ctx, node):
        if node.lower is not None:
            visit_stmt(ctx, node.lower)
        if node.upper is not None:
            visit_stmt(ctx, node.upper)
        if node.step is not None:
            visit_stmt(ctx, node.step)

    @staticmethod
    def visit_AnnAssign(ctx, node):
        visit_stmt(ctx, node.value)
        visit_stmt(ctx, node.target)

    @staticmethod
    def visit_FunctionDef(ctx, node):
        print("inside fundef")
        visit_stmts(ctx, node.body)
        print("end inside fundef")

    @staticmethod
    def visit_Compare(ctx, node):
        visit_stmt(ctx, node.left)
        visit_stmt(ctx, node.comparators[0])

    @staticmethod
    def visit_BoolOp(ctx, node):
        visit_stmts(ctx, node.values)

    @staticmethod
    def visit_If(ctx, node):
        visit_stmt(ctx, node.test)
        visit_stmts(ctx, node.body)
        if len(node.orelse) > 0:
            visit_stmts(ctx, node.orelse)

    @staticmethod
    def visit_While(ctx, node):
        visit_stmt(ctx, node.test)
        visit_stmts(ctx, node.body)
        if len(node.orelse) > 0:
            raise RuntimeError(
                "'else' clause for 'while' not supported in Allo kernels"
            )

    @staticmethod
    def visit_Module(ctx, node):
        for stmt in node.body:
            print("stmt", stmt)
            visit_stmt(ctx, stmt)
            print("end", stmt)

    @staticmethod
    def visit_Call(ctx, node):
        visit_stmt(ctx, node.func)
        visit_stmts(ctx, node.args)

    @staticmethod
    def visit_Return(ctx, node):
        visit_stmt(ctx, node.value)

    @staticmethod
    def visit_Expr(ctx, node):
        visit_stmt(ctx, node.value)

    @staticmethod
    def visit_Pass(ctx, node):
        pass


visit_stmt = ASTVisitor()


def visit_stmts(ctx, stmts):
    results = []
    for stmt in stmts:
        results.append(visit_stmt(ctx, stmt))
    return results
