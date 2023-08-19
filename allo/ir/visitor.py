# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, too-many-instance-attributes

from hcl_mlir import InsertionPoint
from hcl_mlir.dialects import hcl as hcl_d


class LoopScopeGuard:
    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx.loop_band_count += 1


class ASTContext:
    def __init__(self, global_vars, mlir_ctx, enable_tensor=False, verbose=False):
        self.ip_stack = []
        self.buffers = {}
        self.top_func = None
        self.top_func_tree = None
        self.global_vars = global_vars
        self.mlir_ctx = mlir_ctx
        hcl_d.register_dialect(mlir_ctx)
        # map from function name to function arguments
        self.func_args = {}
        # used to avoid loop band naming conflict
        self.loop_band_count = 0
        # used for AffineExpr dim counting
        self.dim_count = 0
        self.unnamed_linalg_op_count = 0
        self.affine_vars = []
        self.enable_tensor = enable_tensor
        self.verbose = verbose

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


class ASTVisitor:
    def __call__(self, ctx, node):
        method = getattr(self, "visit_" + node.__class__.__name__, None)
        if method is None:
            error_msg = f'Unsupported node "{node.__class__.__name__}"'
            raise RuntimeError(error_msg)
        res = method(ctx, node)
        if ctx.verbose and hasattr(self, "print_verbose"):
            self.print_verbose(ctx, node)
        return res
