# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

import ast

from .visitor import ASTVisitor, ASTContext

# b = a + 1
# c = func(b)


class VarNode:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.users = []

    def add_user(self, node):
        self.users.append(node)

    def __repr__(self):
        return f"VarNode({self.path}:{self.name})"


class UseDefChain(ASTVisitor):
    def print_verbose(self, ctx, node):
        print("get here", node)

    @staticmethod
    def visit_AnnAssign(ctx, node):
        ctx.buffers[node.target.id] = VarNode(ctx.path, node.target.id)

    @staticmethod
    def visit_FunctionDef(ctx, node):
        print("my inside fundef")
        for arg in node.args:
            ctx.buffers[node.name] = VarNode(node.name, arg.arg)
        visit_stmts(ctx, node.body)
        print("my fasinside fundef")


visit_stmt = UseDefChain()


def visit_stmts(ctx, stmts):
    results = []
    for stmt in stmts:
        results.append(visit_stmt(ctx, stmt))
    return results
