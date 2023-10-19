# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

import ast

class VarNode:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.users = []

    def add_user(self, node):
        # node is a VarNode
        self.users.append(node)

    def add_users(self, nodes):
        for node in nodes:
            self.users.append(node)

    def __repr__(self):
        return f"VarNode({self.path}:{self.name})"


class UseDefChain(ast.NodeVisitor):
    def __init__(self):
        self.buffers = {}
        self.path = ""

    def visit_Constant(self, node):
        return []

    def visit_Name(self, node):
        if node.id in self.buffers:
            return [self.buffers[node.id]]
        else:
            return []

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return left + right

    def visit_AnnAssign(self, node):
        var = VarNode(self.path, node.target.id)
        parents = self.visit(node.value)
        for parent in parents:
            parent.add_user(var)
        self.buffers[node.target.id] = var

    def visit_FunctionDef(self, node):
        original_path = self.path
        if self.path == "":
            self.path = node.name
        else:
            self.path = ".".join(self.path.split(".") + [node.name])
        # create initial variables
        for arg in node.args.args:
            self.buffers[arg.arg] = VarNode(node.name, arg.arg)
        self.generic_visit(node)
        self.path = original_path
