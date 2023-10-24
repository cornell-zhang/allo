# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

import ast
import inspect
import textwrap

from .symbol_resolver import ASTResolver


class VarNode:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.users = set()

    def add_user(self, node):
        # node is a VarNode
        self.users.add(node)

    def add_users(self, nodes):
        for node in nodes:
            self.users.add(node)

    def __repr__(self):
        return f"VarNode({self.path}:{self.name})"


class UseDefChain(ast.NodeVisitor):
    def __init__(self, global_vars):
        self.buffers = {}
        self.path = ""
        self.global_vars = global_vars
        # Used for nested functions
        self.arg_nodes = []

    def get_name(self, name):
        if self.path == "":
            return name
        else:
            return self.path + "." + name

    def dump_graph(self, top_func_name):
        for var in self.buffers.values():
            if var.path == top_func_name:
                print(f"{var.path}_{var.name} [style=filled, color=gray];")
            users = ", ".join([f"{user.path}_{user.name}" for user in var.users])
            print(f"{var.path}_{var.name} -> {{{users}}}")

    def visit_Constant(self, node):
        return []

    def visit_Name(self, node):
        if self.get_name(node.id) in self.buffers:
            return set([self.buffers[self.get_name(node.id)]])
        else:
            return set()

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return set(left).union(set(right))

    def visit_For(self, node):
        if node.orelse:
            raise RuntimeError("'else' clause for 'for' not supported in Allo kernels")
        if isinstance(node.iter, ast.Call):
            obj = ASTResolver.resolve(node.iter.func, self.global_vars)
            if (
                obj is None
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ) or (obj is not None and obj.__name__ in {"grid", "reduction"}):
                res = []
                for stmt in node.body:
                    res.append(self.visit(stmt))
                return res
        raise RuntimeError("Unsupported for loop")

    def visit_Call(self, node):
        obj = ASTResolver.resolve(node.func, self.global_vars)
        if obj is None:
            raise NotImplementedError(f"Function {node.func.id} not found")
        if obj.__module__.startswith("allo"):
            raise NotImplementedError("allo functions not supported")
        # User-defined subfunction
        func = self.global_vars[node.func.id]
        if isinstance(func, ast.FunctionDef):
            # Has already been defined in the top-level scope
            raise NotImplementedError("Nested functions not supported")
        else:
            # Visit arguments in the top-level
            arg_nodes = []
            # The arguments have order
            for arg in node.args:
                arg_nodes += list(self.visit(arg))
            func = self.global_vars[node.func.id]
            src, _ = inspect.getsourcelines(func)
            src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
            src = textwrap.dedent("\n".join(src))
            tree = ast.parse(src)
            original_arg_nodes = self.arg_nodes
            self.arg_nodes = arg_nodes
            ret = self.visit(tree)
            arg_nodes += list(ret)
            self.arg_nodes = original_arg_nodes
            return arg_nodes

    def visit_Assign(self, node):
        # Compute RHS
        if len(node.targets) > 1:
            raise NotImplementedError(
                "Multiple assignment in one statement not supported"
            )
        parents = self.visit(node.value)
        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node.targets[0], ast.Subscript):
            name = node.targets[0].value.id
        else:
            raise RuntimeError("Unsupported assignment")
        var = VarNode(self.path, name)
        for parent in parents:
            parent.add_user(var)
        self.buffers[self.get_name(name)] = var

    def visit_AnnAssign(self, node):
        var = VarNode(self.path, node.target.id)
        parents = self.visit(node.value)
        for parent in parents:
            parent.add_user(var)
        self.buffers[self.get_name(node.target.id)] = var

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Subscript):
            name = node.target.value.id
        elif isinstance(node.target, ast.Name):  # scalar
            name = node.target.id
        else:
            raise NotImplementedError("Unsupported AugAssign")
        var = VarNode(self.path, name)
        parents = self.visit(node.value)
        for parent in parents:
            parent.add_user(var)
        self.buffers[self.get_name(name)] = var

    def visit_Subscript(self, node):
        res = self.visit(node.value)
        return res

    def visit_FunctionDef(self, node):
        original_path = self.path
        if self.path == "":
            self.path = node.name
            # create initial variables
            for arg in node.args.args:
                self.buffers[self.get_name(arg.arg)] = VarNode(node.name, arg.arg)
        else:
            self.path = node.name
            for inner_arg, outer_arg in zip(node.args.args, self.arg_nodes):
                self.buffers[self.get_name(inner_arg.arg)] = VarNode(
                    self.path, inner_arg.arg
                )
                outer_arg.add_user(self.buffers[self.get_name(inner_arg.arg)])
        res = []
        for stmt in node.body:
            res.append(self.visit(stmt))
        self.path = original_path
        return res[-1]

    def visit_Module(self, node):
        res = []
        assert (
            len(node.body) == 1
        ), "Only one function definition in a module is allowed"
        for stmt in node.body:
            res.append(self.visit(stmt))
        return res[0]

    def visit_Return(self, node):
        return self.visit(node.value)
