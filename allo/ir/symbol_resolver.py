# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Reference: taichi/python/taichi/lang/ast/symbol_resolver.py

import ast


class ASTResolver:
    """Provides helper methods to resolve AST nodes."""

    @staticmethod
    def resolve(node, scope):
        """resolve a given AST node to a Python object.

        This is only intended to check if a given AST node resolves to a symbol
        under some namespaces, e.g. the ``a.b.c.foo`` pattern, but not meant for
        more complicated expressions like ``(a + b).foo``.

        Args:
            node (Union[ast.Attribute, ast.Name]): an AST node to be resolved.
            scope (Dict[str, Any]): Maps from symbol names to objects, for
                example, globals()

        Returns:
            object: The actual Python object that ``node`` resolves to.
        """
        if isinstance(node, ast.Name):
            return scope.get(node.id)

        if not isinstance(node, ast.Attribute):
            return None

        v = node.value
        chain = [node.attr]
        while isinstance(v, ast.Attribute):
            chain.append(v.attr)
            v = v.value
        if not isinstance(v, ast.Name):
            # Example cases that fall under this branch:
            #
            # x[i].attr: ast.Subscript
            # (a + b).attr: ast.BinOp
            # ...
            return None
        chain.append(v.id)

        for attr in reversed(chain):
            try:
                if isinstance(scope, dict):
                    scope = scope[attr]
                else:
                    scope = getattr(scope, attr)
            except (KeyError, AttributeError):
                return None
        # The name ``scope`` here could be a bit confusing
        return scope

    @staticmethod
    def resolve_constant(node, ctx):
        # pylint: disable=eval-used
        return eval(compile(ast.Expression(node), "", "eval"), ctx.global_vars)
