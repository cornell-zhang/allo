# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
from allo.ir.types import bool as allo_bool


def cpp_style_bool_op_rule():
    cpp_style_bool = allo_bool

    def rule(*args):
        # Check if any operand is unsupported
        for t in args:
            print(t)
            if isinstance(t, bool) or t == cpp_style_bool:
                continue
            raise TypeError(f"Type {t} not supported in boolean operation")
        return tuple([cpp_style_bool] * (len(args) + 1))

    return rule


CPP_STYLE_BOOL_OP_RULE = cpp_style_bool_op_rule()

cpp_style_registry = {
    ast.And: CPP_STYLE_BOOL_OP_RULE,
    ast.Or: CPP_STYLE_BOOL_OP_RULE,
}
