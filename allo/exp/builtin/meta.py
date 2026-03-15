# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
import allo._mlir.extras.types as mlir_types
from allo._mlir.dialects import func as func_d, memref as memref_d
from allo._mlir.ir import FlatSymbolRefAttr, UnitAttr
import allo._mlir.extras.dialects.func as func


@register_builtin_handler("get_wid")
class WidHandler(BuiltinHandler):

    def build(self, node: ast.Call, *args):
        assert isinstance(args[0], list) and len(args[0]) == 1
        targets = args[0][0].elts
        num = len(targets)
        callee = self.builder.current_func.name.value
        grid_name = self.builder.symbol_table.mangle_grid_name(callee)
        builtin_func = f"{grid_name}.get_wid"
        # insert function declaration in global
        results = [mlir_types.index()] * num
        with self.builder.get_global_ip():
            func.function(builtin_func, [], results, is_private=True)
        # call function in work
        op = func_d.CallOp(
            results, FlatSymbolRefAttr.get(builtin_func), [], ip=self.builder.get_ip()
        )
        for i, target in enumerate(targets):
            assert isinstance(target, ast.Name)
            self.builder.reserved_bindings[target.id] = op.results[i]
