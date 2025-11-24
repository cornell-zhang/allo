import re

import io
from .._mlir.dialects import allo as allo_d
from .._mlir.ir import Context, Location, Module
from .._mlir.passmanager import PassManager
from ..ir.transform import find_func_in_module

from .xls_wrapper import wrap_xlscc 

class XlsccModule:
    def __init__(self,
                 mlir_text_or_module,
                 top_func_name,
                 project=None):
        self.top_func_name = top_func_name
        self.project = project

        # Parse MLIR + run minimal lowering
        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)

            # mlir may be Module object or string
            if isinstance(mlir_text_or_module, Module):
                self.module = mlir_text_or_module
            else:
                self.module = Module.parse(str(mlir_text_or_module), ctx)

            self.func = find_func_in_module(self.module, top_func_name)

            # run same lowering pipeline (DO NOT lower affine)
            pm = PassManager.parse(
                "builtin.module("
                "empty-tensor-to-alloc-tensor,"
                "func.func(convert-linalg-to-affine-loops)"
                ")"
            )
            pm.run(self.module.operation)

        self.mlir_text = str(self.module)

        # emit the XLS [cc] HLS Code
        buf = io.StringIO()
        allo_d.emit_xhls(self.module, buf)
        buf.seek(0)
        self.core_code = buf.read()

        pattern = r"""
            (?P<rtype>[\w:\<\>\~]+)
            \s+                          
            (?P<name>[A-Za-z_]\w*(?:::\w*)*)
            \s*
            \(
                (?P<params>[^)]*) 
            \)
        """

        match = re.search(pattern, self.core_code, re.VERBOSE)
        func_name = match.group("name")
        func_params = match.group("params")

        # Wrap core C++ with channels + wrapper class
        self.final_cpp = wrap_xlscc(
            mlir_module_text=self.mlir_text,
            core_code=self.core_code,
            function_names=(self.top_func_name, func_name),
            function_inputs=func_params
        )

        if self.project is not None:
            import os
            os.makedirs(self.project, exist_ok=True)
            with open(f"{self.project}/test_block.cpp", "w", encoding="utf-8") as f:
                f.write(self.final_cpp)

    def __repr__(self):
        return self.final_cpp