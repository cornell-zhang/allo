import re
import io
from typing import List

from .._mlir.dialects import allo as allo_d
from .._mlir.ir import Context, Location, Module
from .._mlir.passmanager import PassManager
from ..ir.transform import find_func_in_module

from .xls_wrapper import wrap_xlscc, validate_xls_ir  # <-- NEW import

class XLSCCModule:
    def __init__(self,
                 mlir_text_or_module,
                 top_func_name,
                 project=None,
                 use_memory=False):
        self.top_func_name = top_func_name
        self.project = project
        self.use_memory = use_memory

        # Parse MLIR + run minimal lowering
        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)

            # Always parse as string to ensure correct context
            self.module = Module.parse(str(mlir_text_or_module), ctx)

            # run same lowering pipeline (DO NOT lower affine)
            pm = PassManager.parse(
                "builtin.module("
                "empty-tensor-to-alloc-tensor,"
                "func.func(convert-linalg-to-affine-loops)"
                ")"
            )
            pm.run(self.module.operation)
            
            # Re-find the function after running passes (previous reference is invalidated)
            self.func = find_func_in_module(self.module, top_func_name)

        # 1) Take a snapshot of the lowered MLIR as text and validate for XLS.
        self.mlir_text = str(self.module)
        validate_xls_ir(self.mlir_text, project=project)  # ERROR OUT EARLY IF NOT XLS-LEGAL

        # 2) Emit the XLS[cc] HLS Code from MLIR
        buf = io.StringIO()
        allo_d.emit_xhls(self.module, buf, self.use_memory)
        buf.seek(0)
        self.core_code = buf.read()

        # 3) Extract function name and count array arguments from MLIR
        # For XLS, functions with arrays have no parameters, so we detect array args from MLIR
        func_name = top_func_name
        
        # Count array arguments (non-stream shaped types) from the function
        param_names: List[str] = []
        if self.func:
            for arg in self.func.arguments:
                arg_type_str = str(arg.type)
                # Check if it's a memref/tensor (array) that's not a stream
                if ("memref" in arg_type_str or "tensor" in arg_type_str) and "stream" not in arg_type_str.lower():
                    param_names.append(f"arg{len(param_names)}")

        # Wrap core C++ with channels + wrapper class in XLS[cc] style
        self.final_cpp = wrap_xlscc(
            mlir_module_text=self.mlir_text,
            core_code=self.core_code,
            function_names=(self.top_func_name, func_name),
            function_inputs=param_names,  # <-- PASS LIST OF NAMES, NOT RAW STRING
            use_memory=self.use_memory,
        )

        if self.project is not None:
            import os
            os.makedirs(self.project, exist_ok=True)
            with open(f"{self.project}/test_block.cpp", "w", encoding="utf-8") as f:
                f.write(self.final_cpp)

    def __repr__(self):
        return self.final_cpp
