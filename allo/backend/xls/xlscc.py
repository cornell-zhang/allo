"""MLIR to XLSCC lowering implementation."""

from allo._mlir.ir import Operation, MemRefType
from allo._mlir.ir import IntegerAttr, FloatAttr, DenseElementsAttr
from allo._mlir.dialects import affine as affine_d
from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import arith as arith_d
from allo._mlir.dialects import linalg as linalg_d
from allo._mlir.dialects import memref as memref_d

from .codegen_context import CodegenContext
from .xlscc_nodes import (
    DslxVar, DslxConst, DslxBinOp, DslxLoad, DslxStore,
    DslxFor, DslxLet, DslxArrayInit
)

class XLSCCModule:
    def __init__(self, mod, top_func_name, project=None, configs=None):
        self.top_func_name = top_func_name
        self.project = project
        self.configs = configs or {}

        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.func = find_func_in_module(self.module, top_func_name)

            # 1) Run an XLS-friendly lowering pipeline
            self._run_lowering()

            # 2) Emit XLS[cc] DSL C++
            buf = io.StringIO()
            allo_d.emit_xlscc(self.module, buf)  # <- you'll have to implement this
            buf.seek(0)
            self.xls_code = buf.read()

        if project is not None:
            self._materialize_project()

    def _run_lowering(self):
        #TODO
    
    def _materialize_project(self):
        os.makedirs(self.project, exist_ok=True)
        with open(f"{self.project}/kernel.cc", "w") as f:
            f.write(self.xls_code)
        # Optionally write BUILD / bazel config, test harness, etc.

    def __repr__(self):
        return self.xls_code

    def __call__(self, *args):
        #TODO
