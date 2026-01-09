"""XLS backend for DSLX code generation.

This module provides MLIR to DSLX and XLS IR lowering with AST-based code generation.
"""

# AST infrastructure
from .dslx_ast import DslxProcSerializer
from .dslx_ast.proc_ast import *
from .dslx_ast.function_ast import *

# Lowering
from .lowering import (
    MlirToDslxFnLowerer,
    MlirToXlsIRLowerer,
)

# Module interfaces for full toolchain
from .xls_fn import (
    DslxFunctionModule,
    XlsIRModule,
)

__all__ = [
    # AST
    'DslxProcSerializer',

    # Lowering
    'MlirToDslxFnLowerer',
    'MlirToXlsIRLowerer',

    # Module interfaces
    'DslxFunctionModule',
    'XlsIRModule',
]
