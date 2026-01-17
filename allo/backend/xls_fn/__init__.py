"""XLS backend for DSLX code generation.

This module provides MLIR to DSLX and XLS IR lowering with AST-based code generation.
Supports both DSLX functions (combinational) and procs (stateful with channels).
"""

# AST infrastructure
from .dslx_ast import DslxProcSerializer
from .dslx_ast.dslx_ast import *

# Lowering - Functions
from .lowering import (
    MlirToDslxFnLowerer,
    MlirToXlsIRLowerer,
)

# Lowering - Procs
from .lowering import (
    MlirToDslxProcLowerer,
    DslxCombProcLowerer,
    DslxStatefulProcLowerer,
    lower_mlir_to_proc,
    InstructionEmitter,
    MemoryBinding,
    MemoryEmitter,
    RAM_TEMPLATE,
    discover_memory_bindings,
    build_memory_channels,
)

# Type utilities
from .utils import (
    allo_dtype_to_dslx_type,
    is_float_type,
    float_to_dslx_literal,
    emit_float_defs,
    get_zero_literal,
)

# Module interfaces for full toolchain
from .xls_fn import (
    DslxFunctionModule,
    XlsIRModule,
    DslxProcModule,
)

__all__ = [
    # AST
    'DslxProcSerializer',

    # Lowering - Functions
    'MlirToDslxFnLowerer',
    'MlirToXlsIRLowerer',

    # Lowering - Procs
    'MlirToDslxProcLowerer',
    'DslxCombProcLowerer',
    'DslxStatefulProcLowerer',
    'lower_mlir_to_proc',
    'InstructionEmitter',
    'MemoryBinding',
    'MemoryEmitter',
    'RAM_TEMPLATE',
    'discover_memory_bindings',
    'build_memory_channels',

    # Type utilities
    'allo_dtype_to_dslx_type',
    'is_float_type',
    'float_to_dslx_literal',
    'emit_float_defs',
    'get_zero_literal',

    # Module interfaces
    'DslxFunctionModule',
    'XlsIRModule',
    'DslxProcModule',
]
