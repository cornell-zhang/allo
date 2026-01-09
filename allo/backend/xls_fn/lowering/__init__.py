"""MLIR lowering passes for XLS backend."""

from .mlir_to_dslx_fn import MlirToDslxLowerer as MlirToDslxFnLowerer
from .mlir_to_ir_fn import MlirToXlsIRLowerer

__all__ = [
    'MlirToDslxFnLowerer',      # MLIR → DSLX functions
    'MlirToXlsIRLowerer',        # MLIR → XLS IR functions
]
