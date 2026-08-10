"""MLIR lowering passes for XLS backend."""

from .mlir_to_dslx_fn import MlirToDslxLowerer as MlirToDslxFnLowerer
from .mlir_to_ir_fn import MlirToXlsIRLowerer
from .mlir_to_dslx_proc import (
    MlirToDslxProcLowerer,
    DslxCombProcLowerer,
    DslxStatefulProcLowerer,
    lower_mlir_to_proc,
)
from .instruction_emitter import InstructionEmitter
from .memory import (
    MemoryBinding,
    MemoryEmitter,
    RAM_TEMPLATE,
    discover_memory_bindings,
    build_memory_channels,
)

__all__ = [
    # MLIR → DSLX functions
    'MlirToDslxFnLowerer',
    # MLIR → XLS IR functions
    'MlirToXlsIRLowerer',
    # MLIR → DSLX procs
    'MlirToDslxProcLowerer',
    'DslxCombProcLowerer',
    'DslxStatefulProcLowerer',
    'lower_mlir_to_proc',
    # Instruction emitter
    'InstructionEmitter',
    # Memory
    'MemoryBinding',
    'MemoryEmitter',
    'RAM_TEMPLATE',
    'discover_memory_bindings',
    'build_memory_channels',
]
