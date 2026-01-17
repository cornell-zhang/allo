# XLS Backend Lowering

This directory contains MLIR lowering passes for the XLS backend.

## Structure

```
lowering/
├── mlir_to_dslx_fn.py      # MLIR → DSLX functions
├── mlir_to_dslx_proc.py    # MLIR → DSLX procs (systolic arrays)
├── mlir_to_ir_fn.py        # MLIR → XLS IR functions
├── shared/                  # Shared utilities
│   ├── dslx_nodes.py       # DSLX AST node definitions
│   └── codegen_context.py  # Codegen context for tracking variables
└── __init__.py
```

## Three Lowering Paths

### 1. MLIR → DSLX Functions (`mlir_to_dslx_fn.py`)
**Purpose:** Lower Allo MLIR (affine loops, linalg) directly to DSLX functions

**Use case:** General computation patterns (gemm, conv2d, etc.)

**Example:**
```python
from allo.backend.xls.lowering import MlirToDslxFnLowerer

lowerer = MlirToDslxFnLowerer(func_op)
dslx_code = lowerer.lower()
```

**Features:**
- Handles affine loops with arbitrary nesting
- Supports affine.apply operations
- Filters out Allo dialect metadata operations
- Generates functional DSLX with for-loop expressions

### 2. MLIR → DSLX Procs (`mlir_to_dslx_proc.py`)
**Purpose:** Lower Allo MLIR (with systolic array patterns) to DSLX procs

**Use case:** Systolic array implementations

**Example:**
```python
from allo.backend.xls.lowering import MlirToDslxProcLowererAST

lowerer = MlirToDslxProcLowererAST(mlir_module)
dslx_code = lowerer.lower()
```

**Features:**
- Extracts grid-based PE structures
- Generates channel-based communication
- Creates proc-based systolic arrays

### 3. MLIR → XLS IR Functions (`mlir_to_ir_fn.py`)
**Purpose:** Lower Allo MLIR directly to XLS IR (bypassing DSLX)

**Use case:** Direct IR generation for optimization

**Example:**
```python
from allo.backend.xls.lowering import MlirToXlsIRLowerer

lowerer = MlirToXlsIRLowerer(func_op)
ir_code = lowerer.lower()
```

## Shared Utilities

### `shared/dslx_nodes.py`
DSLX AST node classes:
- `DslxVar` - Variables
- `DslxConst` - Constants
- `DslxBinOp` - Binary operations
- `DslxLoad` / `DslxStore` - Array access
- `DslxFor` - Loop expressions
- `DslxLet` - Let bindings
- `DslxArrayInit` - Array initialization

### `shared/codegen_context.py`
Codegen context for tracking:
- Variable bindings (MLIR values → DSLX nodes)
- Memory buffer shapes and types
- Loop nesting stack
- Generated statements

## Integration

Scripts in `scripts/` directory use these lowerers:
- `mlir_to_dslx_fn.py` - CLI tool for MLIR → DSLX functions
- `mlir_to_verilog_proc.py` - Full pipeline MLIR → DSLX procs → Verilog
