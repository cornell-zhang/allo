# XLS Backend - DSLX Code Generation

AST-based MLIR to DSLX and XLS IR lowering for hardware synthesis.

## Prerequisites

Before running any code, make sure to activate the Allo conda environment:

```bash
conda activate allo
```

Additionally, ensure that the following environment variables are set:
- `LLVM_BUILD_DIR`: Path to your LLVM build directory
- Run on `zhang-21.ece.cornell.edu`
- XLS tools should be available at `/scratch/users/zrs29/xls/xls/`

If you decide to use `havarti.cs.cornell.edu` as your server of choice:
- For conda, `conda activate axls`
- Just run everything from the `/scratch/cys36` directory

## Directory Structure

```
allo/backend/xls_fn/
├── dslx_ast/          # AST node definitions
│   ├── proc_ast.py        # Proc AST (channels, spawn, config)
│   ├── function_ast.py    # Function AST (expressions, statements)
│   └── serializer.py      # AST → DSLX text conversion
│
├── lowering/          # MLIR → DSLX/IR transformation
│   ├── mlir_to_dslx_fn.py    # MLIR → DSLX function lowering
│   ├── mlir_to_ir_fn.py      # MLIR → XLS IR lowering
│   ├── mlir_to_dslx_cli.py   # CLI utilities
│   └── shared/               # Shared lowering utilities
│
├── scripts/           # Utility scripts
│   ├── mlir_to_dslx_fn.py    # MLIR → DSLX function script
│   └── mlir_to_ir_fn.py      # MLIR → XLS IR script
│
├── utils/             # Utilities
│   ├── codegen_context.py
│   └── debug_utils.py
│
└── xls_fn.py          # Module interfaces for full toolchain
```

## Features

This backend provides:
- **MLIR to DSLX Function Lowering**: Convert MLIR functions to DSLX code
- **MLIR to XLS IR Lowering**: Direct MLIR to XLS IR conversion
- **AST-Based Code Generation**: Clean, maintainable AST representation
- **DSLX Serialization**: Pretty-printing of DSLX code from AST
- **Full Toolchain Integration**: End-to-end pipeline from MLIR to Verilog with module interfaces

## Scripts

The [scripts/](scripts/) directory contains utility scripts for the XLS backend workflow:

### [scripts/mlir_to_dslx_fn.py](scripts/mlir_to_dslx_fn.py)

Converts MLIR functions to DSLX code.

Pipeline: `MLIR → DSLX`

Usage:
```bash
./scripts/mlir_to_dslx_fn.py input.mlir -o output.x
cat input.mlir | ./scripts/mlir_to_dslx_fn.py --stdin -o output.x
./scripts/mlir_to_dslx_fn.py input.mlir  # Print to stdout
```

This script generates DSLX function definitions suitable for testing individual operations or building blocks.

### [scripts/mlir_to_ir_fn.py](scripts/mlir_to_ir_fn.py)

Converts MLIR functions directly to XLS IR, bypassing DSLX generation.

Pipeline: `MLIR → XLS IR`

Usage:
```bash
./scripts/mlir_to_ir_fn.py input.mlir -o output.ir
cat input.mlir | ./scripts/mlir_to_ir_fn.py --stdin -o output.ir
./scripts/mlir_to_ir_fn.py input.mlir  # Print to stdout
```

This script generates XLS IR directly from MLIR without the intermediate DSLX step, useful for IR-level optimization and analysis.

## Module Interfaces

The backend provides two high-level module classes for full toolchain integration:

### DslxFunctionModule

Source-to-source workflow: `MLIR → DSLX → XLS IR → Optimized IR → Verilog`

This module uses function-based DSLX lowering and provides methods for:
- `.codegen()` - Generate DSLX from MLIR
- `.interpret()` - Run DSLX interpreter
- `.to_ir()` - Convert DSLX to XLS IR
- `.opt()` - Optimize XLS IR
- `.to_vlog()` - Generate Verilog from optimized IR
- `.flow()` - Run complete pipeline

### XlsIRModule

Direct IR workflow: `MLIR → XLS IR → Optimized IR → Verilog`

This module bypasses DSLX generation and provides methods for:
- `.to_ir()` - Generate XLS IR directly from MLIR
- `.opt()` - Optimize XLS IR
- `.to_vlog()` - Generate Verilog from optimized IR
- `.flow()` - Run complete pipeline

Both modules automatically locate XLS toolchain binaries and manage intermediate files in an output directory.

## Usage

### Module Interfaces (Recommended)

For full toolchain integration, use `DslxFunctionModule` or `XlsIRModule` which provide end-to-end workflows from MLIR to Verilog.

### Direct Lowering

For custom workflows or integration into larger pipelines, use the lowerer classes directly:
- `MlirToDslxFnLowerer` - MLIR to DSLX function lowering
- `MlirToXlsIRLowerer` - MLIR to XLS IR lowering
