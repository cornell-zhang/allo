# XLS Backend Scripts

This directory contains utility scripts for working with the Allo-XLS backend.

## mlir_to_verilog.py

A standalone script that converts MLIR code to Verilog for systolic arrays.

### Pipeline

The script implements the complete compilation pipeline:

```
MLIR → DSLX → XLS IR → Optimized IR → Verilog
```

1. **MLIR → DSLX**: Uses the `MlirToDslxProcLowererAST` to convert MLIR to XLS DSLX code
2. **DSLX → XLS IR**: Uses `ir_converter_main` from XLS to convert DSLX to intermediate representation
3. **Optimize IR**: Uses `opt_main` from XLS to optimize the intermediate representation
4. **Generate Verilog**: Uses `codegen_main` from XLS to generate synthesizable Verilog

### Prerequisites

- Allo-XLS backend installed and in Python path
- XLS tools available at `/scratch/users/zrs29/xls/xls/`
  - `ir_converter_main`
  - `opt_main`
  - `codegen_main`

### Usage

#### Basic usage with input file:

```bash
./mlir_to_verilog.py input.mlir -o output_directory
```

This will:
- Read MLIR from `input.mlir`
- Generate Verilog in `output_directory/systolic.v`
- Clean up intermediate files

#### Read from stdin:

```bash
cat input.mlir | ./mlir_to_verilog.py --stdin -o output_directory
```

#### Keep intermediate files:

```bash
./mlir_to_verilog.py input.mlir -o output_directory --keep-intermediates
```

This preserves:
- `systolic.mlir` - Input MLIR
- `systolic.x` - Generated DSLX code
- `systolic.ir` - XLS intermediate representation
- `systolic_opt.ir` - Optimized IR
- `systolic.v` - Final Verilog output

#### Custom output name:

```bash
./mlir_to_verilog.py input.mlir -o output/ --name my_design
```

Generates `my_design.v` instead of `systolic.v`

#### Adjust pipeline stages:

```bash
./mlir_to_verilog.py input.mlir -o output/ --pipeline-stages 3
```

Controls the number of pipeline stages in the generated Verilog (default: 5)

### Options

```
positional arguments:
  input                 Input MLIR file (omit if using --stdin)

options:
  -h, --help            Show help message
  --stdin               Read MLIR from stdin instead of file
  -o, --output-dir DIR  Output directory for generated files (required)
  --keep-intermediates  Keep intermediate files (DSLX, IR, etc.)
  --pipeline-stages N   Number of pipeline stages (default: 5)
  --name NAME           Base name for output files (default: systolic)
```

### Example

Given a simple 2x2 systolic array MLIR file:

```bash
./mlir_to_verilog.py systolic_2x2.mlir -o ./build --keep-intermediates
```

Output:
```
======================================================================
MLIR → VERILOG PIPELINE
======================================================================

[1/5] Parsing MLIR...
      ✓ Saved to ./build/systolic.mlir

[2/5] MLIR → DSLX...
      ✓ 142 lines → ./build/systolic.x

[3/5] DSLX → XLS IR...
      ✓ 256 lines → ./build/systolic.ir

[4/5] Optimize XLS IR...
      Top proc: __systolic_2x2_k1_uint32__
      ✓ 189 lines → ./build/systolic_opt.ir

[5/5] Generate Verilog...
      ✓ 423 lines → ./build/systolic.v

======================================================================
✅ SUCCESS: Verilog generation completed!
======================================================================

Output files:
  Verilog: ./build/systolic.v (423 lines)
  MLIR:    ./build/systolic.mlir
  DSLX:    ./build/systolic.x
  IR:      ./build/systolic.ir
  Opt IR:  ./build/systolic_opt.ir
```

### Troubleshooting

**Import errors**: Make sure the allo-xls-backend is installed and accessible:
```bash
export PYTHONPATH=/path/to/allo-xls-backend:$PYTHONPATH
```

**XLS tools not found**: Update the `XLS_DIR` variable in the script to point to your XLS installation.

**MLIR parsing errors**: Ensure your MLIR is valid and contains the expected systolic array structure with grid and PE functions.

**Verilog generation fails**: Try adjusting `--pipeline-stages` to a lower value (e.g., 1 or 2).

## Other Scripts

### aie-setup.sh

Setup script for AMD AI Engine development environment.

### lint/

Directory containing linting configuration and scripts.
