#!/usr/bin/env python3
"""
MLIR to DSLX Conversion Script

This script takes raw MLIR text as input and produces DSLX output.

Usage:
  # From a file:
  ./mlir_to_dslx_fn.py input.mlir -o output.x

  # From stdin:
  cat input.mlir | ./mlir_to_dslx_fn.py --stdin -o output.x

  # Print to stdout:
  ./mlir_to_dslx_fn.py input.mlir
"""

import sys
import os
import argparse
from pathlib import Path

# Add allo-xls-backend root to Python path
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import MLIR parsing and lowering
try:
    from allo._mlir.ir import Module as MlirModule
    from allo._mlir import ir as mlir_ir
    from allo._mlir.dialects import func as func_d
    from allo.backend.xls_fn.lowering import MlirToDslxFnLowerer
except ImportError as e:
    print(f"Error: Failed to import allo modules: {e}", file=sys.stderr)
    print("Please ensure allo is properly installed.", file=sys.stderr)
    sys.exit(1)


def parse_mlir(mlir_text):
    """Parse MLIR text into a module."""
    try:
        # Filter out allo dialect operations that are just scheduling metadata.
        # Operations like `%0 = allo.create_op_handle "name"` create handles for
        # Allo's scheduling system but don't affect the computation - the result
        # is never used and doesn't appear in the function's return statement.
        lines = mlir_text.splitlines()
        filtered_lines = []
        for line in lines:
            # Skip allo dialect metadata operations
            if 'allo.create_op_handle' in line or 'allo.yield' in line:
                continue
            filtered_lines.append(line)

        cleaned_mlir = '\n'.join(filtered_lines)

        with mlir_ir.Context() as ctx:
            module = MlirModule.parse(cleaned_mlir, ctx)
            return module
    except Exception as e:
        raise RuntimeError(f"Failed to parse MLIR: {e}")


def mlir_to_dslx(mlir_module):
    """Convert MLIR module to DSLX code."""
    try:
        # Find the first function in the module
        func_op = None
        for op in mlir_module.body.operations:
            if isinstance(op, func_d.FuncOp):
                func_op = op
                break

        if func_op is None:
            raise RuntimeError("No function found in MLIR module")

        # Use function-based lowerer
        lowerer = MlirToDslxFnLowerer(func_op)
        dslx_code = lowerer.lower()

        # Check for error messages in generated code
        if "ERROR" in dslx_code and dslx_code.startswith("//"):
            raise RuntimeError(f"MLIR to DSLX lowering failed:\n{dslx_code}")

        return dslx_code
    except Exception as e:
        raise RuntimeError(f"Failed to convert MLIR to DSLX: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MLIR to DSLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mlir -o output.x
  %(prog)s input.mlir -o output_dir/
  cat input.mlir | %(prog)s --stdin -o output.x
  %(prog)s input.mlir  # Print to stdout
        """
    )

    parser.add_argument(
        'input',
        nargs='?',
        help='Input MLIR file (omit if using --stdin)'
    )
    parser.add_argument(
        '--stdin',
        action='store_true',
        help='Read MLIR from stdin instead of file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file or directory (if omitted, prints to stdout)'
    )
    parser.add_argument(
        '--name',
        default='output',
        help='Base name for output file when output is a directory (default: output)'
    )

    args = parser.parse_args()

    # Validate input
    if args.stdin and args.input:
        parser.error("Cannot specify both input file and --stdin")
    if not args.stdin and not args.input:
        parser.error("Must specify either input file or --stdin")

    # Read MLIR input
    try:
        if args.stdin:
            mlir_text = sys.stdin.read()
            input_name = args.name
        else:
            with open(args.input, 'r') as f:
                mlir_text = f.read()
            input_name = Path(args.input).stem
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        return 1

    try:
        # Parse MLIR
        mlir_module = parse_mlir(mlir_text)

        # Convert to DSLX
        dslx_code = mlir_to_dslx(mlir_module)

        # Determine output
        if args.output:
            output_path = Path(args.output)

            # If output is a directory, create file inside it
            if output_path.is_dir() or (not output_path.exists() and not output_path.suffix):
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"{input_name}.x"
            else:
                # Create parent directory if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_file = output_path

            with open(output_file, 'w') as f:
                f.write(dslx_code)

            print(f"✓ Generated DSLX: {output_file}")
        else:
            # Print to stdout
            print(dslx_code)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
