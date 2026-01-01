# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ValidationError:
    line_num: int
    line_content: str
    error_type: str
    message: str


# Help messages for each error type
ERROR_TYPE_HELP = {
    'FLOAT_TYPE': [
        "FLOAT_TYPE:",
        "  XLS[cc] does not support floating-point arithmetic.",
        "  Options:",
        "    1. Use integer types with manual scaling (e.g., Q16.16 fixed-point)",
        "    2. Use the Vitis HLS backend instead: s.build(target='vitis_hls')",
    ],
    'FIXED_TYPE': [
        "FIXED_TYPE:",
        "  XLS[cc] does not support Allo's fixed-point types.",
        "  Use integer types with manual fixed-point arithmetic.",
    ],
    'UNSUPPORTED_PRAGMA': [
        "UNSUPPORTED_PRAGMA:",
        "  XLS only supports 'pipeline' and 'unroll' loop directives.",
        "  Remove other scheduling primitives or use Vitis HLS backend.",
    ],
    'DYNAMIC_SHAPE': [
        "DYNAMIC_SHAPE:",
        "  XLS requires all array sizes to be compile-time constants.",
        "  Ensure all dimensions are specified as integer literals.",
    ],
    'DYNAMIC_ALLOC': [
        "DYNAMIC_ALLOC:",
        "  XLS requires all array sizes to be compile-time constants.",
        "  Ensure all dimensions are specified as integer literals.",
    ],
}


def generate_diagnostic_report(
    errors: list[ValidationError],
    mlir_text: str,
    context_radius: int = 3
) -> str:
    lines = mlir_text.splitlines()
    
    diagnostic_lines = [
        "=" * 80,
        "XLS BACKEND VALIDATION FAILED",
        "=" * 80,
        "",
        f"Found {len(errors)} unsupported feature(s) in the MLIR module.",
        "The XLS[cc] backend only supports a subset of Allo features.",
        "",
        "-" * 80,
        "ERRORS:",
        "-" * 80,
    ]
    
    # Group errors by type for summary
    error_by_type: dict[str, list[ValidationError]] = {}
    for err in errors:
        error_by_type.setdefault(err.error_type, []).append(err)
    
    # Summary
    diagnostic_lines.append("")
    diagnostic_lines.append("SUMMARY:")
    for err_type, err_list in error_by_type.items():
        diagnostic_lines.append(f"  - {err_type}: {len(err_list)} occurrence(s)")
    diagnostic_lines.append("")
    
    # Detailed errors with line annotations
    diagnostic_lines.append("-" * 80)
    diagnostic_lines.append("ANNOTATED MLIR (showing problematic lines):")
    diagnostic_lines.append("-" * 80)
    diagnostic_lines.append("")
    
    # Build a set of error line numbers for quick lookup
    error_lines = {err.line_num: err for err in errors}
    
    # Show context around errors
    lines_to_show = set()
    for err in errors:
        for i in range(max(1, err.line_num - context_radius), 
                       min(len(lines) + 1, err.line_num + context_radius + 1)):
            lines_to_show.add(i)
    
    prev_shown = 0
    for line_num in sorted(lines_to_show):
        # Add separator if there's a gap
        if prev_shown > 0 and line_num > prev_shown + 1:
            diagnostic_lines.append("        ...")
        
        line_content = lines[line_num - 1]
        prefix = f"{line_num:6d}| "
        
        if line_num in error_lines:
            err = error_lines[line_num]
            diagnostic_lines.append(f"{prefix}{line_content}")
            # Add error annotation arrow
            diagnostic_lines.append(f"      |  ^^^^ ERROR: {err.message}")
            diagnostic_lines.append("")
        else:
            diagnostic_lines.append(f"{prefix}{line_content}")
        
        prev_shown = line_num
    
    # How to fix section
    diagnostic_lines.append("")
    diagnostic_lines.append("-" * 80)
    diagnostic_lines.append("HOW TO FIX:")
    diagnostic_lines.append("-" * 80)
    
    for err_type in error_by_type:
        if err_type in ERROR_TYPE_HELP:
            diagnostic_lines.append("")
            diagnostic_lines.extend(ERROR_TYPE_HELP[err_type])
    
    diagnostic_lines.append("")
    diagnostic_lines.append("=" * 80)
    
    return "\n".join(diagnostic_lines)


def write_diagnostic_file(
    errors: list[ValidationError],
    mlir_text: str,
    project: str
) -> str:
    os.makedirs(project, exist_ok=True)
    diag_path = f"{project}/xls_validation_errors.txt"
    
    report = generate_diagnostic_report(errors, mlir_text)
    
    with open(diag_path, 'w', encoding='utf-8') as f:
        f.write(report)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("FULL MLIR MODULE:\n")
        f.write("=" * 80 + "\n")
        f.write(mlir_text)
    
    return diag_path


def format_error_summary(
    errors: list[ValidationError],
    project: str = None
) -> str:
    # Group errors by type
    error_by_type: dict[str, list[ValidationError]] = {}
    for err in errors:
        error_by_type.setdefault(err.error_type, []).append(err)
    
    error_summary = []
    for err_type, err_list in error_by_type.items():
        sample = err_list[0]
        if len(err_list) == 1:
            error_summary.append(f"  - {err_type} at line {sample.line_num}: {sample.message}")
        else:
            error_summary.append(f"  - {err_type}: {len(err_list)} occurrences (first at line {sample.line_num})")
    
    msg = (
        f"MLIR module is not legal for XLS[cc] backend ({len(errors)} error(s)):\n"
        + "\n".join(error_summary)
    )
    
    if project:
        msg += f"\n\nSee {project}/xls_validation_errors.txt for detailed diagnostics."
    
    return msg