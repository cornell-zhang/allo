# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from rich.console import Console
from rich.text import Text
from rich.panel import Panel


def print_error_message(error, stmt, tree):
    console = Console()

    # Display only 10 lines of the source code, and highlight the error line
    source_code = ast.unparse(tree)
    source_stmt = ast.unparse(stmt)
    strip_lines = [line.strip() for line in source_code.splitlines()]
    target_idx = strip_lines.index(source_stmt.splitlines()[0].strip())
    line_number = stmt.lineno
    start_offset = min(target_idx, 5)
    code_lines = source_code.splitlines()[target_idx - 5 : target_idx + 5]
    highlighted_code = []
    for idx, line in enumerate(code_lines, start=line_number - start_offset):
        line = line.replace("[", r"\[")
        if idx == line_number:
            highlighted_code.append(f"[bold red]{idx:4}: {line}[/bold red]")
        else:
            highlighted_code.append(f"{idx:4}: {line}")

    # Format the error message with the traceback
    error_message = Text.from_markup(f"[bold red]Error:[/bold red] {error}")
    line_info = Text(f"Line: {line_number}", style="bold yellow")
    full_error = Panel(
        "\n".join(highlighted_code),
        title=f"[bold red]Traceback (most recent call last):[/]\n[bold yellow]{line_info}",
        subtitle="Source Code",
        border_style="red",
    )

    # Print the error message and highlighted code
    console.print(error_message)
    console.print(full_error)
