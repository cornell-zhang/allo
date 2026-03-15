# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import copy
from dataclasses import dataclass
import hashlib
import inspect
import numpy as np
from pathlib import Path
from collections.abc import Callable
from types import FunctionType as PyFunctionType
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from allo.ir.types import AlloType
from allo.memory import Memory


@dataclass(frozen=True)
class ErrorMsg:
    msg: Exception  # The exception or error to report.
    node: ast.AST  # The AST node associated with the error (must have lineno).
    source_file: str  # Path to source code.


def report_error(
    error: ErrorMsg,
    context: int = 3,
) -> None:
    """
    Report an error with source location from AST node.

    Uses lineno/col_offset from the AST node to display a rich-formatted
    error message pointing at the offending source line.

    Args:
        error: Error information.
        context: Number of context lines to show above/below the error line.
    """
    node = error.node
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    end_lineno = getattr(node, "end_lineno", lineno)
    end_col_offset = getattr(node, "end_col_offset", None)
    source_lines = Path(error.source_file).read_text().splitlines()

    console = Console(stderr=True)

    # Build the source display
    if source_lines and 1 <= lineno <= len(source_lines):
        start = max(0, lineno - 1 - context)
        end = min(len(source_lines), end_lineno + context)
        snippet_lines = []
        for i in range(start, end):
            line_num = i + 1
            line_text = source_lines[i].replace("[", r"\[")
            if line_num == lineno:
                snippet_lines.append(
                    f"[bold red]{line_num:5d} | {line_text}[/bold red]"
                )
                # column marker
                if col_offset is not None:
                    marker_end = (
                        end_col_offset
                        if (end_col_offset and end_lineno == lineno)
                        else col_offset + 1
                    )
                    padding = " " * (8 + col_offset)
                    underline = "^" * max(1, marker_end - col_offset)
                    snippet_lines.append(f"[bold red]{padding}{underline}[/bold red]")
            else:
                snippet_lines.append(f"{line_num:5d} | {line_text}")

        error_text = Text.from_markup(f"[bold red]Error:[/bold red] {error.msg}")
        panel = Panel(
            "\n".join(snippet_lines),
            title=f"[bold yellow]Line {lineno}[/bold yellow]",
            subtitle="Source Code",
            border_style="red",
        )
        console.print(error_text)
        console.print(panel)
    else:
        loc = f" (line {lineno})" if lineno else ""
        console.print(f"[bold red]Error:[/bold red] {error.msg}{loc}")


def get_ast(src) -> ast.FunctionDef:
    assert hasattr(src, "_ast") and hasattr(src, "_type"), "Invalid function"
    node = copy.deepcopy(src._ast)  # get a new copy to avoid overwriting
    node._type = src._type
    node._source = src._source
    return node


class SymbolTable:
    def __init__(self):
        # function name -> function instance (instantiated from templates) node
        self.functions = {}
        # global: constant name -> constant value
        self.constants = {}
        # global: variable name -> variable node
        self.variables = {}

        self.types = {}  # str(dtype) -> AlloType / refinement
        self.global_symbols = {}  # str -> python object

        self.global_ops = []
        # ----- tools -----
        self.tmpl_instantiations = {}  # template name -> instance args -> instance name

    def mangle_template_name(self, name: str, args: list) -> str:
        """
        Name mangling for instantiated functions.

        Args:
            name: The name of the function.
            args: The arguments to instantiate the function. If the last argument is a string, it is used as the user defined suffix of the instantiated function.

        Returns:
            The name of the instantiated function.
        """
        if isinstance(args[-1], str):
            suffix = args.pop()
            return "_" + name + "_" + suffix
        key = tuple(args)
        func_dict = self.tmpl_instantiations.setdefault(name, {})
        if key not in func_dict:
            func_dict[key] = (
                "_" + name + "_" + "_".join(map(str, args)) + "_" + str(len(func_dict))
            )
        return func_dict[key]

    def mangle_with_namespace(self, name: str, namespace: str) -> str:
        return f"{namespace}.{name}"

    def mangle_grid_name(self, work_name) -> str:
        return f"{work_name}.mesh"

    @staticmethod
    def get_namespace(name) -> str:
        return name.partition(".")[0]

    @staticmethod
    def get_hash(arr):
        assert isinstance(arr, np.ndarray), "only support np.ndarray"
        return hashlib.sha256(
            arr.tobytes() + str((arr.shape, arr.dtype)).encode()
        ).hexdigest()[:16]


def get_global_vars(func):
    def _get_global_vars(_func, skip={"get_global_vars", "process"}, stop={"<module>"}):
        if isinstance(_func, Callable):
            # Discussions: https://github.com/taichi-dev/taichi/issues/282
            global_vars = _func.__globals__.copy()
        else:
            global_vars = {}

        # Get back to outer scopes
        # Mainly used to get the annotation definitions (shape and type),
        # which are probably not defined in __globals__
        frame = inspect.currentframe().f_back
        while frame:
            if frame.f_code.co_name in skip:
                frame = frame.f_back
                continue
            # collect allowed types
            for name, var in frame.f_locals.items():
                # FIXME: find a better way to collect required symbols
                if isinstance(
                    var, (int, float, AlloType, Memory, list)
                ) or inspect.isfunction(var):
                    global_vars[name] = var
            # boundary
            if frame.f_code.co_name in stop:
                break
            frame = frame.f_back

        if isinstance(_func, Callable):
            freevar_names = _func.__code__.co_freevars
            closure = _func.__closure__
            if closure:
                freevar_values = [x.cell_contents for x in closure]
                for name, value in zip(freevar_names, freevar_values):
                    global_vars[name] = value
        return global_vars

    all_globals = {}
    worklist = [func]
    visited_funcs = set()

    while worklist:
        f = worklist.pop()
        if f in visited_funcs:
            continue
        visited_funcs.add(f)

        gv = _get_global_vars(f)
        for name, val in gv.items():
            if name not in all_globals:
                all_globals[name] = val
                # import functions from other files
                if isinstance(val, PyFunctionType):
                    worklist.append(val)

    return all_globals


class Scope:
    def __init__(self):
        self.consts = {}
        self.vars = {}


class ErrorValue:
    def __init__(self, name, msg):
        self.name = name
        self.msg = msg

    def __getattr__(self, attr):
        raise RuntimeError(f"Use of invalid symbol '{self.name}': {self.msg}")
