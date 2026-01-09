"""Serializer for DSLX Proc AST to DSLX text.

Converts AST nodes to properly formatted DSLX code.
"""

from .proc_ast import *


class DslxProcSerializer:
    """Serializes DSLX Proc AST to text."""

    def __init__(self, indent_width=2):
        self.indent_width = indent_width
        self.indent_level = 0

    def serialize(self, node):
        """Serialize any AST node to DSLX text."""
        if isinstance(node, DslxModule):
            return self._serialize_module(node)
        elif isinstance(node, DslxProc):
            return self._serialize_proc(node)
        elif isinstance(node, DslxImport):
            return self._serialize_import(node)
        elif isinstance(node, DslxTypeAlias):
            return self._serialize_type_alias(node)
        elif isinstance(node, DslxExpr):
            return self._serialize_expr(node)
        elif isinstance(node, DslxStmt):
            return self._serialize_stmt(node)
        else:
            return str(node)

    def _indent(self):
        return " " * (self.indent_level * self.indent_width)

    def _serialize_module(self, module):
        """Serialize a complete module."""
        lines = []

        # Imports
        for imp in module.imports:
            lines.append(self._serialize_import(imp))

        # Type aliases
        if module.type_aliases:
            if module.imports:
                lines.append("")
            for alias in module.type_aliases:
                lines.append(self._serialize_type_alias(alias))

        # Procs
        for proc in module.procs:
            lines.append("")
            if lines[-1] != "":
                lines.append("")
            lines.append(self._serialize_proc(proc))

        return "\n".join(lines)

    def _serialize_import(self, imp):
        """Serialize import statement."""
        return f"import {imp.module_name};"

    def _serialize_type_alias(self, alias):
        """Serialize type alias."""
        return f"type {alias.alias_name} = {alias.target_type};"

    def _serialize_proc(self, proc):
        """Serialize proc definition."""
        lines = []

        # Proc header
        header = "pub proc" if proc.is_public else "proc"
        header += f" {proc.name}"

        if proc.type_params:
            params_str = ", ".join(f"{p.name}: {p.param_type}" for p in proc.type_params)
            header += f"<{params_str}>"

        header += " {"
        lines.append(header)

        self.indent_level += 1

        # Channel declarations
        for chan in proc.channels:
            # Handle array channel declarations
            if chan.array_dims:
                dims_str = "".join(f"[{dim}]" for dim in chan.array_dims)
                chan_str = f"{self._indent()}{chan.name}: chan<{chan.chan_type}>{dims_str} {chan.direction};"
            else:
                chan_str = f"{self._indent()}{chan.name}: chan<{chan.chan_type}> {chan.direction};"
            lines.append(chan_str)

        if proc.channels:
            lines.append("")

        # Config function
        if proc.config:
            lines.append(self._serialize_config(proc.config))
            lines.append("")

        # Init function
        if proc.init:
            lines.append(self._serialize_init(proc.init))
            lines.append("")

        # Next function
        if proc.next_func:
            lines.append(self._serialize_next(proc.next_func))

        self.indent_level -= 1
        lines.append("}")

        return "\n".join(lines)

    def _serialize_config(self, config):
        """Serialize config function."""
        lines = []

        # Config header
        params_strs = []
        for param in config.params:
            if len(param) == 4:
                name, chan_type, direction, array_dims = param
                dims_str = "".join(f"[{dim}]" for dim in array_dims)
                params_strs.append(f"{name}: chan<{chan_type}>{dims_str} {direction}")
            else:
                name, chan_type, direction = param
                params_strs.append(f"{name}: chan<{chan_type}> {direction}")

        # Split into multiple lines if too long
        if len(params_strs) <= 3:
            params_line = ", ".join(params_strs)
            lines.append(f"{self._indent()}config({params_line}) {{")
        else:
            lines.append(f"{self._indent()}config(")
            self.indent_level += 1
            for i, param_str in enumerate(params_strs):
                comma = "," if i < len(params_strs) - 1 else ""
                lines.append(f"{self._indent()}{param_str}{comma}")
            self.indent_level -= 1
            lines.append(f"{self._indent()}) {{")

        # Config body - handle both DslxBlock and simple expressions
        self.indent_level += 1
        if isinstance(config.body, DslxBlock):
            # Body is a block of statements
            for stmt in config.body.stmts:
                stmt_str = self._serialize_stmt(stmt)
                if stmt_str:
                    lines.append(f"{self._indent()}{stmt_str}")
        else:
            # Body is a single expression (backward compatibility)
            body_str = self._serialize_expr(config.body)
            lines.append(f"{self._indent()}{body_str}")
        self.indent_level -= 1

        lines.append(f"{self._indent()}}}")

        return "\n".join(lines)

    def _serialize_init(self, init):
        """Serialize init function."""
        init_str = self._serialize_expr(init.init_expr)
        return f"{self._indent()}init {{ {init_str} }}"

    def _serialize_next(self, next_func):
        """Serialize next function."""
        lines = []

        lines.append(f"{self._indent()}next(state: {next_func.state_type}) {{")

        self.indent_level += 1
        if isinstance(next_func.body, DslxBlock):
            for stmt in next_func.body.stmts:
                stmt_str = self._serialize_stmt(stmt)
                if stmt_str:
                    lines.append(f"{self._indent()}{stmt_str}")
        else:
            # Body is a single expression
            body_str = self._serialize_expr(next_func.body)
            lines.append(f"{self._indent()}{body_str}")
        self.indent_level -= 1

        lines.append(f"{self._indent()}}}")

        return "\n".join(lines)

    def _serialize_expr(self, expr):
        """Serialize expression."""
        if isinstance(expr, DslxLiteral):
            if expr.lit_type:
                return f"{expr.lit_type}:{expr.value}"
            return str(expr.value)

        elif isinstance(expr, DslxVar):
            return expr.name

        elif isinstance(expr, DslxFuncCall):
            args_str = ", ".join(self._serialize_expr(arg) for arg in expr.args)
            return f"{expr.func_name}({args_str})"

        elif isinstance(expr, DslxBinOp):
            lhs = self._serialize_expr(expr.lhs)
            rhs = self._serialize_expr(expr.rhs)
            return f"{lhs} {expr.op} {rhs}"

        elif isinstance(expr, DslxTuple):
            elements_str = ", ".join(self._serialize_expr(e) for e in expr.elements)
            return f"({elements_str})"

        elif isinstance(expr, DslxArrayLiteral):
            # For nested arrays, serialize inner array literals without type annotation
            elements = []
            for e in expr.elements:
                if isinstance(e, DslxArrayLiteral):
                    # Nested array - serialize elements without type prefix
                    inner_elements = ", ".join(self._serialize_expr(ie) for ie in e.elements)
                    elements.append(f"[{inner_elements}]")
                else:
                    elements.append(self._serialize_expr(e))
            elements_str = ", ".join(elements)
            # Only add type prefix if elem_type is set and this is the outermost array
            # For now, never add type prefix for array literals
            return f"[{elements_str}]"

        elif isinstance(expr, DslxChannelOp):
            args_str = ", ".join(self._serialize_expr(arg) for arg in expr.args)
            return f"{expr.op_type}({args_str})"

        elif isinstance(expr, DslxIf):
            cond = self._serialize_expr(expr.condition)

            # Handle then branch - could be Block or simple expression
            if isinstance(expr.then_expr, DslxBlock):
                then_lines = []
                self.indent_level += 1
                for stmt in expr.then_expr.stmts:
                    stmt_str = self._serialize_stmt(stmt)
                    if stmt_str:
                        then_lines.append(f"{self._indent()}{stmt_str}")
                self.indent_level -= 1
                then_body = "\n".join(then_lines)
            else:
                then_body = f"{self._indent()}  {self._serialize_expr(expr.then_expr)}"

            # Handle else branch - could be Block or simple expression
            if isinstance(expr.else_expr, DslxBlock):
                else_lines = []
                self.indent_level += 1
                for stmt in expr.else_expr.stmts:
                    stmt_str = self._serialize_stmt(stmt)
                    if stmt_str:
                        else_lines.append(f"{self._indent()}{stmt_str}")
                self.indent_level -= 1
                else_body = "\n".join(else_lines)
            else:
                else_body = f"{self._indent()}  {self._serialize_expr(expr.else_expr)}"

            return f"if {cond} {{\n{then_body}\n{self._indent()}}} else {{\n{else_body}\n{self._indent()}}}"

        elif isinstance(expr, DslxArrayIndex):
            array_str = self._serialize_expr(expr.array)
            indices_str = "][".join(self._serialize_expr(idx) for idx in expr.indices)
            return f"{array_str}[{indices_str}]"

        else:
            return str(expr)

    def _serialize_stmt(self, stmt):
        """Serialize statement."""
        if isinstance(stmt, DslxLet):
            pattern = self._serialize_expr(stmt.pattern)
            expr = self._serialize_expr(stmt.expr)
            return f"let {pattern} = {expr};"

        elif isinstance(stmt, DslxChannelCreate):
            # Build type string with optional FIFO depth
            type_str = stmt.chan_type
            if stmt.fifo_depth:
                type_str = f"{stmt.chan_type}, {stmt.fifo_depth}"

            # Handle array channel creation
            if stmt.array_dims:
                dims_str = "".join(f"[{dim}]" for dim in stmt.array_dims)
                return f"let ({stmt.sender_name}, {stmt.receiver_name}) = chan<{type_str}>{dims_str}(\"{stmt.label}\");"
            else:
                return f"let ({stmt.sender_name}, {stmt.receiver_name}) = chan<{type_str}>(\"{stmt.label}\");"

        elif isinstance(stmt, DslxSpawn):
            args_str = ", ".join(self._serialize_expr(arg) for arg in stmt.args)
            if stmt.type_params:
                type_params_str = ", ".join(self._serialize_expr(tp) for tp in stmt.type_params)
                return f"spawn {stmt.proc_name}<{type_params_str}>({args_str});"
            else:
                return f"spawn {stmt.proc_name}({args_str});"

        elif isinstance(stmt, DslxUnrollFor):
            lines = []
            # unroll_for! (row, tok): (u32, token) in u32:0..ROWS {
            vars_str = ", ".join(f"{name}" for name, _ in stmt.loop_vars)
            types_str = ", ".join(f"{typ}" for _, typ in stmt.loop_vars)
            lines.append(f"unroll_for! ({vars_str}): ({types_str}) in {stmt.range_expr} {{")

            # Body
            self.indent_level += 1
            if isinstance(stmt.body, list):
                for body_stmt in stmt.body:
                    stmt_str = self._serialize_stmt(body_stmt)
                    if stmt_str:
                        lines.append(f"{self._indent()}{stmt_str}")
            elif isinstance(stmt.body, DslxBlock):
                for body_stmt in stmt.body.stmts:
                    stmt_str = self._serialize_stmt(body_stmt)
                    if stmt_str:
                        lines.append(f"{self._indent()}{stmt_str}")
            self.indent_level -= 1

            # Closing with init expression
            init_str = self._serialize_expr(stmt.init_expr)
            lines.append(f"}}({init_str});")
            return "\n".join(lines)

        elif isinstance(stmt, DslxConst):
            value_str = self._serialize_expr(stmt.value)
            return f"const {stmt.name} = {value_str};"

        elif isinstance(stmt, DslxExpr):
            # Statement that's just an expression
            return self._serialize_expr(stmt)

        else:
            return str(stmt)
