# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module,no-member

import io
import os
import re

from .._mlir.dialects import allo as allo_d
from .._mlir.ir import Context, Location, Module, UnitAttr
from .._mlir.passmanager import PassManager
from ..ir.transform import find_func_in_module
from ..passes import decompose_library_function, _mlir_lower_pipeline

# Regex patterns for floats, dynamic shapes, and fixed-point types
FLOAT_RE = re.compile(r"\b(f16|f32|f64|bf16)\b")
DYNAMIC_RE = re.compile(r"\b(memref|tensor)<[^>]*\?>")
FIXED_TYPE_RE = re.compile(r"!allo\.(Fixed|UFixed)<\d+,\s*\d+>")

# Manually designed Fixed Point Struct modeled via Fixed Integer type
# Emitted with rest of code when fixed-point types are used in the MLIR
FIXED_POINT_STRUCT = """
// XLS-compatible Fixed-Point Template
template <int WIDTH, int FRAC, bool SIGNED = true>
struct Fixed {
  ac_int<WIDTH, SIGNED> value;
  Fixed() : value(0) {}
  Fixed(int v) : value(static_cast<ac_int<WIDTH, SIGNED>>(v) << FRAC) {}
  static Fixed from_raw(ac_int<WIDTH, SIGNED> raw) { Fixed f; f.value = raw; return f; }
  int to_int() const { return static_cast<int>(value >> FRAC); }
  ac_int<WIDTH, SIGNED> raw() const { return value; }
  Fixed operator+(const Fixed& o) const { Fixed r; r.value = value + o.value; return r; }
  Fixed operator-(const Fixed& o) const { Fixed r; r.value = value - o.value; return r; }
  Fixed operator*(const Fixed& o) const {
    Fixed r;
    ac_int<WIDTH*2, SIGNED> t = static_cast<ac_int<WIDTH*2, SIGNED>>(value) *
                                 static_cast<ac_int<WIDTH*2, SIGNED>>(o.value);
    r.value = static_cast<ac_int<WIDTH, SIGNED>>(t >> FRAC);
    return r;
  }
  Fixed operator/(const Fixed& o) const {
    Fixed r;
    ac_int<WIDTH*2, SIGNED> t = static_cast<ac_int<WIDTH*2, SIGNED>>(value) << FRAC;
    r.value = static_cast<ac_int<WIDTH, SIGNED>>(t / o.value);
    return r;
  }
  bool operator==(const Fixed& o) const { return value == o.value; }
  bool operator<(const Fixed& o) const { return value < o.value; }
};
template <int WIDTH, int FRAC> using UFixed = Fixed<WIDTH, FRAC, false>;
"""


def _validate_xls_ir(mlir_text, project=None):
    errors = []
    for i, line in enumerate(mlir_text.splitlines(), 1):
        # Float types
        m = FLOAT_RE.search(line)
        if m:
            errors.append(f"Line {i}: Float type '{m.group()}'.")
        # Dynamic shapes (memref with ? dimension)
        if DYNAMIC_RE.search(line):
            errors.append(f"Line {i}: Dynamic shapes ('?').")

    if errors:
        err_msg = f"XLS [CC] validation failed ({len(errors)} errors):\n" + "\n".join(
            errors
        )
        # write error message to file if project is provided
        if project:
            os.makedirs(project, exist_ok=True)
            with open(f"{project}/xls_errors.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(errors))
            err_msg += f"\nSee {project}/xls_errors.txt"
        raise RuntimeError(err_msg)  # raise error


def _extract_body(code, func_name):
    func_head = rf"void\s+{re.escape(func_name)}\s*\(\s*\)\s*\{{"
    match = re.search(func_head, code)
    if not match:
        return f"// Could not extract body of {func_name}\n"

    # use brace matching to extract the function body
    start, depth, pos = match.end(), 1, match.end()
    while pos < len(code) and depth > 0:
        if code[pos] == "{":
            depth += 1
        elif code[pos] == "}":
            depth -= 1
        pos += 1

    return (
        code[start : pos - 1] if depth == 0 else f"// Unmatched braces in {func_name}\n"
    )


def _reindent(body, indent="    "):
    lines = body.split("\n")
    # Very large number, very unlikely to have that many indentations in the body
    min_indent = 2**20

    for ln in lines:
        if ln.strip():
            min_indent = min(len(ln) - len(ln.lstrip()), min_indent)

    # Set to 0 if no indentation is found
    if min_indent == 2**20:
        min_indent = 0

    result = []
    for ln in lines:
        if not ln.strip():
            result.append("")
        else:
            result.append(indent + ln[min_indent:])

    return "\n".join(result)


def _parse_memory_comments(body):
    # Regex patterns __xls_memory_decl__ and __xls_state_vars__
    mem_re = re.compile(r"^\s*//\s*__xls_memory_decl__:\s*(.+)$")
    state_re = re.compile(r"^\s*//\s*__xls_state_vars__:\s*(.+)$")

    # Collect memory declarations, state variables.
    decls, states = [], []
    mems = []  # Information about memories to generate XLS's RAM Config Files
    result = []  # final result with cleaned up body.

    for line in body.split("\n"):
        mem_m = mem_re.match(line)
        state_m = state_re.match(line)

        if mem_m:
            parts = [
                p.split(";")[0].split("//")[0].strip()
                for p in mem_m.group(1).split(",")
            ]
            if len(parts) >= 3:
                elem_type, size_str, name = parts[0], parts[1], parts[2]
                dims = [d for d in parts[3:] if d]
                decls.append(f"__xls_memory<{elem_type}, {size_str}> {name};")
                # Extract numeric size
                size_m = re.match(r"(\d+)", size_str)
                if size_m:
                    size = int(size_m.group(1))
                    # Size of array dims, e.g. [100, 100]
                    int_dims = [
                        int(re.match(r"(\d+)", d).group(1))
                        for d in dims
                        if re.match(r"(\d+)", d)
                    ]
                    mems.append((name, elem_type, size, int_dims or [size]))
        elif state_m:
            for v in state_m.group(1).split(","):
                v = v.split(";")[0].split("//")[0].strip()
                if v:
                    states.append(f"int {v} = 0;")
        else:
            result.append(line)

    return decls, states, "\n".join(result), mems


# Generates the RAM configuration required by XLS for the memory interface
def _gen_textproto(mems):
    """Generate RAM rewrites textproto for memories."""
    if not mems:
        return ""
    lines = [
        "# Automatocally Generated by Allo (Allo XLS [CC] Backend)",
        "# proto-file: xls/codegen/ram_rewrite.proto",
        "# proto-message: Ram Configuration Files",
        "",  # empty line
    ]
    for name, _, size, _ in mems:
        # Default to 1R1W RAM, XLS memory uses val/rdy interface hence the4 reqs/resps channels
        lines.append(
            f"""rewrites {{
  from_config {{ kind: RAM_ABSTRACT depth: {size} }}
  to_config {{ kind: RAM_1R1W depth: {size} }}
  from_channels_logical_to_physical: {{ key: "abstract_read_req" value: "{name}_read_request" }}
  from_channels_logical_to_physical: {{ key: "abstract_read_resp" value: "{name}_read_response" }}
  from_channels_logical_to_physical: {{ key: "abstract_write_req" value: "{name}_write_request" }}
  from_channels_logical_to_physical: {{ key: "write_completion" value: "{name}_write_response" }}
  to_name_prefix: "{name}_"
}}
"""
        )
    return "\n".join(lines)


def _render_testblock(
    in_chans, out_chans, body, top_name, mem_decls=None, state_decls=None
):
    """Render TestBlock class with channels and optional memory/state declarations."""

    # Templated channels for inputs and outputs
    in_decl = "\n  ".join(
        f"__xls_channel<int, __xls_channel_dir_In> {c};" for c in in_chans
    )
    out_decl = "\n  ".join(
        f"__xls_channel<int, __xls_channel_dir_Out> {c};" for c in out_chans
    )
    indented = _reindent(body)  # reindent body to match TestBlock class indentation

    mem_section = ""
    if mem_decls:
        mem_section = "\n  // Memory\n  " + "\n  ".join(mem_decls) + "\n"

    state_section = ""
    if state_decls:
        state_section = "\n  // State\n  " + "\n  ".join(state_decls) + "\n"

    return f"""class TestBlock {{
public:
  {in_decl}
  {out_decl}
{mem_section}{state_section}
  #pragma hls_top
  void {top_name}() {{
{indented}
  }}
}};
"""


def _wrap_xlscc(core_code, top_name, func_name, inputs, use_memory):
    # No array inputs -> combinational, return without channels, memory, TestBlock class
    if not inputs:
        return core_code, ""

    # Extract function body and headers
    body = _extract_body(core_code, func_name)
    match = re.search(rf"void\s+{re.escape(func_name)}\s*\(\s*\)\s*\{{", core_code)
    headers = core_code[: match.start()] if match else ""

    # Parse memory declarations from body
    mem_decls, state_decls, result, mems = _parse_memory_comments(body)

    # Generate channel names
    in_chans = [f"v{i}_in" for i in range(len(inputs))]
    out_chans = ["out"]

    # Render testblock
    if use_memory:
        testblock = _render_testblock(
            in_chans, out_chans, result, top_name, mem_decls, state_decls
        )
        textproto = _gen_textproto(mems)
    else:
        testblock = _render_testblock(in_chans, out_chans, body, top_name)
        textproto = ""

    return headers + testblock, textproto


class XLSCCModule:
    def __init__(
        self, mlir_text_or_module, top_func_name, project=None, use_memory=False
    ):
        self.top_func_name = top_func_name
        self.project = project
        self.use_memory = use_memory

        # Parse MLIR + run minimal lowering
        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mlir_text_or_module), ctx)

            self.module = decompose_library_function(self.module)
            _mlir_lower_pipeline(self.module, lower_linalg=True)
            # Run through lowering passes
            pm = PassManager.parse(
                "builtin.module("
                "empty-tensor-to-alloc-tensor,"
                "func.func(convert-linalg-to-affine-loops)"
                ")"
            )
            pm.run(self.module.operation)

            self.func = find_func_in_module(self.module, top_func_name)
            if self.func:
                self.func.attributes["top"] = UnitAttr.get()

        self.mlir_text = str(self.module)

        # validate XLS IR does not contain unsupported features
        _validate_xls_ir(self.mlir_text, project)

        # Emit Core XLS C++ from MLIR
        buf = io.StringIO()
        allo_d.emit_xhls(self.module, buf, use_memory)
        buf.seek(0)
        self.core_code = buf.read()

        # Get array parameter names
        param_names = []
        if self.func:
            for arg in self.func.arguments:
                t = str(arg.type)
                if ("memref" in t or "tensor" in t) and "stream" not in t.lower():
                    param_names.append(f"arg{len(param_names)}")

        # Add fixed-point struct if needed
        fixed_header = (
            FIXED_POINT_STRUCT if FIXED_TYPE_RE.search(self.mlir_text) else ""
        )

        # Use _wrap_xlscc to handle C++ templated memory/channels, headers, top pragma, etc. as well
        # as RAM Configuration File (rewrites.textproto) which are not handled by emit_xls
        cpp, self.rewrites_textproto = _wrap_xlscc(
            self.core_code, top_func_name, top_func_name, param_names, use_memory
        )
        self.final_cpp = fixed_header + cpp

        # Write output (C++ and RAM Configuration File) files if project is provided
        if project:
            os.makedirs(project, exist_ok=True)
            with open(f"{project}/test_block.cpp", "w", encoding="utf-8") as f:
                f.write(self.final_cpp)
            if use_memory and self.rewrites_textproto:
                with open(f"{project}/rewrites.textproto", "w", encoding="utf-8") as f:
                    f.write(self.rewrites_textproto)

    def __repr__(self):
        return self.final_cpp

    def print_textproto(self):
        """Print RAM rewrites textproto if available."""
        if self.rewrites_textproto:
            print("=" * 60)
            print("RAM REWRITES TEXTPROTO:")
            print("=" * 60)
            print(self.rewrites_textproto)
        else:
            print("No RAM rewrites (not in memory mode)")
