# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module,no-member

import io
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

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


#  Check if g++ is available for sw_emu compilation
def is_available():
    try:
        result = subprocess.run(["g++", "--version"], capture_output=True, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _validate_xls_ir(mlir_text, project=None):
    errors = []
    for i, line in enumerate(mlir_text.splitlines(), 1):
        # Float types
        m = FLOAT_RE.search(line)
        if m:
            errors.append(
                f"Line {i}: Floating-point type '{m.group()}' is not supported by XLS [CC] backend. "
                "Please use integer or fixed-point types instead."
            )
        # Dynamic shapes (memref with ? dimension)
        if DYNAMIC_RE.search(line):
            errors.append(
                f"Line {i}: Dynamic shapes (with '?' dimension) are not supported by the XLS [CC] backend. "
                "All array dimensions must be statically known."
            )

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
            # Format: elem_type, size, name, memory_space, dim0, dim1, ...
            if len(parts) < 4:
                continue  # Invalid format, skip

            elem_type, size_str, name, mem_space_str = (
                parts[0],
                parts[1],
                parts[2],
                parts[3],
            )
            dims = [d for d in parts[4:] if d]

            decls.append(f"__xls_memory<{elem_type}, {size_str}> {name};")

            size_m = re.match(r"(\d+)", size_str)
            mem_space_m = re.match(r"(\d+)", mem_space_str)
            if size_m:
                size = int(size_m.group(1))
                memory_space = int(mem_space_m.group(1)) if mem_space_m else 0
                # Decode: memory_space = resource_code * 16 + storage_type_code
                storage_type_code = memory_space % 16
                # Default to RAM_2P (dual port, maps to RAM_1R1W in XLS) when not specified
                if storage_type_code == 0:
                    storage_type_code = 2

                int_dims = [
                    int(re.match(r"(\d+)", d).group(1))
                    for d in dims
                    if re.match(r"(\d+)", d)
                ]
                mems.append(
                    (name, elem_type, size, int_dims or [size], storage_type_code)
                )

        elif state_m:
            for v in state_m.group(1).split(","):
                v = v.split(";")[0].split("//")[0].strip()
                if v:
                    states.append(f"int {v} = 0;")
        else:
            result.append(line)

    return decls, states, "\n".join(result), mems


# Map storage_type_code to XLS RAM kind. Default is RAM_2P.
STORAGE_TYPE_TO_XLS_RAM = {
    1: "RAM_1RW",  # RAM_1P (single port)
    2: "RAM_1R1W",  # RAM_2P (dual port) - default
    5: "RAM_1R1W",  # RAM_S2P (alias of RAM_2P)
    6: "RAM_1RW",  # ROM_1P (single port ROM)
}

# Mapping from storage_type_code to human-readable name for error messages
STORAGE_TYPE_CODE_TO_NAME = {
    1: "RAM_1P",
    2: "RAM_2P",
    3: "RAM_T2P",
    4: "RAM_1WNR",
    5: "RAM_S2P",
    6: "ROM_1P",
    7: "ROM_2P",
    8: "ROM_NP",
}

# Supported storage types for XLS backend (1=RAM_1P, 2=RAM_2P, 6=ROM_1P)
SUPPORTED_STORAGE_TYPES = {1, 2, 6}


# Generates the RAM configuration required by XLS for the memory interface
def _gen_textproto(mems):
    if not mems:
        return ""
    lines = [
        "# Automatically Generated by Allo (Allo XLS [CC] Backend)",
        "# proto-file: xls/codegen/ram_rewrite.proto",
        "# proto-message: Ram Configuration Files",
        "",
    ]
    for name, _, size, _, storage_type_code in mems:
        # Validate storage_type_code is supported by XLS backend
        if storage_type_code not in SUPPORTED_STORAGE_TYPES:
            storage_type_name = STORAGE_TYPE_CODE_TO_NAME.get(
                storage_type_code, f"Unknown (code {storage_type_code})"
            )
            supported_names = ", ".join(
                STORAGE_TYPE_CODE_TO_NAME.get(c, str(c))
                for c in sorted(SUPPORTED_STORAGE_TYPES)
            )
            raise RuntimeError(
                f"XLS [CC] validation failed: Memory '{name}' uses storage_type '{storage_type_name}' "
                f"which is not supported by the XLS backend. "
                f"XLS only supports RAM_1RW and RAM_1R1W configurations. "
                f"Supported storage_types are: {supported_names}, or leave unspecified for default (RAM_1R1W)."
            )

        # Determine XLS RAM kind based on storage_type_code
        xls_ram_kind = STORAGE_TYPE_TO_XLS_RAM[storage_type_code]

        # Use size (array size) as depth by default
        depth = size

        # Determine channel mappings based on RAM kind
        if xls_ram_kind == "RAM_1RW":
            # Single port RAM - shared channels for read and write
            read_req = f"{name}_req"
            read_resp = f"{name}_resp"
            write_req = f"{name}_req"
            write_resp = f"{name}_resp"
        else:
            # Dual port RAM (RAM_1R1W) - separate read and write channels
            read_req = f"{name}_read_request"
            read_resp = f"{name}_read_response"
            write_req = f"{name}_write_request"
            write_resp = f"{name}_write_response"

        # Generate rewrite configuration (only difference is to_config kind)
        lines.append(
            f"""rewrites {{
  from_config {{ kind: RAM_ABSTRACT depth: {depth} }}
  to_config {{ kind: {xls_ram_kind} depth: {depth} }}
  from_channels_logical_to_physical: {{ key: "abstract_read_req" value: "{read_req}" }}
  from_channels_logical_to_physical: {{ key: "abstract_read_resp" value: "{read_resp}" }}
  from_channels_logical_to_physical: {{ key: "abstract_write_req" value: "{write_req}" }}
  from_channels_logical_to_physical: {{ key: "write_completion" value: "{write_resp}" }}
  to_name_prefix: "{name}_"
}}
"""
        )
    return "\n".join(lines)


# Render TestBlock class with channels and optional memory/state declarations.
def _render_testblock(
    in_chans, out_chans, body, top_name, mem_decls=None, state_decls=None
):

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


class XLSCCModule:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        mlir_text_or_module,
        top_func_name,
        project=None,
        use_memory=False,
        mode=None,
    ):
        self.top_func_name = top_func_name
        self.project = project
        self.use_memory = use_memory
        self.mode = mode
        self._binary_path = None
        self._project_dir = project
        self._arg_info = []
        self._has_arrays = False
        self._output_info = None

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

        # Extract argument info for sw_emu
        self._extract_arg_info()

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

        # Build sw_emu binary if requested
        if mode == "sw_emu":
            self._build_sw_emu()

    def _extract_arg_info(self):
        if not self.func:
            return

        for i, arg in enumerate(self.func.arguments):
            t = str(arg.type)
            info = {"index": i, "type_str": t, "is_array": False}

            if "memref" in t or "tensor" in t:
                info["is_array"] = True
                self._has_arrays = True
                match = re.search(r"<([^>]+)>", t)
                if match:
                    parts = match.group(1).split("x")
                    if len(parts) >= 2:
                        info["shape"] = [int(p) for p in parts[:-1]]
                        info["dtype"] = parts[-1]
                        info["size"] = 1
                        for dim in info["shape"]:
                            info["size"] *= dim
            else:
                info["dtype"] = t

            self._arg_info.append(info)

        # Extract output info from function return type
        self._extract_output_info()

    # Extract output shape and dtype from function return type.
    def _extract_output_info(self):
        if not self.func:
            return
        result_types = list(self.func.type.results)
        if not result_types:
            return
        ret_t = str(result_types[0])
        if "memref" not in ret_t and "tensor" not in ret_t:
            return
        match = re.search(r"<([^>]+)>", ret_t)
        if not match:
            return
        parts = match.group(1).split("x")
        if len(parts) < 2:
            return
        self._output_info = {
            "shape": [int(p) for p in parts[:-1]],
            "dtype": parts[-1],
        }
        self._output_info["size"] = 1
        for dim in self._output_info["shape"]:
            self._output_info["size"] *= dim

    def _get_harness_path(self):
        return Path(__file__).parent.parent / "harness" / "xlscc"

    def _preprocess_cpp(self, code):
        lines = []
        all_lines = code.split("\n")
        skip_next = False
        for i, line in enumerate(all_lines):
            if skip_next:
                skip_next = False
                continue
            # Skip XLS-specific includes
            if "#include" in line and (
                "xls" in line.lower() or "ac_int" in line or "ac_channel" in line
            ):
                continue
            # Skip template line if next line is using ac_int (they come as a pair)
            if line.strip().startswith("template") and i + 1 < len(all_lines):
                next_line = all_lines[i + 1]
                if "using ac_int" in next_line or "using ac_uint" in next_line:
                    skip_next = True
                    continue
            # Skip using ac_int/ac_uint lines
            if "using ac_int" in line or "using ac_uint" in line:
                continue
            lines.append(line)
        return "\n".join(lines)

    def _generate_harness_scalar(self):
        harness_path = self._get_harness_path()
        processed = self._preprocess_cpp(self.final_cpp)

        harness = f"""// Auto-generated sw_emu harness (scalar)
#include "{harness_path}/XlsInt.h"
#include "{harness_path}/Channel.h"
#include "{harness_path}/Memory.h"
#include <iostream>
#include <cstdlib>

{processed}

int main(int argc, char** argv) {{
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <inputs...>" << std::endl;
        return 1;
    }}

"""
        args = []
        for i, info in enumerate(self._arg_info):
            if not info["is_array"]:
                harness += f"    int arg{i} = std::atoi(argv[{i + 1}]);\n"
                args.append(f"arg{i}")

        if args:
            harness += f"\n    auto result = {self.top_func_name}({', '.join(args)});\n"
            harness += '    std::cout << "RESULT:" << result << std::endl;\n'
        else:
            harness += f"\n    {self.top_func_name}();\n"
            harness += '    std::cout << "RESULT:OK" << std::endl;\n'

        harness += "    return 0;\n}\n"
        return harness

    def _generate_harness_array(self):
        harness_path = self._get_harness_path()
        processed = self._preprocess_cpp(self.final_cpp)
        output_size = self._output_info["size"] if self._output_info else 1

        harness = f"""// Auto-generated sw_emu harness (array)
#include "{harness_path}/XlsInt.h"
#include "{harness_path}/Channel.h"
#include "{harness_path}/Memory.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

{processed}

int main(int argc, char** argv) {{
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <project_dir>" << std::endl;
        return 1;
    }}
    std::string project_dir = argv[1];
    TestBlock block;

"""
        for i, info in enumerate(self._arg_info):
            if info["is_array"]:
                size = info.get("size", 1)
                harness += f"""
    std::ifstream in{i}(project_dir + "/input{i}.data", std::ios::binary);
    if (!in{i}.is_open()) {{
        std::cerr << "Failed to open input{i}.data" << std::endl;
        return 1;
    }}
    std::vector<int32_t> input{i}({size});
    in{i}.read(reinterpret_cast<char*>(input{i}.data()), {size} * sizeof(int32_t));
    in{i}.close();
    for (int j = 0; j < {size}; ++j) block.v{i}_in.write(input{i}[j]);
"""

        harness += f"""
    block.{self.top_func_name}();

    std::cout << "RESULT:";
    for (int j = 0; j < {output_size}; ++j) {{
        int32_t val = block.out.read();
        std::cout << val;
        if (j < {output_size} - 1) std::cout << ",";
    }}
    std::cout << std::endl;
    return 0;
}}
"""
        return harness

    def _generate_makefile(self):
        """Generate a Makefile for the sw_emu project."""
        harness_path = self._get_harness_path()
        makefile = f"""# Auto-generated Makefile for XLS sw_emu
# Generated by Allo XLS [CC] Backend

CXX = g++
CXXFLAGS = -std=c++17 -O2
INCLUDES = -I{harness_path}

TARGET = test_binary
SRCS = test_harness.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
\t$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

%.o: %.cpp
\t$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

run: $(TARGET)
"""
        if self._has_arrays:
            makefile += "\t./$(TARGET) .\n"
        else:
            makefile += "\t./$(TARGET)\n"

        makefile += """
clean:
\trm -f $(TARGET) $(OBJS)

help:
\t@echo "Makefile targets:"
\t@echo "  all   - Build the test binary (default)"
\t@echo "  run   - Build and run the test binary"
\t@echo "  clean - Remove build artifacts"
\t@echo "  help  - Show this help message"
"""
        return makefile

    def _build_sw_emu(self):
        if not is_available():
            raise RuntimeError("g++ not found. Cannot build sw_emu.")

        project_dir = self.project or tempfile.mkdtemp(prefix="xls_sw_emu_")
        self._project_dir = project_dir
        os.makedirs(project_dir, exist_ok=True)

        harness = (
            self._generate_harness_array()
            if self._has_arrays
            else self._generate_harness_scalar()
        )
        harness_path = os.path.join(project_dir, "test_harness.cpp")
        binary_path = os.path.join(project_dir, "test_binary")

        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(harness)

        # Generate Makefile
        makefile = self._generate_makefile()
        makefile_path = os.path.join(project_dir, "Makefile")
        with open(makefile_path, "w", encoding="utf-8") as f:
            f.write(makefile)

        result = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-o", binary_path, harness_path],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"sw_emu compilation failed:\n{result.stderr}\n{result.stdout}"
            )

        self._binary_path = binary_path

    def __call__(self, *args):
        if self.mode != "sw_emu":
            raise RuntimeError("__call__ only supported in sw_emu mode")
        if not self._binary_path or not os.path.exists(self._binary_path):
            raise RuntimeError("sw_emu binary not built")

        if self._has_arrays:
            return self._call_array(*args)
        return self._call_scalar(*args)

    def _call_scalar(self, *args):
        str_args = [str(a) for a in args]
        result = subprocess.run(
            [self._binary_path] + str_args, capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            raise RuntimeError(f"sw_emu execution failed:\n{result.stderr}")

        for line in result.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                val = line[7:]
                if val == "OK":
                    return None
                try:
                    return int(val)
                except ValueError:
                    return val
        return None

    def _call_array(self, *args):
        for i, (arg, info) in enumerate(zip(args, self._arg_info)):
            if info["is_array"]:
                arr = np.asarray(arg, dtype=np.int32).flatten()
                with open(f"{self._project_dir}/input{i}.data", "wb") as f:
                    f.write(arr.tobytes())

        result = subprocess.run(
            [self._binary_path, self._project_dir],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"sw_emu execution failed:\n{result.stderr}")

        for line in result.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                val = line[7:]
                if val == "OK":
                    return None
                vals = [int(v) for v in val.split(",")]
                if self._output_info:
                    return np.array(vals, dtype=np.int32).reshape(
                        self._output_info["shape"]
                    )
                return np.array(vals, dtype=np.int32)
        return None

    def __repr__(self):
        return self.final_cpp

    # Print RAM rewrites textproto if available
    def print_textproto(self):
        if self.rewrites_textproto:
            print("=" * 60)
            print("RAM REWRITES TEXTPROTO:")
            print("=" * 60)
            print(self.rewrites_textproto)
        else:
            print("No RAM rewrites (not in memory mode)")
