# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module,no-member
# pylint: disable=too-many-branches,too-many-nested-blocks

from __future__ import annotations

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

from .xlscc.xls_wrapper import wrap_xlscc, validate_xls_ir


def is_available():
    try:
        result = subprocess.run(
            ["g++", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


class XLSCCModule:  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
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
        self._arg_info = []
        self._has_arrays = False
        self._output_info = None

        # Parse MLIR + run minimal lowering
        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mlir_text_or_module), ctx)

            pm = PassManager.parse(
                "builtin.module("
                "empty-tensor-to-alloc-tensor,"
                "func.func(convert-linalg-to-affine-loops)"
                ")"
            )
            pm.run(self.module.operation)

            self.func = find_func_in_module(self.module, top_func_name)
            if self.func is not None:
                self.func.attributes["top"] = UnitAttr.get()

        self.mlir_text = str(self.module)
        validate_xls_ir(self.mlir_text, project=project)

        # Emit the XLS[cc] HLS Code from MLIR
        buf = io.StringIO()
        allo_d.emit_xhls(self.module, buf, self.use_memory)
        buf.seek(0)
        self.core_code = buf.read()

        # Extract argument info for sw_emu
        self._extract_arg_info()

        func_name = top_func_name
        param_names: list[str] = []
        if self.func:
            for arg in self.func.arguments:
                arg_type_str = str(arg.type)
                if (
                    "memref" in arg_type_str or "tensor" in arg_type_str
                ) and "stream" not in arg_type_str.lower():
                    param_names.append(f"arg{len(param_names)}")

        self.final_cpp, self.rewrites_textproto = wrap_xlscc(
            mlir_module_text=self.mlir_text,
            core_code=self.core_code,
            function_names=(self.top_func_name, func_name),
            function_inputs=param_names,
            use_memory=self.use_memory,
        )

        if self.project is not None:
            os.makedirs(self.project, exist_ok=True)
            with open(f"{self.project}/test_block.cpp", "w", encoding="utf-8") as f:
                f.write(self.final_cpp)

            if self.use_memory and self.rewrites_textproto:
                with open(
                    f"{self.project}/rewrites.textproto", "w", encoding="utf-8"
                ) as f:
                    f.write(self.rewrites_textproto)

        # Build sw_emu binary if mode is sw_emu
        if self.mode == "sw_emu":
            self._build_sw_emu()

    def _extract_arg_info(self):
        if not self.func:
            return

        for i, arg in enumerate(self.func.arguments):
            arg_type_str = str(arg.type)
            info = {"index": i, "type_str": arg_type_str, "is_array": False}

            if "memref" in arg_type_str or "tensor" in arg_type_str:
                info["is_array"] = True
                self._has_arrays = True
                # Parse shape: memref<2x2xi32> -> shape=[2,2], dtype=i32
                match = re.search(r"<([^>]+)>", arg_type_str)
                if match:
                    content = match.group(1)
                    parts = content.split("x")
                    if len(parts) >= 2:
                        info["shape"] = [int(p) for p in parts[:-1]]
                        info["dtype"] = parts[-1]
                        info["size"] = 1
                        for dim in info["shape"]:
                            info["size"] *= dim
            else:
                info["dtype"] = arg_type_str

            self._arg_info.append(info)

        # Extract output info from function return type
        if self.func:
            result_types = list(self.func.type.results)
            if result_types:
                ret_type_str = str(result_types[0])
                if "memref" in ret_type_str or "tensor" in ret_type_str:
                    match = re.search(r"<([^>]+)>", ret_type_str)
                    if match:
                        content = match.group(1)
                        parts = content.split("x")
                        if len(parts) >= 2:
                            self._output_info = {
                                "shape": [int(p) for p in parts[:-1]],
                                "dtype": parts[-1],
                            }
                            self._output_info["size"] = 1
                            for dim in self._output_info["shape"]:
                                self._output_info["size"] *= dim

    def _get_sw_emu_header_path(self):
        return Path(__file__).parent / "xlscc" / "xls_sw_emu.h"

    def _preprocess_for_sw_emu(self, cpp_code):
        lines = cpp_code.split("\n")
        filtered = []
        skip_next = False
        for line in lines:
            if "#include" in line and (
                "xls" in line.lower()
                or "ac_int" in line
                or "ac_channel" in line
                or "hls_" in line
            ):
                continue
            if "template <int Width" in line:
                skip_next = True
                continue
            if skip_next and "using ac_int" in line:
                skip_next = False
                continue
            if "using ac_int" in line or "using ac_uint" in line:
                continue
            skip_next = False
            filtered.append(line)
        return "\n".join(filtered)

    def _generate_test_harness_scalar(self):
        header_path = self._get_sw_emu_header_path()
        processed_cpp = self._preprocess_for_sw_emu(self.final_cpp)

        harness = f"""// Auto-generated test harness for XLS sw_emu (scalar mode)
#include "{header_path}"
#include <iostream>
#include <cstdlib>
#include <cstring>

{processed_cpp}

int main(int argc, char** argv) {{
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <inputs...>" << std::endl;
        return 1;
    }}

"""
        args = []
        for i, info in enumerate(self._arg_info):
            if not info["is_array"]:
                dtype = info.get("dtype", "i32")
                if "i32" in dtype:
                    harness += f"    int arg{i} = std::atoi(argv[{i + 1}]);\n"
                elif "i64" in dtype:
                    harness += f"    long arg{i} = std::atol(argv[{i + 1}]);\n"
                else:
                    harness += f"    int arg{i} = std::atoi(argv[{i + 1}]);\n"
                args.append(f"arg{i}")

        func_name = self.top_func_name
        if args:
            harness += f"\n    auto result = {func_name}({', '.join(args)});\n"
            harness += '    std::cout << "RESULT:" << result << std::endl;\n'
        else:
            harness += f"\n    {func_name}();\n"
            harness += '    std::cout << "RESULT:OK" << std::endl;\n'

        harness += """
    return 0;
}
"""
        return harness

    def _generate_test_harness_array(self):
        header_path = self._get_sw_emu_header_path()
        processed_cpp = self._preprocess_for_sw_emu(self.final_cpp)
        output_size = self._output_info["size"] if self._output_info else 1

        harness = f"""// Auto-generated test harness for XLS sw_emu (array mode)
#include "{header_path}"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <vector>

{processed_cpp}

int main(int argc, char** argv) {{
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <project_dir>" << std::endl;
        return 1;
    }}
    std::string project_dir = argv[1];

    // Instantiate the TestBlock
    TestBlock block;

"""
        # Read input arrays from files and write to channels
        for i, info in enumerate(self._arg_info):
            if info["is_array"]:
                size = info.get("size", 1)
                harness += f"""
    // Read input{i} from file
    std::ifstream in{i}(project_dir + "/input{i}.data", std::ios::binary);
    if (!in{i}.is_open()) {{
        std::cerr << "Failed to open input{i}.data" << std::endl;
        return 1;
    }}
    std::vector<int32_t> input{i}({size});
    in{i}.read(reinterpret_cast<char*>(input{i}.data()), {size} * sizeof(int32_t));
    in{i}.close();

    // Write to input channel
    for (int j = 0; j < {size}; ++j) {{
        block.v{i}_in.write(input{i}[j]);
    }}
"""

        # Run the block (top function is the method name)
        harness += f"""
    // Run the computation
    block.{self.top_func_name}();

"""
        # Read output from channel
        harness += f"""
    // Read output from channel and print
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

    def _generate_test_harness(self):
        if self._has_arrays:
            return self._generate_test_harness_array()
        return self._generate_test_harness_scalar()

    def _build_sw_emu(self):
        if not is_available():
            raise RuntimeError("g++ not found. Cannot build sw_emu.")

        project_dir = self.project or tempfile.mkdtemp(prefix="xls_sw_emu_")
        self._project_dir = project_dir
        os.makedirs(project_dir, exist_ok=True)

        harness_code = self._generate_test_harness()
        harness_path = os.path.join(project_dir, "test_harness.cpp")
        binary_path = os.path.join(project_dir, "test_binary")

        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(harness_code)

        compile_cmd = [
            "g++",
            "-std=c++17",
            "-O2",
            "-o",
            binary_path,
            harness_path,
        ]

        result = subprocess.run(
            compile_cmd,
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
            [self._binary_path] + str_args,
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
                try:
                    return int(val)
                except ValueError:
                    try:
                        return float(val)
                    except ValueError:
                        return val
        return None

    def _call_array(self, *args):
        # Write input arrays to files
        for i, (arg, info) in enumerate(zip(args, self._arg_info)):
            if info["is_array"]:
                arr = np.asarray(arg, dtype=np.int32).flatten()
                with open(f"{self._project_dir}/input{i}.data", "wb") as f:
                    f.write(arr.tobytes())

        # Run binary with project dir as argument
        result = subprocess.run(
            [self._binary_path, self._project_dir],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"sw_emu execution failed:\n{result.stderr}")

        # Parse output array
        for line in result.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                val = line[7:]
                if val == "OK":
                    return None
                # Parse comma-separated values
                vals = [int(v) for v in val.split(",")]
                if self._output_info:
                    return np.array(vals, dtype=np.int32).reshape(
                        self._output_info["shape"]
                    )
                return np.array(vals, dtype=np.int32)
        return None

    def __repr__(self):
        return self.final_cpp

    def print_textproto(self):
        if self.rewrites_textproto:
            print("\n" + "=" * 60)
            print("RAM REWRITES TEXTPROTO (rewrites.textproto):")
            print("=" * 60)
            print(self.rewrites_textproto)
        else:
            print(
                "No RAM rewrites textproto (not in memory mode or no memories detected)"
            )
