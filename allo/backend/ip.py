# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import re
import importlib
import subprocess
import traceback
import time

# https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
allo2c_type = {
    "float32": "float",
    "float64": "double",
    "int1": "bool",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int",
    "int64": "int64_t",
    "int128": "ap_int<128>",
    # bitwidth larger than 64 is not supported by numpy+pybind11
    "uint1": "bool",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "unsigned int",
    "uint64": "uint64_t",
    "uint128": "ap_uint<128>",
}

c2allo_type = {v: k for k, v in allo2c_type.items()}
c2allo_type["int32_t"] = "int32"
c2allo_type["uint32_t"] = "uint32"


def parse_cpp_function(code, target_function):
    """
    Parse a C++ file to find a specific function and extract its parameter types and shapes.

    Args:
        code (str): The C++ code as a string
        target_function (str): The name of the function to find

    Returns:
        list: A list of tuples containing (type, shape) for each parameter
            - shape is a tuple of dimensions for arrays
            - shape is () for scalars
            - shape is None for pointers
    """
    # Function pattern that works for both declarations and definitions
    function_pattern = r"(\w+)\s+" + re.escape(target_function) + r"\s*\((.*?)\)\s*[{;]"

    # Find the function in the code
    function_match = re.search(function_pattern, code, re.DOTALL)
    if not function_match:
        return None

    # Extract return type and parameters
    # return_type = function_match.group(1)
    params_str = function_match.group(2)

    # Split parameters
    params = []
    current_param = ""
    bracket_count = 0

    for char in params_str:
        if char == "," and bracket_count == 0:
            params.append(current_param.strip())
            current_param = ""
        else:
            current_param += char
            if char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1

    if current_param.strip():
        params.append(current_param.strip())

    # Process each parameter to extract type and shape
    result = []
    for param in params:
        # Check if parameter is a pointer
        pointer_pattern = r"(\w+)\s+\*(\w+)"
        pointer_match = re.search(pointer_pattern, param)

        if pointer_match:
            param_type = pointer_match.group(1)
            result.append((param_type, None))
            continue

        # Check if parameter is an array (more careful matching)
        # We'll extract the full array part and process it separately
        array_pattern = r"(\w+)\s+(\w+)(\[\d+\](?:\[\d+\])*)"
        array_match = re.search(array_pattern, param)

        if array_match:
            param_type = array_match.group(1)
            array_dims_str = array_match.group(3)

            # Extract all dimensions using a separate regex
            dims = []
            dim_pattern = r"\[(\d+)\]"
            for dim_match in re.finditer(dim_pattern, array_dims_str):
                dims.append(int(dim_match.group(1)))

            result.append((param_type, tuple(dims)))
            continue

        # If we get here, it's a scalar
        scalar_pattern = r"(\w+)\s+(\w+)"
        scalar_match = re.search(scalar_pattern, param)

        if scalar_match:
            param_type = scalar_match.group(1)
            result.append((param_type, ()))

    return result


class IPModule:
    def __init__(self, top, impl, include_paths=None, link_hls=True):
        self.top = top
        self.abs_path = os.path.dirname(traceback.extract_stack()[-2].filename)
        self.temp_path = os.path.join(self.abs_path, "_tmp")
        os.makedirs(self.temp_path, exist_ok=True)
        self.impl = os.path.join(self.abs_path, impl)
        if include_paths is None:
            include_paths = []
        self.include_paths = include_paths + [self.abs_path]
        if link_hls:
            if os.system("which vitis_hls >> /dev/null") == 0:
                self.include_paths.append(
                    "/".join(os.popen("which vitis_hls").read().split("/")[:-2])
                    + "/include"
                )
            elif os.system("which vivado_hls >> /dev/null") == 0:
                self.include_paths.append(
                    "/".join(os.popen("which vivado_hls").read().split("/")[:-2])
                    + "/include"
                )
            else:
                raise RuntimeError(
                    "Please install Vivado/Vitis HLS and add it to your PATH"
                )

        # Parse signature
        with open(self.impl, "r", encoding="utf-8") as f:
            code = f.read()
            self.args = parse_cpp_function(code, self.top)
        assert self.args is not None, f"Failed to parse {self.impl}"
        self.lib_name = f"py{self.top}_{hash(time.time_ns())}"
        self.c_wrapper_file = os.path.join(self.temp_path, f"{self.lib_name}.cpp")

    def generate_pybind11_wrapper(self):
        out_str = "// Auto-generated by Allo\n\n"
        # Add headers
        out_str += "#include <iostream>\n"
        out_str += "#include <pybind11/numpy.h>\n"
        out_str += "#include <pybind11/pybind11.h>\n"
        out_str += f'#include "{self.impl}"\n'
        # Add source headers
        out_str += "\nnamespace py = pybind11;\n\n"
        # Generate function interface
        out_str += f"void {self.lib_name}(\n"
        for i, (arg_type, arg_shape) in enumerate(self.args):
            if arg_shape is None or len(arg_shape) > 0:
                # pointer or array
                out_str += f"  py::array_t<{arg_type}> &arg{i}"
            else:
                # scalar
                out_str += f"  {arg_type} arg{i}"
            out_str += ",\n" if i < len(self.args) - 1 else ") {\n"
        # Generate function body
        out_str += "\n"
        in_ptrs = []
        for i, (arg_type, arg_shape) in enumerate(self.args):
            if arg_shape is None or len(arg_shape) == 1:
                # pointer or rank-1 array
                out_str += f"  py::buffer_info buf{i} = arg{i}.request();\n"
                out_str += f"  {arg_type} *p_arg{i} = ({arg_type} *)buf{i}.ptr;\n"
                in_ptrs.append(f"p_arg{i}")
            elif len(arg_shape) == 0:
                out_str += f"  {arg_type} p_arg{i} = arg{i};\n"
                in_ptrs.append(f"p_arg{i}")
            else:
                out_str += f"  py::buffer_info buf{i} = arg{i}.request();\n"
                out_str += f"  {arg_type} *p_arg{i} = ({arg_type} *)buf{i}.ptr;\n"
                tail_shape = "[" + "][".join([str(s) for s in arg_shape[1:]]) + "]"
                out_str += f"  {arg_type} (*p_arg{i}_nd){tail_shape} = "
                out_str += f"reinterpret_cast<{arg_type} (*){tail_shape}>(p_arg{i});\n"
                in_ptrs.append(f"p_arg{i}_nd")
        # function call
        out_str += "\n"
        out_str += f"  {self.top}({', '.join(in_ptrs)});\n"
        # Return
        out_str += "}\n\n"
        # Add pybind11 wrapper
        out_str += f"\nPYBIND11_MODULE({self.lib_name}, m) {{\n"
        out_str += f'  m.def("{self.top}", &{self.lib_name}, "{self.top} wrapper");\n'
        out_str += "}\n"
        with open(self.c_wrapper_file, "w", encoding="utf-8") as f:
            f.write(out_str)
        return self.c_wrapper_file

    def compile_pybind11(self):
        self.generate_pybind11_wrapper()
        cmd = "g++ -shared -std=c++14 -fPIC"
        cmd += " `python3 -m pybind11 --includes` "
        cmd += " ".join(
            ["-I" + (path if path != "" else ".") for path in self.include_paths]
        )
        srcs = [self.c_wrapper_file]
        cmd += " " + " ".join(srcs)
        cmd += (
            f" -o {self.temp_path}/{self.lib_name}`python3-config --extension-suffix`"
        )
        print(cmd)
        try:
            subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to compile pybind wrapper for {self.lib_name}!"
            ) from exc

    def generate_mlir_c_wrapper(self):
        out_str = "// Auto-generated by Allo\n\n"
        # Add headers
        out_str += "#include <iostream>\n"
        out_str += '#include "mlir/ExecutionEngine/CRunnerUtils.h"\n'
        out_str += f'#include "{self.impl}"\n'
        out_str += "\n"
        # Generate function interface
        unranked_memrefs = []
        for i, (arg_type, arg_shape) in enumerate(self.args):
            if len(arg_shape) > 0:
                unranked_memrefs.append(f"int64_t rank_{i}, void *ptr_{i}")
            else:
                unranked_memrefs.append(f"{arg_type} in{i}")
        unranked_memrefs_str = ", ".join(unranked_memrefs)
        out_str += f'extern "C" void {self.lib_name}({unranked_memrefs_str}) {{\n'
        in_ptrs = []
        for i, (arg_type, arg_shape) in enumerate(self.args):
            if len(arg_shape) == 0:  # scalar
                in_ptrs.append(f"in{i}")
                continue
            out_str += (
                f"  UnrankedMemRefType<{arg_type}> in{i} = {{rank_{i}, ptr_{i}}};\n"
            )
            out_str += f"  DynamicMemRefType<{arg_type}> ranked_in{i}(in{i});\n"
            out_str += f"  {arg_type} *in{i}_ptr = ({arg_type} *)ranked_in{i}.data;\n"
            if len(arg_shape) == 1:
                in_ptrs.append(f"in{i}_ptr")
            else:
                tail_shape = "[" + "][".join([str(s) for s in arg_shape[1:]]) + "]"
                out_str += f"  {arg_type} (*in{i}_nd){tail_shape} = "
                out_str += f"reinterpret_cast<{arg_type} (*){tail_shape}>(in{i}_ptr);\n"
                in_ptrs.append(f"in{i}_nd")
        # Call library function
        out_str += f"  {self.top}({', '.join(in_ptrs)});\n"
        out_str += "}\n"
        with open(self.c_wrapper_file, "w", encoding="utf-8") as f:
            f.write(out_str)
        return self.c_wrapper_file

    def compile_shared_lib(self):
        # Used in direct function call in an Allo kernel
        self.generate_mlir_c_wrapper()
        if os.system("which llvm-config >> /dev/null") != 0:
            raise RuntimeError("Please install LLVM and add it to your PATH")
        cmd = "g++ -c -std=c++14 -fpic "
        # suppose the build directory is under llvm-project
        self.include_paths.append(
            "/".join(os.popen("which llvm-config").read().split("/")[:-3])
            + "/mlir/include"
        )
        cmd += " ".join(
            ["-I" + (path if path != "" else ".") for path in self.include_paths]
        )
        srcs = [self.c_wrapper_file]
        obj_files = []
        for src in srcs:
            subcmd = cmd
            subcmd += " " + src
            obj = f"{self.temp_path}/{src.split('/')[-1]}.o"
            subcmd += " -o " + obj
            print(subcmd)
            try:
                subprocess.check_output(subcmd, shell=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Failed to compile {src.split('/')[-1]}.o!"
                ) from exc
            obj_files.append(obj)
        cmd = f"g++ -shared -o {self.temp_path}/lib{self.top}.so " + " ".join(obj_files)
        print(cmd)
        try:
            subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to compile {self.top}.so!") from exc
        return f"{self.temp_path}/lib{self.top}.so"

    def __call__(self, *args):
        self.compile_pybind11()
        sys.path.append(self.temp_path)
        self.lib = importlib.import_module(f"{self.lib_name}")
        return getattr(self.lib, f"{self.top}")(*args)
