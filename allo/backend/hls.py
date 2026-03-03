# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=consider-using-with, no-name-in-module, too-many-branches

import os
import re
import io
import subprocess
import time
import numpy as np
from .._mlir.dialects import allo as allo_d
from .._mlir.ir import (
    Context,
    Location,
    Module,
    UnitAttr,
)
from .._mlir.passmanager import PassManager

from .config import DEFAULT_CONFIG, PART_NUMBER
from .vitis import (
    codegen_host,
    postprocess_hls_code,
    generate_description_file,
    write_tensor_to_file,
    read_tensor_from_file,
    generate_hbm_config,
    extract_hls_arg_names,
)
from .pynq import (
    postprocess_hls_code_pynq,
    codegen_pynq_host,
)
from .tapa import (
    codegen_tapa_host,
)
from .catapult import (
    codegen_tcl as codegen_tcl_catapult,
    codegen_host as codegen_host_catapult,
)
from .ip import IPModule
from .report import parse_xml
from ..passes import (
    _mlir_lower_pipeline,
    decompose_library_function,
    generate_input_output_buffers,
)
from ..harness.makefile_gen.makegen import generate_makefile
from ..ir.transform import find_func_in_module
from ..utils import (
    get_func_inputs_outputs,
    c2allo_type,
    get_bitwidth_from_type,
    np_supported_types,
)


def is_available(backend="vivado_hls"):
    if backend == "vivado_hls":
        return os.system("which vivado_hls >> /dev/null") == 0
    if backend == "tapa":
        return os.system("which tapa >> /dev/null") == 0
    return os.system("which vitis_hls >> /dev/null") == 0


def run_process(cmd, pattern=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if err:
        raise RuntimeError("Error raised: ", err.decode())
    if pattern:
        return re.findall(pattern, out.decode("utf-8"))
    return out.decode("utf-8")


def codegen_tcl(top, configs):
    out_str = """# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#=============================================================================
# run.tcl 
#=============================================================================
# Project name
set hls_prj out.prj

# Open/reset the project
open_project ${hls_prj} -reset

open_solution -reset solution1 -flow_target vivado

"""
    out_str += f'# Top function of the design is "{top}"\n'
    out_str += f"set_top {top}\n"
    out_str += """
# Add design and testbench files
add_files kernel.cpp
add_files -tb host.cpp -cflags "-std=gnu++0x"
open_solution "solution1"
"""
    device = configs["device"]
    frequency = configs["frequency"]
    mode = configs["mode"]
    if device not in PART_NUMBER:
        raise RuntimeError(
            f"Device {device} not supported. Available devices: {list(PART_NUMBER.keys())}"
        )
    out_str += f"\n# Target device is {device}\n"
    out_str += f"set_part {{{PART_NUMBER[device]}}}\n\n"
    out_str += "# Target frequency\n"
    out_str += f"create_clock -period {1000 / frequency:.2f}\n\n"
    out_str += "# Run HLS\n"
    if "csim" in mode or "sw_emu" in mode:
        out_str += "csim_design -O\n"
    if "csyn" in mode or "debug" in mode:
        out_str += "csynth_design\n"
    if "cosim" in mode or "hw_emu" in mode:
        out_str += "cosim_design\n"
    if "impl" in mode or "hw" in mode:
        if device in {"ultra96v2", "pynqz2", "zedboard"}:
            # Embedded boards: export IP only, bitstream happens in Python/Vivado later
            out_str += "export_design -rtl verilog -format ip_catalog\n"
        else:
            # Other platforms: run full impl in HLS
            out_str += "export_design -flow impl\n"
    out_str += "\nexit\n"
    return out_str


def copy_ext_libs(ext_libs, project):
    for ext_lib in ext_libs:
        impl_path = ext_lib.impl
        cpp_file = impl_path.split("/")[-1]
        assert cpp_file != "kernel.cpp", "kernel.cpp is reserved for the top function"
        os.system(f"cp {impl_path} {project}/{cpp_file}")


def separate_header(hls_code, top=None, extern_c=True):
    func_decl = False
    sig_str = "#ifndef KERNEL_H\n"
    sig_str += "#define KERNEL_H\n\n"
    args = []
    if extern_c:
        sig_str += 'extern "C" {\n'
    for line in hls_code.split("\n"):
        if line.startswith(f"void {top}"):
            func_decl = True
            sig_str += line + "\n"
        elif func_decl and line.startswith(") {"):
            func_decl = False
            sig_str += ");\n"
            break
        elif func_decl:
            arg_type = line.strip()
            _, var = arg_type.rsplit(" ", 1)
            comma = "," if var[-1] == "," else ""
            ele_type = arg_type.split("[")[0].split(" ")[0].strip()
            allo_type = None
            if ele_type in c2allo_type:
                allo_type = c2allo_type[ele_type]
            else:
                pattern = r"^ap_(u?)int<(\d+)>$"
                match = re.match(pattern, ele_type)
                if not match:
                    raise ValueError(f"Fail to resolve ctype {ele_type}")
                unsigned_flag, width = match.groups()
                allo_type = f"{'u' if unsigned_flag else ''}int{int(width)}"
            shape = tuple(s.split("]")[0] for s in arg_type.split("[")[1:])
            args.append((allo_type, shape))
            if "[" in var:  # array
                var = var.split("[")[0]
                sig_str += "  " + ele_type + " *" + var + f"{comma}\n"
            else:  # scalar
                var = var.split(",")[0]
                sig_str += "  " + ele_type + " " + var + f"{comma}\n"
    if extern_c:
        sig_str += '} // extern "C"\n'
    sig_str += "\n#endif // KERNEL_H\n"
    return sig_str, args


class HLSModule:
    def __init__(
        self,
        mod,
        top_func_name,
        platform="vivado_hls",
        mode=None,
        project=None,
        ext_libs=None,
        configs=None,
        func_args=None,
        wrap_io=True,
    ):
        self.top_func_name = top_func_name
        self.mode = mode
        self.project = project
        self.platform = platform
        self.ext_libs = [] if ext_libs is None else ext_libs
        self.num_output_args = 0  # Will be set from configs if provided
        if configs is not None:
            new_configs = DEFAULT_CONFIG.copy()
            new_configs.update(configs)
            configs = new_configs
            self.num_output_args = configs.get("num_output_args", 0)
        else:
            configs = DEFAULT_CONFIG.copy()
        if self.mode is not None:
            configs["mode"] = self.mode
        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            func = find_func_in_module(self.module, top_func_name)
            func.attributes["top"] = UnitAttr.get()

            if platform in {"vitis_hls", "pynq"}:
                assert func_args is not None, "Need to specify func_args"
                if wrap_io:
                    generate_input_output_buffers(
                        self.module,
                        top_func_name,
                        flatten=True,
                        mappings=configs.get("mappings", None),
                    )

            self.module = decompose_library_function(self.module)
            _mlir_lower_pipeline(self.module, lower_linalg=True)
            # Run through lowering passes
            pm = PassManager.parse(
                "builtin.module("
                # used for lowering tensor.empty
                "empty-tensor-to-alloc-tensor,"
                # translate tensor dialect (virtual) to memref dialect (physical)
                # "one-shot-bufferize{bufferize-function-boundaries},"
                # common lowering passes
                "func.func(convert-linalg-to-affine-loops)"
                # DO NOT LOWER AFFINE DIALECT
                ")"
            )
            pm.run(self.module.operation)
        buf = io.StringIO()
        success = True
        match platform:
            case "tapa":
                success = allo_d.emit_thls(self.module, buf)
            case "intel_hls":
                success = allo_d.emit_ihls(self.module, buf)
            case "catapult":
                success = allo_d.emit_catapult(self.module, buf)
            case _:
                # wrap_io=True has already linearized array indexing in
                # generate_input_output_buffers, so we don't need to do it again
                flatten = False if platform == "vivado_hls" else (not wrap_io)
                success = allo_d.emit_vhls(self.module, buf, flatten=flatten)

        if not success:
            raise RuntimeError(
                "Failed to emit HLS code. Check error messages above for details. "
                "Common issues: nested functions with multi-dimensional arrays when wrap_io=False."
            )

        buf.seek(0)
        self.hls_code = buf.read()
        if project is not None:
            assert mode is not None, "mode must be specified when project is specified"
            os.makedirs(project, exist_ok=True)
            path = os.path.dirname(__file__)
            path = os.path.join(path, "../harness/")
            if platform in {"vivado_hls", "vitis_hls", "tapa", "pynq", "catapult"}:
                os.system("cp " + path + f"{platform.split('_')[0]}/* " + project)
                with open(f"{project}/run.tcl", "w", encoding="utf-8") as outfile:
                    if platform == "catapult":
                        outfile.write(codegen_tcl_catapult(top_func_name, configs))
                    else:
                        outfile.write(codegen_tcl(top_func_name, configs))
            copy_ext_libs(ext_libs, project)
            if self.platform == "vitis_hls":
                assert self.mode in {
                    "csim",
                    "csyn",
                    "sw_emu",
                    "hw_emu",
                    "hw",
                }, "Invalid mode"
                assert (
                    self.top_func_name != "kernel"
                ), "kernel is a reserved keyword for vitis_hls"
                path = os.path.dirname(__file__)
                path = os.path.join(path, "../harness/")
                dst_path = os.path.join(project, "description.json")
                generate_description_file(
                    self.top_func_name,
                    path + "makefile_gen/description.json",
                    dst_path,
                    frequency=configs["frequency"],
                )
                hbm_mapping = configs.get("hbm_mapping", None)
                generate_makefile(dst_path, project, self.platform, hbm_mapping)
                header, self.args = separate_header(self.hls_code, self.top_func_name)
                with open(f"{project}/kernel.h", "w", encoding="utf-8") as outfile:
                    outfile.write(header)
                self.hls_code = postprocess_hls_code(self.hls_code, self.top_func_name)

                # Generate HBM/DDR configuration file if hbm_mapping is provided
                # This must be done AFTER postprocess_hls_code to get correct arg names
                if hbm_mapping is not None:
                    # Extract HLS argument names from the postprocessed code
                    hls_arg_names = extract_hls_arg_names(
                        self.hls_code, self.top_func_name
                    )
                    # Build mapping from user arg names to HLS arg names
                    user_arg_names = []
                    if func_args is not None and self.top_func_name in func_args:
                        for arg in func_args[self.top_func_name]:
                            if hasattr(arg, "name"):
                                user_arg_names.append(arg.name)
                            else:
                                user_arg_names.append(str(arg))
                    # Add return value name - it becomes the last argument
                    # Use the last HLS arg name count to determine if there's a return
                    if len(hls_arg_names) > len(user_arg_names):
                        # There's a return value, add placeholder names
                        for i in range(len(hls_arg_names) - len(user_arg_names)):
                            user_arg_names.append(f"output_{i}")

                    arg_name_mapping = None
                    if len(user_arg_names) == len(hls_arg_names):
                        arg_name_mapping = dict(zip(user_arg_names, hls_arg_names))

                    cfg_content = generate_hbm_config(
                        self.top_func_name, hbm_mapping, arg_name_mapping
                    )
                    cfg_path = os.path.join(project, f"{self.top_func_name}.cfg")
                    with open(cfg_path, "w", encoding="utf-8") as cfg_file:
                        cfg_file.write(cfg_content)
                for lib in self.ext_libs:
                    cpp_file = lib.impl.split("/")[-1]
                    with open(f"{project}/{cpp_file}", "r", encoding="utf-8") as infile:
                        new_code = postprocess_hls_code(
                            infile.read(), lib.top, pragma=False
                        )
                    with open(
                        f"{project}/{cpp_file}", "w", encoding="utf-8"
                    ) as outfile:
                        outfile.write(new_code)
                self.host_code = codegen_host(
                    self.top_func_name,
                    self.module,
                    num_output_args=self.num_output_args,
                )
            elif self.platform == "catapult":
                assert self.mode in {
                    "csim",
                    "csyn",
                }, "Invalid mode for catapult"

                if self.mode == "csim":
                    self.host_code = codegen_host_catapult(
                        self.top_func_name,
                        self.module,
                    )
                else:
                    self.host_code = ""

                # For Catapult, we don't have separate kernel.h generation logic yet
                # similar to separate_header. The kernel.cpp contains everything needed
                # or headers are handled differently.
                # If we want to support csim, kernel.cpp usually needs a header
                # referenced by host.cpp.
                # allo/backend/catapult.py's codegen_host includes "kernel.h".
                # So we SHOULD generate kernel.h.
                # Re-using separate_header which is generic enough for C-style headers.
                #
                # However, separate_header currently only understands builtin and
                # ap_(u)int<...> types. When Catapult emits ac_int<...> (e.g., for
                # non-standard integer widths), separate_header can raise ValueError.
                # Fall back to including kernel.cpp directly if that happens.
                try:
                    header, self.args = separate_header(
                        self.hls_code, self.top_func_name
                    )
                except ValueError:
                    header = '#pragma once\n#include "kernel.cpp"\n'
                    self.args = []
                with open(f"{project}/kernel.h", "w", encoding="utf-8") as outfile:
                    outfile.write(header)
            elif self.platform == "tapa":
                assert self.mode in {
                    "csim",
                    "fast_hw_emu",
                    "hw_emu",
                    "hw",
                }, "Invalid mode"
                assert (
                    self.top_func_name != "kernel"
                ), "kernel is a reserved keyword for tapa"
                path = os.path.dirname(__file__)
                path = os.path.join(path, "../harness/")
                dst_path = os.path.join(project, "description.json")
                generate_description_file(
                    self.top_func_name,
                    path + "makefile_gen/description.json",
                    dst_path,
                    frequency=configs["frequency"],
                )
                self.args = []
                hbm_mapping = configs.get("hbm_mapping", None)
                generate_makefile(dst_path, project, self.platform, hbm_mapping)
                # Generate HBM/DDR configuration file if hbm_mapping is provided
                if hbm_mapping is not None:
                    # Extract HLS argument names from the code
                    hls_arg_names = extract_hls_arg_names(
                        self.hls_code, self.top_func_name
                    )
                    # Build mapping from user arg names to HLS arg names
                    user_arg_names = []
                    if func_args is not None and self.top_func_name in func_args:
                        for arg in func_args[self.top_func_name]:
                            if hasattr(arg, "name"):
                                user_arg_names.append(arg.name)
                            else:
                                user_arg_names.append(str(arg))
                    # Add placeholder for return values if needed
                    if len(hls_arg_names) > len(user_arg_names):
                        for i in range(len(hls_arg_names) - len(user_arg_names)):
                            user_arg_names.append(f"output_{i}")

                    arg_name_mapping = None
                    if len(user_arg_names) == len(hls_arg_names):
                        arg_name_mapping = dict(zip(user_arg_names, hls_arg_names))

                    cfg_content = generate_hbm_config(
                        self.top_func_name, hbm_mapping, arg_name_mapping
                    )
                    cfg_path = os.path.join(project, f"{self.top_func_name}.cfg")
                    with open(cfg_path, "w", encoding="utf-8") as cfg_file:
                        cfg_file.write(cfg_content)
                # [NOTE] (Shihan): I guess tapa backend do not use this one. I modified codegen_host for vitis, similar logic should be updated for tapa if self.host_code is useful here
                self.host_code = codegen_host(
                    self.top_func_name,
                    self.module,
                )
                self.tapa_host = codegen_tapa_host(
                    self.top_func_name,
                    self.module,
                    self.hls_code,
                )
                with open(f"{project}/tapa_host.cpp", "w", encoding="utf-8") as outfile:
                    outfile.write(self.tapa_host)
            elif self.platform == "pynq":
                assert self.mode in {"csim", "csyn", "impl"}, "Invalid mode for pynq"
                kernel_h = os.path.join(project, "kernel.h")

                # Generate kernel.h
                header, self.args = separate_header(self.hls_code, self.top_func_name)
                with open(kernel_h, "w", encoding="utf-8") as outfile:
                    outfile.write(header)

                # Apply PYNQ-specific HLS code tweaks and write kernel.cpp
                self.hls_code = postprocess_hls_code_pynq(
                    self.hls_code, self.top_func_name
                )
            else:
                self.host_code = ""
            with open(f"{project}/kernel.cpp", "w", encoding="utf-8") as outfile:
                outfile.write(self.hls_code)
            if hasattr(self, "host_code") and self.host_code:
                with open(f"{project}/host.cpp", "w", encoding="utf-8") as outfile:
                    outfile.write(self.host_code)
            if len(ext_libs) > 0:
                for lib in ext_libs:
                    # Update kernel.cpp
                    new_kernel = ""
                    with open(
                        os.path.join(project, "kernel.cpp"), "r", encoding="utf-8"
                    ) as kernel:
                        for line in kernel:
                            new_kernel += line
                            if "#include <stdint.h>" in line:
                                new_kernel += f'#include "{lib.impl.split("/")[-1]}"\n'
                    with open(
                        os.path.join(project, "kernel.cpp"), "w", encoding="utf-8"
                    ) as kernel:
                        kernel.write(new_kernel)
                    # Update tcl file
                    new_tcl = ""
                    with open(
                        os.path.join(project, "run.tcl"), "r", encoding="utf-8"
                    ) as tcl_file:
                        for line in tcl_file:
                            new_tcl += line
                            if "# Add design and testbench files" in line:
                                cpp_file = lib.impl.split("/")[-1]
                                new_tcl += f"add_files {cpp_file}\n"
                    with open(
                        os.path.join(project, "run.tcl"), "w", encoding="utf-8"
                    ) as tcl_file:
                        tcl_file.write(new_tcl)

    def __repr__(self):
        if self.mode is None:
            return self.hls_code
        return f"HLSModule({self.top_func_name}, {self.mode}, {self.project})"

    def __call__(self, *args, shell=True):
        if self.platform == "vivado_hls":
            assert is_available("vivado_hls"), "vivado_hls is not available"
            ver = run_process("g++ --version", r"\d+\.\d+\.\d+")[0].split(".")
            assert (
                int(ver[0]) * 10 + int(ver[1]) >= 48
            ), f"g++ version too old {ver[0]}.{ver[1]}.{ver[2]}"

            cmd = f"cd {self.project}; make "
            if self.mode == "csim":
                cmd += "csim"
                out = run_process(cmd + " 2>&1")
                runtime = [k for k in out.split("\n") if "seconds" in k][0]
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Simulation runtime {runtime}"
                )

            elif "csyn" in self.mode or self.mode == "custom" or self.mode == "debug":
                cmd += self.platform
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to synthesize the design")
                if self.mode != "custom":
                    out = parse_xml(
                        self.project,
                        "Vivado HLS",
                        top=self.top_func_name,
                        print_flag=True,
                    )

            else:
                raise RuntimeError(f"{self.platform} does not support {self.mode} mode")
        elif self.platform == "vitis_hls":
            assert is_available("vitis_hls"), "vitis_hls is not available"
            if self.mode == "csim":
                mod = IPModule(
                    top=self.top_func_name,
                    impl=f"{self.project}/kernel.cpp",
                    include_paths=[self.project],
                    link_hls=True,
                )
                mod(*args)
                return
            if self.mode == "csyn":
                cmd = f"cd {self.project}; vitis_hls -f run.tcl"
                assert len(args) == 0, "csyn mode does not need to pass in arguments"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to synthesize the design")
                return
            # Use Makefile (sw_emu, hw_emu, hw)
            assert "XDEVICE" in os.environ, "Please set XDEVICE in your environment"
            # prepare data
            func = find_func_in_module(self.module, self.top_func_name)
            inputs, outputs = get_func_inputs_outputs(func)
            assert len(args) == len(inputs) + len(
                outputs
            ), f"Number of arguments mismatch, got {len(args)}, expected {len(inputs) + len(outputs)}"
            for i, ((in_dtype, in_shape), arg) in enumerate(zip(inputs, args)):
                assert (len(in_shape) == 0 and np.isscalar(arg)) or np.prod(
                    arg.shape
                ) == np.prod(
                    in_shape
                ), f"invalid arguemnt {i}, {np.asarray(arg).shape}-{in_shape}"
                ele_bitwidth = get_bitwidth_from_type(in_dtype)
                assert (
                    ele_bitwidth == 1 or ele_bitwidth % 8 == 0
                ), "can only handle bytes"
                # store as byte stream
                with open(f"{self.project}/input{i}.data", "wb") as f:
                    if np.isscalar(arg):
                        arg = np.array(arg, dtype=np_supported_types[in_dtype])
                    f.write(arg.tobytes())
            # check if the build folder exists
            bitstream_folder = f"{self.project}/build_dir.{self.mode}.{os.environ['XDEVICE'].rsplit('/')[-1].split('.')[0]}"
            if not os.path.exists(
                os.path.join(bitstream_folder, f"{self.top_func_name}.xclbin")
            ):
                cmd = (
                    f"cd {self.project}; make run TARGET={self.mode} PLATFORM=$XDEVICE"
                )
                print(cmd)
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to build the project")
            else:
                print("Build folder exists, skip building")
                # run the executable
                prefix = f"cd {self.project};"
                if not os.path.exists(f"{self.project}/{self.top_func_name}"):
                    prefix += " make host PLATFORM=$XDEVICE;"
                prefix += (
                    f" XCL_EMULATION_MODE={self.mode}" if self.mode != "hw" else ""
                )
                cmd = f"{prefix} ./{self.top_func_name} ../{bitstream_folder}/{self.top_func_name}.xclbin"
                print(cmd)
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to run the executable")
            # Read output tensors from files
            # Determine how many output files to read
            func = find_func_in_module(self.module, self.top_func_name)
            _, outputs = get_func_inputs_outputs(func)
            if len(outputs) > 0:
                # Original behavior: single output.data file
                if np.isscalar(args[-1]):
                    raise RuntimeError("The output must be a tensor")
                arr = np.fromfile(f"{self.project}/output.data", dtype=args[-1].dtype)
                args[-1][:] = arr.reshape(args[-1].shape)
            else:
                # Multiple output files: output0.data, output1.data, etc.
                num_out = self.num_output_args if self.num_output_args > 0 else 1
                for idx in range(num_out):
                    out_arg_idx = len(inputs) - num_out + idx
                    if out_arg_idx < 0 or out_arg_idx >= len(args):
                        continue
                    out_arg = args[out_arg_idx]
                    if np.isscalar(out_arg):
                        continue
                    arr = np.fromfile(
                        f"{self.project}/output{idx}.data", dtype=out_arg.dtype
                    )
                    out_arg[:] = arr.reshape(out_arg.shape)
            return
        elif self.platform == "pynq":
            # Do not assert PYNQ availability here; the presence of a physical
            # PYNQ device should be checked by callers that need it.
            if self.mode == "csim":
                cwd = os.getcwd()
                mod = IPModule(
                    top=self.top_func_name,
                    impl=f"{cwd}/{self.project}/kernel.cpp",
                    link_hls=True,
                )
                mod(*args)
                return
            if self.mode in {"csyn", "impl"}:
                # HLS synthesis
                cmd = f"cd {self.project}; vitis_hls -f run.tcl"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project ..."
                )
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to synthesize the design")

                if self.mode == "impl":
                    # Produce host (deploy.py)
                    host_code = codegen_pynq_host(
                        self.top_func_name,
                        self.module,
                        self.project,
                    )
                    with open(
                        f"{self.project}/deploy.py", "w", encoding="utf-8"
                    ) as outfile:
                        outfile.write(host_code)

                    # Vivado block design
                    bd_script = "block_design.tcl"
                    bd_script = os.path.basename(bd_script)
                    cmd = f"cd {self.project}; vivado -mode batch -source {bd_script}"
                    print(
                        f"[{time.strftime('%H:%M:%S', time.gmtime())}] Running Vivado Block Design ..."
                    )
                    process = subprocess.Popen(cmd, shell=True)
                    process.wait()
                    if process.returncode != 0:
                        raise RuntimeError(
                            "Failed to create block design / generate bitstream"
                        )

                    # Package .bit / .hwh / deploy.py into deploy/ folder
                    deploy_dir = os.path.join(self.project, "deploy")
                    cmd = (
                        f"mkdir -p {deploy_dir}; "
                        f"cp {self.project}/build_vivado/project_1.runs/impl_1/project_1_bd_wrapper.bit {deploy_dir}/{self.top_func_name}.bit; "
                        f"cp {self.project}/build_vivado/project_1.gen/sources_1/bd/project_1_bd/hw_handoff/project_1_bd.hwh {deploy_dir}/{self.top_func_name}.hwh; "
                        f"cp {self.project}/deploy.py {deploy_dir}/deploy.py"
                    )
                    print(
                        f"[{time.strftime('%H:%M:%S', time.gmtime())}] Collecting files for deployment ..."
                    )
                    print(f"Files for deployment located in {deploy_dir}")
                    process = subprocess.Popen(cmd, shell=True)
                    process.wait()
                    if process.returncode != 0:
                        raise RuntimeError("Failed to collect files")
                return
        elif self.platform == "tapa":
            assert is_available("tapa"), "tapa is not available"
            # Use Makefile (sw_emu, hw_emu, hw)
            assert "XDEVICE" in os.environ, "Please set XDEVICE in your environment"
            # prepare data
            func = find_func_in_module(self.module, self.top_func_name)
            inputs, _ = get_func_inputs_outputs(func)
            for i, ((_, in_shape), arg) in enumerate(zip(inputs, args)):
                write_tensor_to_file(
                    arg,
                    in_shape,
                    f"{self.project}/input{i}.data",
                )
            # check if the build folder exists
            if self.mode in {"csim", "fast_hw_emu"}:
                cmd = f"cd {self.project}; make {self.mode}"
                print(cmd)
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to run tapa executable")
                return
            bitstream_folder = f"{self.project}/build_dir.{self.mode}.{os.environ['XDEVICE'].rsplit('/')[-1].split('.')[0]}"
            if not os.path.exists(
                os.path.join(bitstream_folder, f"{self.top_func_name}.xclbin")
            ):
                cmd = (
                    f"cd {self.project}; make run TARGET={self.mode} PLATFORM=$XDEVICE"
                )
                print(cmd)
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to build the project")
            else:
                print("Build folder exists, skip building")
                # run the executable
                prefix = f"cd {self.project};"
                if not os.path.exists(f"{self.project}/{self.top_func_name}"):
                    prefix += " make host PLATFORM=$XDEVICE;"
                prefix += (
                    f" XCL_EMULATION_MODE={self.mode}" if self.mode != "hw" else ""
                )
                cmd = f"{prefix} ./{self.top_func_name} ../{bitstream_folder}/{self.top_func_name}.xclbin"
                print(cmd)
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to run the executable")
            # suppose the last argument is the output tensor
            result = read_tensor_from_file(
                inputs[-1][0], args[-1].shape, f"{self.project}/output.data"
            )
            args[-1][:] = result
            return
        if self.platform == "catapult":
            if self.mode == "csim":
                # Check for input arguments
                func = find_func_in_module(self.module, self.top_func_name)
                inputs, outputs = get_func_inputs_outputs(func)
                assert len(args) == len(inputs) + len(
                    outputs
                ), f"Number of arguments mismatch, got {len(args)}, expected {len(inputs) + len(outputs)}"

                # Generate kernel.h
                # self.args might be updated by separate_header if needed, but for csim we use passed args
                header, _ = separate_header(
                    self.hls_code, self.top_func_name, extern_c=False
                )
                with open(
                    os.path.join(self.project, "kernel.h"), "w", encoding="utf-8"
                ) as outfile:
                    outfile.write(header)

                # Write input data
                for i, ((in_dtype, in_shape), arg) in enumerate(
                    zip(inputs, args[: len(inputs)])
                ):
                    write_tensor_to_file(arg, in_shape, f"{self.project}/input{i}.data")

                # Compilation with g++
                # Assuming 'g++' is in PATH.
                # Include path for ac_types
                mgc_home = os.environ.get("MGC_HOME")
                if not mgc_home:
                    raise RuntimeError(
                        "MGC_HOME environment variable is not set. Please set it to the Catapult installation directory."
                    )

                ac_include = os.path.join(mgc_home, "shared/include")
                if not os.path.isdir(ac_include):
                    raise RuntimeError(
                        f"Catapult headers not found at {ac_include}. Check MGC_HOME."
                    )

                cmd = f"cd {self.project}; g++ -std=c++11 -I{ac_include} kernel.cpp host.cpp -o sim"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Compiling with g++ ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError(
                        "Failed to compile with g++. Check if g++ is installed and ac_types headers are correct."
                    )

                # Execution
                cmd = f"cd {self.project}; ./sim"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Running simulation ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Simulation failed.")

                # Read outputs
                for i, ((out_dtype, out_shape), out_arg) in enumerate(
                    zip(outputs, args[len(inputs) :])
                ):
                    if not os.path.exists(f"{self.project}/output{i}.data"):
                        raise RuntimeError(
                            f"Output file output{i}.data not found. Simulation might have failed."
                        )
                    result = read_tensor_from_file(
                        out_dtype, out_shape, f"{self.project}/output{i}.data"
                    )
                    out_arg[:] = result
                return

            if self.mode == "csyn":
                catapult_cmd = "catapult"
                if "MGC_HOME" in os.environ:
                    catapult_cmd = os.path.join(os.environ["MGC_HOME"], "bin/catapult")

                cmd = f"cd {self.project}; {catapult_cmd} -shell -f run.tcl"
                assert len(args) == 0, "csyn mode does not need to pass in arguments"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project with Catapult HLS ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError(
                        "Failed to synthesize the design with Catapult HLS"
                    )
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Catapult HLS synthesis completed successfully"
                )
                return
            raise RuntimeError(
                f"Catapult backend currently only supports 'csyn' and 'csim' mode, got '{self.mode}'"
            )
        raise RuntimeError("Not implemented")
