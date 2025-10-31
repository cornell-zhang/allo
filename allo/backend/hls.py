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
)
from .._mlir.passmanager import PassManager

from .config import DEFAULT_CONFIG, PART_NUMBER
from .vitis import (
    codegen_host,
    postprocess_hls_code,
    generate_description_file,
    write_tensor_to_file,
    read_tensor_from_file,
)
from .pynq import (
    postprocess_hls_code_pynq,
    codegen_pynq_host,
)
from .tapa import (
    codegen_tapa_host,
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
    # Avoid running the full implementation flow by default. Allow opt-in
    # using configs['run_impl'] or by including 'impl' in the mode string.
    if configs.get("run_impl", False) or (isinstance(mode, str) and "impl" in mode):
        out_str += "export_design -flow impl\n"

    # Export RTL/IP for specific embedded targets (PYNQ/Ultra96/Zedboard).
    if device in ("ultra96v2", "pynqz2", "zedboard"):
        out_str += "export_design -rtl verilog -format ip_catalog\n"
    out_str += "\nexit\n"
    return out_str


def copy_ext_libs(ext_libs, project):
    for ext_lib in ext_libs:
        impl_path = ext_lib.impl
        cpp_file = impl_path.split("/")[-1]
        assert cpp_file != "kernel.cpp", "kernel.cpp is reserved for the top function"
        os.system(f"cp {impl_path} {project}/{cpp_file}")


def separate_header(hls_code, top=None):
    func_decl = False
    sig_str = "#ifndef KERNEL_H\n"
    sig_str += "#define KERNEL_H\n\n"
    args = []
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
        custom_bd_tcl=None
    ):
        self.top_func_name = top_func_name
        self.mode = mode
        self.project = project
        self.platform = platform
        self.ext_libs = [] if ext_libs is None else ext_libs
        if custom_bd_tcl and not os.path.exists(custom_bd_tcl):
            raise FileNotFoundError(f"Custom Tcl file not found: {custom_bd_tcl}")
        self.custom_bd_tcl = custom_bd_tcl

        if configs is not None:
            new_configs = DEFAULT_CONFIG
            new_configs.update(configs)
            configs = new_configs
        else:
            configs = DEFAULT_CONFIG
        if self.mode is not None:
            configs["mode"] = self.mode
        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.func = find_func_in_module(self.module, top_func_name)
            if platform == "vitis_hls" or platform == "pynq":
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
        match platform:
            case "tapa":
                allo_d.emit_thls(self.module, buf)
            case "intel_hls":
                allo_d.emit_ihls(self.module, buf)
            case _:
                allo_d.emit_vhls(self.module, buf)
        buf.seek(0)
        self.hls_code = buf.read()
        if project is not None:
            assert mode is not None, "mode must be specified when project is specified"
            os.makedirs(project, exist_ok=True)
            path = os.path.dirname(__file__)
            path = os.path.join(path, "../harness/")
            if platform in {"vivado_hls", "vitis_hls", "tapa", "pynq"}:
                os.system("cp " + path + f"{platform.split('_')[0]}/* " + project)

                if platform == "pynq" and self.custom_bd_tcl:
                    default_bd = os.path.join(project, "block_design.tcl")
                    if(os.path.exists(default_bd)):
                        os.remove(default_bd)

                with open(f"{project}/run.tcl", "w", encoding="utf-8") as outfile:
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
                generate_makefile(dst_path, project, self.platform)
                header, self.args = separate_header(self.hls_code, self.top_func_name)
                with open(f"{project}/kernel.h", "w", encoding="utf-8") as outfile:
                    outfile.write(header)
                self.hls_code = postprocess_hls_code(self.hls_code, self.top_func_name)
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
                )
            elif self.platform == "pynq":
                # PYNQ flow only supports simulation and synthesis here.
                assert self.mode in {"csim", "csyn"}, "Invalid mode for pynq"
                assert (
                    self.top_func_name != "kernel"
                ), "kernel is a reserved keyword for pynq"

                header, self.args = separate_header(self.hls_code, self.top_func_name)

                if self.custom_bd_tcl:
                    bd_dst = os.path.join(project, "custom_block_design.tcl")
                    os.system(f"cp {self.custom_bd_tcl} {bd_dst}")
                    self.custom_bd_tcl = "custom_block_design.tcl"

                with open(f"{project}/kernel.h", "w", encoding="utf-8") as outfile:
                    outfile.write(header)
                self.hls_code = postprocess_hls_code_pynq(self.hls_code, self.top_func_name)
                for lib in self.ext_libs:
                    cpp_file = lib.impl.split("/")[-1]
                    with open(f"{project}/{cpp_file}", "r", encoding="utf-8") as infile:
                        new_code = postprocess_hls_code_pynq(
                            infile.read(), lib.top, pragma=True
                        )
                    with open(
                        f"{project}/{cpp_file}", "w", encoding="utf-8"
                    ) as outfile:
                        outfile.write(new_code)

                
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
                generate_makefile(dst_path, project, self.platform)
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
                cwd = os.getcwd()
                mod = IPModule(
                    top=self.top_func_name,
                    impl=f"{cwd}/{self.project}/kernel.cpp",
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
                prefix = f"XCL_EMULATION_MODE={self.mode}" if self.mode != "hw" else ""
                prefix += f" cd {self.project};"
                if not os.path.exists(f"{self.project}/{self.top_func_name}"):
                    prefix += " make host PLATFORM=$XDEVICE;"
                cmd = f"{prefix} ./{self.top_func_name} ../{bitstream_folder}/{self.top_func_name}.xclbin"
                print(cmd)
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to run the executable")
            # suppose the last argument is the output tensor
            if np.isscalar(args[-1]):
                raise RuntimeError("The output must be a tensor")
            arr = np.fromfile(f"{self.project}/output.data", dtype=args[-1].dtype)
            args[-1][:] = arr.reshape(args[-1].shape)
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

                # vivado block design
                bd_script = self.custom_bd_tcl if self.custom_bd_tcl else "block_design.tcl"
                cmd = f"cd {self.project}; vivado -mode batch -source {bd_script}"
                print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] Running Vivado Block Design ...")
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to create block design/generate bitstream")

                # produce host code
                host_code = codegen_pynq_host(
                    self.top_func_name,
                    self.module,
                    self.project,
                )

                with open(f"{self.project}/host.py", "w", encoding="utf-8") as outfile:
                    outfile.write(host_code)

                # group .bit, .hwh, host.py -> deploy folder under the project
                deploy_dir = os.path.join(self.project, "deploy")
                cmd = (
                    f"mkdir -p {deploy_dir}; "
                    f"cp {self.project}/build_vivado/project_1.runs/impl_1/project_1_bd_wrapper.bit {deploy_dir}/{self.top_func_name}.bit; "
                    f"cp {self.project}/build_vivado/project_1.gen/sources_1/bd/project_1_bd/hw_handoff/project_1_bd.hwh {deploy_dir}/{self.top_func_name}.hwh; "
                    f"cp {self.project}/host.py {deploy_dir}/host.py"
                )

                print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] Collecting files for deployment ...")
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to collect files")

                print(f"Files for deployment located in {deploy_dir}")
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
                prefix = f"XCL_EMULATION_MODE={self.mode}" if self.mode != "hw" else ""
                prefix += f" cd {self.project};"
                if not os.path.exists(f"{self.project}/{self.top_func_name}"):
                    prefix += " make host PLATFORM=$XDEVICE;"
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
        else:
            raise RuntimeError("Not implemented")
