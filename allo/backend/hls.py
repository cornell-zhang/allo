# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=consider-using-with, no-name-in-module, unused-argument

import os
import re
import io
import subprocess
import time
from .._mlir.dialects import allo as allo_d
from .._mlir.ir import (
    Context,
    Location,
    Module,
)
from .._mlir.passmanager import PassManager

from .vitis import (
    codegen_host,
    postprocess_hls_code,
    generate_description_file,
    update_makefile,
    write_tensor_to_file,
    read_tensor_from_file,
)
from .ip import IPModule, c2allo_type
from .report import parse_xml
from ..passes import (
    _mlir_lower_pipeline,
    decompose_library_function,
    generate_input_output_buffers,
)
from ..harness.makefile_gen.makegen import generate_makefile
from ..ir.transform import find_func_in_module
from ..utils import get_func_inputs_outputs

# from .. import primitives as prim


def is_available(backend="vivado_hls"):
    if backend == "vivado_hls":
        return os.system("which vivado_hls >> /dev/null") == 0
    return (
        os.system("which vitis_hls >> /dev/null") == 0
        and os.environ.get("XDEVICE", None) is not None
    )


def run_process(cmd, pattern=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if err:
        raise RuntimeError("Error raised: ", err.decode())
    if pattern:
        return re.findall(pattern, out.decode("utf-8"))
    return out.decode("utf-8")


def copy_build_files(top, project, mode, platform="vivado_hls", script=None):
    # make the project folder and copy files
    os.makedirs(project, exist_ok=True)
    path = os.path.dirname(__file__)
    path = os.path.join(path, "../harness/")
    if platform in {"vivado_hls", "vitis_hls"}:
        os.system("cp " + path + f"{platform.split('_')[0]}/* " + project)
        if mode == "debug":
            mode = "csyn"
        elif mode == "sw_emu":
            mode = "csim"
        elif mode == "hw_emu":
            mode = "cosim"
        else:
            mode = "csyn"
        if mode != "custom":
            removed_mode = ["csyn", "csim", "cosim", "impl"]
            selected_mode = mode.split("|")
            for s_mode in selected_mode:
                removed_mode.remove(s_mode)

            new_tcl = ""
            with open(
                os.path.join(project, "run.tcl"), "r", encoding="utf-8"
            ) as tcl_file:
                for line in tcl_file:
                    if "set_top" in line:
                        line = "set_top " + top + "\n"
                    # pylint: disable=too-many-boolean-expressions
                    if (
                        ("csim_design" in line and "csim" in removed_mode)
                        or ("csynth_design" in line and "csyn" in removed_mode)
                        or ("cosim_design" in line and "cosim" in removed_mode)
                        or ("export_design" in line and "impl" in removed_mode)
                    ):
                        new_tcl += "#" + line
                    else:
                        new_tcl += line
        else:  # custom tcl
            print("Warning: custom Tcl file is used, and target mode becomes invalid.")
            new_tcl = script

        with open(os.path.join(project, "run.tcl"), "w", encoding="utf-8") as tcl_file:
            tcl_file.write(new_tcl)
        return "success"
    raise RuntimeError("Not implemented")


def copy_ext_libs(ext_libs, project):
    impls = []
    headers = []
    for ext_lib in ext_libs:
        for header in ext_lib.headers:
            header_path = os.path.join(ext_lib.abs_path, header)
            os.system(f"cp {header_path} {project}")
            headers.append(header)
        for impl_path in ext_lib.impls:
            cpp_file = impl_path.split("/")[-1]
            assert (
                cpp_file != "kernel.cpp"
            ), "kernel.cpp is reserved for the top function"
            os.system(f"cp {impl_path} {project}/{cpp_file}")
            impls.append(cpp_file)


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
            allo_type = c2allo_type[ele_type]
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
    ):
        self.top_func_name = top_func_name
        self.mode = mode
        self.project = project
        self.platform = platform
        self.ext_libs = [] if ext_libs is None else ext_libs
        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.func = find_func_in_module(self.module, top_func_name)
            if platform == "vitis_hls":
                if configs is not None:
                    mappings = configs.get("mappings", None)
                else:
                    mappings = None
                # buffers = generate_input_output_buffers(
                generate_input_output_buffers(
                    self.func, flatten=True, mappings=mappings
                )
                # TODO: Fix dataflow!
                # if "dataflow" in self.func.attributes:
                #     assert func_args is not None, "Need to specify func_args"
                #     for inp in buffers["inputs"]:
                #         prim.to(
                #             self.module,
                #             inp,
                #             "",
                #             depth=4,
                #             func_args=func_args,
                #             top_func_name=top_func_name,
                #         )
                #     for out in buffers["outputs"]:
                #         prim.to(
                #             self.module,
                #             out,
                #             "",
                #             depth=4,
                #             func_args=func_args,
                #             top_func_name=top_func_name,
                #         )
            self.module = decompose_library_function(self.module)
            _mlir_lower_pipeline(self.module, lower_linalg=True)
            # Run through lowering passes
            pm = PassManager.parse(
                "builtin.module("
                # used for lowering tensor.empty
                "empty-tensor-to-alloc-tensor,"
                # translate tensor dialect (virtual) to memref dialect (physical)
                "one-shot-bufferize{bufferize-function-boundaries},"
                # common lowering passes
                "func.func(convert-linalg-to-affine-loops)"
                # DO NOT LOWER AFFINE DIALECT
                ")"
            )
            pm.run(self.module.operation)
        buf = io.StringIO()
        allo_d.emit_vhls(self.module, buf)
        buf.seek(0)
        self.hls_code = buf.read()
        if project is not None:
            assert mode is not None, "mode must be specified when project is specified"
            copy_build_files(self.top_func_name, project, mode, platform=platform)
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
                )
                generate_makefile(dst_path, project)
                for postfix in ("us_alveo", "versal_alveo", "versal_ps", "zynqmp"):
                    update_makefile(
                        os.path.join(project, f"makefile_{postfix}.mk"), self.ext_libs
                    )
                header, self.args = separate_header(self.hls_code, self.top_func_name)
                with open(f"{project}/kernel.h", "w", encoding="utf-8") as outfile:
                    outfile.write(header)
                self.hls_code = postprocess_hls_code(self.hls_code, self.top_func_name)
                for lib in self.ext_libs:
                    for header in lib.headers:
                        header = header.split("/")[-1]
                        with open(
                            f"{project}/{header}", "r", encoding="utf-8"
                        ) as infile:
                            new_code = postprocess_hls_code(infile.read())
                        with open(
                            f"{project}/{header}", "w", encoding="utf-8"
                        ) as outfile:
                            outfile.write(new_code)
                    for impl_path in lib.impls:
                        cpp_file = impl_path.split("/")[-1]
                        with open(
                            f"{project}/{cpp_file}", "r", encoding="utf-8"
                        ) as infile:
                            new_code = postprocess_hls_code(infile.read())
                        with open(
                            f"{project}/{cpp_file}", "w", encoding="utf-8"
                        ) as outfile:
                            outfile.write(new_code)
                self.host_code = codegen_host(
                    self.top_func_name,
                    self.module,
                )
            else:
                self.host_code = ""
            with open(f"{project}/kernel.cpp", "w", encoding="utf-8") as outfile:
                outfile.write(self.hls_code)
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
                                for header in lib.headers:
                                    header = header.split("/")[-1]
                                    new_kernel += f'#include "{header}"\n'
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
                                for impl in lib.impls:
                                    cpp_file = impl.split("/")[-1]
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
                    subprocess.Popen(cmd, shell=True).wait()
                else:
                    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).wait()
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
                    headers=[f"{cwd}/{self.project}/kernel.h"],
                    impls=[f"{cwd}/{self.project}/kernel.cpp"],
                    signature=[
                        f"{dtype}[{', '.join(shape)}]" for dtype, shape in self.args
                    ],
                    link_hls=True,
                )
                mod(*args)
                return
            if self.mode == "csyn":
                cmd = f"cd {self.project}; vitis_hls -f run.tcl"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project ..."
                )
                if shell:
                    subprocess.Popen(cmd, shell=True).wait()
                else:
                    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).wait()
                return
            # Use Makefile (sw_emu, hw_emu, hw)
            assert "XDEVICE" in os.environ, "Please set XDEVICE in your environment"
            # prepare data
            func = find_func_in_module(self.module, self.top_func_name)
            inputs, _ = get_func_inputs_outputs(func)
            for i, ((in_dtype, in_shape), arg) in enumerate(zip(inputs, args)):
                write_tensor_to_file(
                    arg,
                    in_dtype,
                    in_shape,
                    f"in_data_{i}",
                    f"{self.project}/input_{i}.h",
                )
            # check if the build folder exists
            bitstream_folder = f"{self.project}/build_dir.{self.mode}.{os.environ['XDEVICE'].rsplit('/')[-1].split('.')[0]}"
            if not os.path.exists(bitstream_folder):
                cmd = (
                    f"cd {self.project}; make run TARGET={self.mode} PLATFORM=$XDEVICE"
                )
                print(cmd)
                if shell:
                    subprocess.Popen(cmd, shell=True).wait()
                else:
                    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).wait()
            else:
                print("Build folder exists, skip building")
                # run the executable
                cmd = f"cd {self.project}; ./top ../{bitstream_folder}/top.xclbin"
                subprocess.Popen(cmd, shell=True).wait()
            # suppose the last argument is the output tensor
            args[-1][:] = read_tensor_from_file(
                inputs[-1][0], inputs[-1][1], f"{self.project}/output.data"
            )
            return
        else:
            raise RuntimeError("Not implemented")
