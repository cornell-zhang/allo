# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=consider-using-with, no-name-in-module

import os
import re
import io
import subprocess
import time
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import (
    Context,
    Module,
)
from hcl_mlir.passmanager import PassManager

from .vitis import codegen_host, postprocess_hls_code
from .report import parse_xml
from ..passes import _mlir_lower_pipeline, generate_input_output_buffers
from ..harness.makefile_gen.makegen import generate_makefile
from ..ir.transform import find_func_in_module


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
        if platform == "vitis_hls":
            # generate description file
            desc = open(
                path + "makefile_gen/description.json", "r", encoding="utf-8"
            ).read()
            desc = desc.replace("top", top)
            with open(
                os.path.join(project, "description.json"), "w", encoding="utf-8"
            ) as outfile:
                outfile.write(desc)
            # generate Makefile
            generate_makefile(os.path.join(project, "description.json"), project)
        if mode == "debug":
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


class HLSModule:
    def __init__(
        self, mod, top_func_name, platform="vivado_hls", mode=None, project=None
    ):
        self.top_func_name = top_func_name
        self.mode = mode
        self.project = project
        self.platform = platform
        with Context() as ctx:
            hcl_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.func = find_func_in_module(self.module, top_func_name)
            if platform == "vitis_hls":
                generate_input_output_buffers(self.func, flatten=True)
            _mlir_lower_pipeline(self.module, canonicalize=True, lower_linalg=True)
            # Run through lowering passes
            pm = PassManager.parse(
                "builtin.module("
                # used for lowering tensor.empty
                "empty-tensor-to-alloc-tensor,"
                # translate tensor dialect (virtual) to memref dialect (physical)
                "one-shot-bufferize{allow-return-allocs bufferize-function-boundaries},"
                # used for lowering memref.subview
                "expand-strided-metadata,"
                # common lowering passes
                "func.func(convert-linalg-to-affine-loops)"
                # DO NOT LOWER AFFINE DIALECT
                ")"
            )
            pm.run(self.module.operation)
        buf = io.StringIO()
        hcl_d.emit_vhls(self.module, buf)
        buf.seek(0)
        self.hls_code = buf.read()
        if project is not None:
            assert mode is not None, "mode must be specified when project is specified"
            copy_build_files(self.top_func_name, project, mode, platform=platform)
            if self.platform == "vitis_hls":
                self.hls_code = postprocess_hls_code(self.hls_code)
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

    def __repr__(self):
        if self.mode is None:
            return self.hls_code
        return f"HLSModule({self.top_func_name}, {self.mode}, {self.project})"

    def __call__(self, shell=True):
        if self.platform in {"vivado_hls", "vitis_hls"}:
            assert (
                os.system(f"which {self.platform} >> /dev/null") == 0
            ), f"cannot find {self.platform} on system path"
            ver = run_process("g++ --version", r"\d\.\d\.\d")[0].split(".")
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
        else:
            raise RuntimeError("Not implemented")
