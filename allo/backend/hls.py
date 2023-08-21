# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=consider-using-with

import os
import re
import io
import subprocess
import time
from hcl_mlir.dialects import hcl as hcl_d
from .report import parse_xml


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
        os.system("cp " + path + "vivado/* " + project)
        if platform == "vitis_hls":
            os.system("cp " + path + "vitis/run.tcl " + project)
        os.system("cp " + path + "harness.mk " + project)
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
    def __init__(self, mod, top_func_name, mode=None, project=None):
        self.module = mod
        self.top_func_name = top_func_name
        self.mode = mode
        self.project = project
        buf = io.StringIO()
        hcl_d.emit_vhls(self.module, buf)
        buf.seek(0)
        self.hls_code = buf.read()
        if project is not None:
            assert mode is not None, "mode must be specified when project is specified"
            copy_build_files(self.top_func_name, project, mode)
            with open(f"{project}/kernel.cpp", "w", encoding="utf-8") as outfile:
                outfile.write(self.hls_code)
            with open(f"{project}/host.cpp", "w", encoding="utf-8") as outfile:
                outfile.write("")

    def __repr__(self):
        if self.mode is None:
            return self.hls_code
        return f"HLSModule({self.top_func_name}, {self.mode}, {self.project})"

    def __call__(self, shell=True):
        platform = "vivado_hls"
        if platform in {"vivado_hls", "vitis_hls"}:
            assert (
                os.system(f"which {platform} >> /dev/null") == 0
            ), f"cannot find {platform} on system path"
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
                cmd += platform
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
                raise RuntimeError(f"{platform} does not support {self.mode} mode")
        else:
            raise RuntimeError("Not implemented")
