# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=consider-using-with

import os
import re
import subprocess


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
    path = os.path.join(path, "harness/")
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
