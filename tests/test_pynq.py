# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import shutil

import allo
from allo.primitives.unify import unify
from allo.ir.types import int8
from allo.ir.types import float32

import warnings
import numpy as np


def _verify_pynq_project(project_dir, top_func_name=None):
    """Verify PYNQ HLS scaffolding exists and contains basic expected headers.

    - project_dir: path where run.tcl, kernel.cpp, kernel.h should live
    - top_func_name: optional string of the top function name to check in kernel.cpp
    """
    run_tcl = os.path.join(project_dir, "run.tcl")
    bd_tcl = os.path.join(project_dir, "block_design.tcl")
    kernel_cpp = os.path.join(project_dir, "kernel.cpp")
    kernel_h = os.path.join(project_dir, "kernel.h")

    assert os.path.exists(run_tcl), "run.tcl not generated"
    assert os.path.exists(kernel_cpp), "kernel.cpp not generated"
    assert os.path.exists(kernel_h), "kernel.h not generated"
    assert os.path.exists(bd_tcl), "block_design.tcl not generated"

    # run.tcl header
    with open(run_tcl, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert any("Copyright Allo authors" in l for l in lines[:6])

    # optional: check kernel.cpp contains the top function declaration
    if top_func_name is not None:
        with open(kernel_cpp, "r", encoding="utf-8") as f:
            content = f.read()
        assert (f"void {top_func_name}" in content) or (
            f"void {top_func_name}(" in content
        )

    # kernel.h should contain include guard and extern "C"
    with open(kernel_h, "r", encoding="utf-8") as f:
        hcontent = f.read()
    assert "KERNEL_H" in hcontent and 'extern "C"' in hcontent


def test_add_pynq():
    # Simple scalar add

    def add(A: float32, B: float32) -> float32:
        C: float32 = 0.0
        C = A + B
        return C

    s = allo.customize(add)

    project_dir = os.path.join(tempfile.gettempdir(), "allo_test_pynq_prj")
    os.makedirs(project_dir, exist_ok=True)
    hls_mod = s.build(
        target="pynq", mode="csim", project=project_dir, configs={"device": "ultra96v2"}
    )
    _verify_pynq_project(project_dir, s.top_func_name)
    shutil.rmtree(project_dir)


def test_vvadd_pynq():
    # Vector-vector add example

    # Make the test generate the PYNQ HLS scaffolding and verify files instead of running LLVM binaries.
    project_dir = os.path.join(tempfile.gettempdir(), "allo_test_pynq_prj")
    os.makedirs(project_dir, exist_ok=True)
    hls_mod = s.build(
        target="pynq", mode="impl", project=project_dir, configs={"device": "ultra96v2"}
    )
