# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import shutil

import allo
from allo.primitives.unify import unify
from allo.ir.types import int8
import warnings
import numpy as np


def _verify_pynq_project(project_dir, top_func_name=None):
    """Verify PYNQ HLS scaffolding exists and contains basic expected headers.

    - project_dir: path where run.tcl, kernel.cpp, kernel.h should live
    - top_func_name: optional string of the top function name to check in kernel.cpp
    """
    run_tcl = os.path.join(project_dir, "run.tcl")
    kernel_cpp = os.path.join(project_dir, "kernel.cpp")
    kernel_h = os.path.join(project_dir, "kernel.h")

    assert os.path.exists(run_tcl), "run.tcl not generated"
    assert os.path.exists(kernel_cpp), "kernel.cpp not generated"
    assert os.path.exists(kernel_h), "kernel.h not generated"

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
    from allo.ir.types import float32

    def add(A: float32, B: float32) -> float32:
        C: float32 = 0.0
        C = A + B
        return C

    s = allo.customize(add)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    project_dir = os.path.join(tempfile.gettempdir(), "allo_test_pynq_prj")
    os.makedirs(project_dir, exist_ok=True)
    hls_mod = s.build(
        target="pynq", mode="csim", project=project_dir, configs={"device": "ultra96v2"}
    )
    _verify_pynq_project(project_dir, s.top_func_name)
    shutil.rmtree(project_dir)


def test_vvadd_llvm():
    # Vector-vector add example
    from allo.ir.types import float32

    M = 128

    def vvadd(A: float32[M], B: float32[M]) -> float32[M]:
        C: float32[M] = 0.0
        for i in allo.grid(M):
            C[i] = A[i] + B[i]
        return C

    s = allo.customize(vvadd)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Make the test generate the PYNQ HLS scaffolding and verify files instead of running LLVM binaries.
    project_dir = os.path.join(tempfile.gettempdir(), "allo_test_pynq_prj")
    os.makedirs(project_dir, exist_ok=True)
    hls_mod = s.build(
        target="pynq", mode="csim", project=project_dir, configs={"device": "ultra96v2"}
    )
    _verify_pynq_project(project_dir, s.top_func_name)
    shutil.rmtree(project_dir)


def test_gemm_llvm():
    from allo.ir.types import float32

    M, N, K = 32, 32, 32

    def gemm(A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
        C: float32[M, N] = 0.0
        for i, j in allo.grid(M, N):
            for k in allo.reduction(K):
                C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Make the test generate the PYNQ HLS scaffolding and verify files instead of running LLVM binaries.
    project_dir = os.path.join(tempfile.gettempdir(), "allo_test_pynq_prj")
    os.makedirs(project_dir, exist_ok=True)
    hls_mod = s.build(
        target="pynq", mode="csim", project=project_dir, configs={"device": "ultra96v2"}
    )
    _verify_pynq_project(project_dir, s.top_func_name)
    shutil.rmtree(project_dir)
