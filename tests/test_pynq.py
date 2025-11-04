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


def test_add_llvm():
    # Simple scalar add
    from allo.ir.types import float32

    def add(A: float32, B: float32) -> float32:
        C: float32 = 0.0
        C = A + B
        return C

    s = allo.customize(add)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Build for local execution (LLVM) and run a few random checks
    executable = s.build(target="llvm")

    for _ in range(5):
        # Use Python floats for scalar LLVM execution (LLVMModule expects int/float or numpy arrays)
        a = float(np.random.rand())
        b = float(np.random.rand())
        out = executable(a, b)
        assert np.allclose(out, a + b, rtol=1e-3, atol=1e-3)


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

    executable = s.build(target="llvm")

    np_A = np.random.rand(M).astype(np.float32)
    np_B = np.random.rand(M).astype(np.float32)
    np_C = executable(np_A, np_B)

    golden_C = np.add(np_A, np_B)
    np.testing.assert_allclose(np_C, golden_C, rtol=1e-3, atol=1e-3)


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

    executable = s.build(target="llvm")

    np_A = np.random.rand(M, K).astype(np.float32)
    np_B = np.random.rand(K, N).astype(np.float32)
    np_C = executable(np_A, np_B)

    golden_C = np.matmul(np_A, np_B)
    np.testing.assert_allclose(np_C, golden_C, rtol=1e-3, atol=1e-3)


def test_pynq_files_generated():
    # Small kernel to trigger HLS generation for PYNQ
    from allo.ir.types import float32

    def add(A: float32, B: float32) -> float32:
        C: float32 = 0.0
        C = A + B
        return C

    s = allo.customize(add)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Create a temporary project directory and build for PYNQ (csim mode is sufficient to generate files)
    project_dir = tempfile.mkdtemp()
    try:
        hls_mod = s.build(target="pynq", mode="csim", project=project_dir)

        # Check files exist
        run_tcl = os.path.join(project_dir, "run.tcl")
        kernel_cpp = os.path.join(project_dir, "kernel.cpp")
        kernel_h = os.path.join(project_dir, "kernel.h")

        assert os.path.exists(run_tcl), "run.tcl not generated"
        assert os.path.exists(kernel_cpp), "kernel.cpp not generated"
        assert os.path.exists(kernel_h), "kernel.h not generated"

        # Check first few lines of run.tcl contain the expected copyright/header
        with open(run_tcl, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert any("Copyright Allo authors" in l for l in lines[:6])

        # kernel.cpp should contain the top function declaration
        with open(kernel_cpp, "r", encoding="utf-8") as f:
            content = f.read()
        assert (
            f"void {s.top_func_name}" in content
            or f"void {s.top_func_name}(" in content
        )

        # kernel.h should contain include guard and extern "C"
        with open(kernel_h, "r", encoding="utf-8") as f:
            hcontent = f.read()
        assert "KERNEL_H" in hcontent and 'extern "C"' in hcontent

    finally:
        shutil.rmtree(project_dir)
