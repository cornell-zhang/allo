# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# RUN: %PYTHON %s
import os
import ctypes
import numpy as np

from hcl_mlir.ir import *
from hcl_mlir.passmanager import *
from hcl_mlir.execution_engine import *
from hcl_mlir.runtime import *
from hcl_mlir.dialects import hcl as hcl_d


def get_assembly(filename):
    with open(filename, "r") as f:
        code = f.read()
    return code


def test_execution_engine(P=16, Q=22, R=18, S=24):
    code = get_assembly(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "affine_dialect.mlir"))

    # Add shared library
    if os.getenv("LLVM_BUILD_DIR") is not None:
        shared_libs = [
            os.path.join(os.getenv("LLVM_BUILD_DIR"),
                         'lib', 'libmlir_runner_utils.so'),
            os.path.join(os.getenv("LLVM_BUILD_DIR"),
                         'lib', 'libmlir_c_runner_utils.so')
        ]
    else:
        shared_libs = None

    A = np.random.randint(10, size=(P, Q)).astype(np.float32)
    B = np.random.randint(10, size=(Q, R)).astype(np.float32)
    C = np.random.randint(10, size=(R, S)).astype(np.float32)
    D = np.random.randint(10, size=(P, S)).astype(np.float32)
    # res1 = np.zeros((P, S), dtype=np.float32)

    A_memref = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(A)))
    B_memref = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(B)))
    C_memref = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(C)))
    D_memref = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(D)))
    # res1_memref = ctypes.pointer(
    # ctypes.pointer(get_ranked_memref_descriptor(res1))
    # )
    res1 = make_nd_memref_descriptor(2, ctypes.c_float)()
    res1_memref = ctypes.pointer(ctypes.pointer(res1))

    with Context() as ctx:
        module = Module.parse(code)
        hcl_d.lower_hcl_to_llvm(module, ctx)
        if shared_libs is not None:
            execution_engine = ExecutionEngine(
                module, opt_level=3, shared_libs=shared_libs)
        else:
            execution_engine = ExecutionEngine(module)
        execution_engine.invoke(
            "top", res1_memref, A_memref, B_memref, C_memref, D_memref)

    ret = ranked_memref_to_numpy(res1_memref[0])
    golden = 0.1 * np.matmul(np.matmul(A, B), C) + 0.1 * D
    assert np.allclose(ret, golden)


if __name__ == "__main__":
    test_execution_engine()
