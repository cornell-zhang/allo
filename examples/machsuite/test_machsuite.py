# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# MachSuite benchmark test suite
# Each test imports and runs the benchmark's test function with "small" problem sizes.
# Run with: pytest examples/machsuite/test_machsuite.py -v

import importlib.util
import os
import sys

_machsuite_dir = os.path.dirname(os.path.abspath(__file__))


def _load_test(subdir, filename, func_name):
    """Load a test function from a file using importlib to avoid package conflicts."""
    filepath = os.path.join(_machsuite_dir, subdir, filename)
    # Ensure the file's directory is on sys.path for its local imports
    filedir = os.path.dirname(filepath)
    if filedir not in sys.path:
        sys.path.insert(0, filedir)
    # Clear potentially conflicting cached modules (e.g., 'md' used by both grid and knn)
    for name in list(sys.modules):
        mod = sys.modules[name]
        if hasattr(mod, "__file__") and mod.__file__ and filedir in mod.__file__:
            continue  # keep modules from this directory
        # Remove short module names that might conflict across sub-benchmarks
        mod_file = getattr(mod, "__file__", None)
        if mod_file and _machsuite_dir in mod_file and name.count(".") == 0:
            if name not in (
                "allo",
                "numpy",
                "np",
                "os",
                "sys",
                "json",
                "math",
                "random",
            ):
                del sys.modules[name]
    mod_name = f"machsuite_{subdir.replace('/', '_')}_{filename.replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


def test_aes():
    _load_test("aes", "run_test.py", "test_aes")()


def test_backprop():
    _load_test("backprop", "run_test.py", "test_backprop")("small")


def test_bfs_bulk():
    _load_test("bfs/bulk", "run_test.py", "test_bfs_bulk")("small")


def test_bfs_queue():
    _load_test("bfs/queue", "run_test.py", "test_bfs_queue")("small")


def test_fft_strided():
    _load_test("fft/strided", "run_test.py", "test_strided_fft")("small")


def test_fft_transpose():
    _load_test("fft/transpose", "run_test.py", "test_transpose_fft")("small")


def test_gemm_ncubed():
    _load_test("gemm", "run_test.py", "test_gemm_ncubed")("small")


def test_gemm_blocked():
    _load_test("gemm", "run_test.py", "test_gemm_blocked")("small")


def test_kmp():
    _load_test("kmp", "kmp.py", "test_kmp")("small")


def test_md_grid():
    _load_test("md/grid", "run_test.py", "test_md_grid")("small")


def test_md_knn():
    _load_test("md/knn", "run_test.py", "test_md_knn")("small")


def test_merge():
    _load_test("merge", "run_test.py", "test_merge")("small")


def test_nw():
    _load_test("nw", "run_test.py", "test_nw")("small")


def test_radix():
    _load_test("radix_sort", "run_test.py", "test_radix_sort")("small")


def test_spmv_crs():
    _load_test("spmv/crs", "run_test.py", "test_spmv_crs")("small")


def test_spmv_ellpack():
    _load_test("spmv/ellpack", "run_test.py", "test_spmv_ellpack")("small")


def test_stencil2d():
    _load_test("stencil", "run_test.py", "test_stencil2d")("small")


def test_stencil3d():
    _load_test("stencil", "run_test.py", "test_stencil3d")("small")


def test_viterbi():
    _load_test("viterbi", "run_test.py", "test_viterbi")("small")
