# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "ALLO"

# Use lit's internal shell by default, unless explicitly disabled.
use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env is not None:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

# Set the test format based on shell configuration.
config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [
    ".td",
    ".mlir",
    ".toy",
    ".ll",
    ".tc",
    ".py",
    ".yaml",
    ".test",
    ".pdll",
    ".c",
    ".spvasm",
]

# excludes: A list of files/directories to skip in the testsuite.
config.excludes = [
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
# Prefer values from lit.site.cfg.py when available, then env var, then source root.
mlir_obj_root = getattr(config, "mlir_obj_root", None) or os.environ.get(
    "MLIR_OBJ_ROOT"
)
if mlir_obj_root:
    config.test_exec_root = os.path.join(mlir_obj_root, "test")
else:
    config.test_exec_root = config.test_source_root

# Preserve commonly needed system environment variables.
llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

# Standard substitutions used by LLVM/MLIR style tests.
llvm_config.use_default_substitutions()

config.substitutions.append(("%PATH%", config.environment.get("PATH", "")))
config.substitutions.append(("%shlibext", getattr(config, "llvm_shlib_ext", "")))

# Allow optional project-specific paths to be injected from lit.site.cfg.py.
allo_obj_root = getattr(config, "allo_obj_root", None) or os.environ.get(
    "ALLO_OBJ_ROOT"
)
allo_tools_dir = getattr(config, "allo_tools_dir", None)
if not allo_tools_dir and allo_obj_root:
    allo_tools_dir = os.path.join(allo_obj_root, "bin")

allo_libs_dir = getattr(config, "allo_libs_dir", None)
if not allo_libs_dir and allo_obj_root:
    allo_libs_dir = os.path.join(allo_obj_root, "lib")

if allo_libs_dir:
    config.substitutions.append(("%allo_libs", allo_libs_dir))

tool_dirs = []
for attr in ["allo_tools_dir", "mlir_tools_dir", "llvm_tools_dir"]:
    value = getattr(config, attr, None)
    if value:
        tool_dirs.append(value)
if allo_tools_dir:
    tool_dirs.insert(0, allo_tools_dir)

if tool_dirs:
    llvm_config.with_environment("PATH", tool_dirs, append_path=True)

# Register required tools used by tests.
# Unresolved substitutions are left as-is so partial environments still work.
llvm_config.add_tool_substitutions(
    [
        ToolSubst("allo-opt", unresolved="ignore"),
        ToolSubst("FileCheck", unresolved="ignore"),
        ToolSubst("not", unresolved="ignore"),
        ToolSubst("mlir-opt", unresolved="ignore"),
    ],
    tool_dirs,
)

# Optional Python package path injection for integration tests.
python_paths = []
mlir_obj_dir = getattr(config, "mlir_obj_dir", None) or os.environ.get("MLIR_OBJ_DIR")
if mlir_obj_dir:
    python_paths.append(os.path.join(mlir_obj_dir, "python_packages"))

if python_paths:
    llvm_config.with_environment("PYTHONPATH", python_paths, append_path=True)
