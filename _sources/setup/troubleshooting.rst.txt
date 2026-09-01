..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _source-installation-troubleshooting:

###################################
Source Installation Troubleshooting
###################################

This page covers failures in the Linux source-installation workflow. The
commands assume that the repository root is stored in ``ALLO_ROOT`` and that
the ``allo`` Conda environment is active. Follow :ref:`install-from-source`
before using the fixes below.

Initial Checks
==============

Run these checks before retrying a long build:

.. code-block:: bash

    echo "ALLO_ROOT=$ALLO_ROOT"
    echo "LLVM_BUILD_DIR=$LLVM_BUILD_DIR"
    pwd
    python3 --version
    python3 -c "import sys; print(sys.executable)"
    cmake --version
    ninja --version
    cc --version
    c++ --version
    python3 -m pip show nanobind

The expected results are:

* Python (must be at least 3.12) belongs to the active ``allo`` environment.
* CMake, Ninja, and C/C++ compilers are available.
* nanobind is version 2.x. Current Allo builds require
  ``nanobind>=2.9,<3.0``.
* ``ALLO_ROOT`` points to the Allo repository root.
* ``LLVM_BUILD_DIR`` points to the LLVM *build* directory, not the LLVM source
  directory.

If the issues and fixes below do not cover the problem, raise a GitHub issue
for help. Include the failing command, the first relevant error, the output of
the checks above, and these additional diagnostics. Avoid including only the
final generic pip traceback.

.. code-block:: bash

    git rev-parse HEAD
    git submodule status
    test -x "$LLVM_BUILD_DIR/bin/llvm-config"; echo "llvm-config: $?"
    test -f "$LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake"; echo "MLIRConfig.cmake: $?"

Editable Install Runs in the Wrong Directory
============================================

**Symptom**

.. code-block:: text

    ERROR: ... does not appear to be a Python project:
    neither 'setup.py' nor 'pyproject.toml' found.

**Cause**

The LLVM build finishes in ``allo/externals/llvm-project/build``. Running
``pip install -e .`` there asks pip to install the LLVM build directory rather
than Allo.

**Fix**

Return to the repository root and confirm that the packaging files exist:

.. code-block:: bash

    cd "$ALLO_ROOT"
    test -f pyproject.toml
    test -f setup.py
    python3 -m pip install -v -e .

``LLVM_BUILD_DIR`` Is Not Set
=============================

**Symptom**

.. code-block:: text

    RuntimeError: LLVM_BUILD_DIR environment variable is not set

**Cause**

Allo uses ``LLVM_BUILD_DIR`` to locate the LLVM and MLIR build products. A new
shell does not retain variables exported in an earlier shell.

**Fix**

Set the variable to the absolute LLVM build directory, add its tools to
``PATH``, and verify the expected files:

.. code-block:: bash

    export LLVM_BUILD_DIR="$ALLO_ROOT/externals/llvm-project/build"
    export PATH="$LLVM_BUILD_DIR/bin:$PATH"

    test -x "$LLVM_BUILD_DIR/bin/llvm-config"
    test -f "$LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake"

If either check fails, the path is wrong or the LLVM/MLIR build is incomplete.
The presence of ``llvm-config`` alone does not prove that MLIR finished
building.

``MLIRConfig.cmake`` Cannot Be Found
====================================

**Symptom**

.. code-block:: text

    Could not find a package configuration file provided by "MLIR"
    with any of the following names:
      MLIRConfig.cmake
      mlir-config.cmake

**Check**

.. code-block:: bash

    test -f "$LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake"

**Cause**

The most common causes are:

* ``LLVM_BUILD_DIR`` points to the source directory or a different build.
* The LLVM build stopped before producing the MLIR CMake package.
* LLVM was configured without MLIR in ``LLVM_ENABLE_PROJECTS``.

**Fix**

Configure the pinned LLVM submodule with MLIR enabled, finish the Ninja build,
and repeat the check before installing Allo:

.. code-block:: bash

    cd "$ALLO_ROOT/externals/llvm-project/build"
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="clang;mlir;openmp" \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE="$(command -v python3)"
    ninja

    export LLVM_BUILD_DIR="$(pwd)"
    test -f "$LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake"

Use the complete CMake options from :ref:`install-from-source` when creating a
new build directory.

nanobind Is Missing During LLVM Configuration
==============================================

**Symptom**

.. code-block:: text

    not found (install via 'pip install nanobind' or set nanobind_DIR)

**Cause**

MLIR checks for nanobind in the Python interpreter selected by
``Python3_EXECUTABLE``. Installing nanobind later, while installing Allo, does
not make it available to an earlier LLVM configuration.

**Fix**

Activate the same environment used to configure LLVM, install the supported
nanobind version, and rerun CMake:

.. code-block:: bash

    conda activate allo
    python3 -m pip install "nanobind>=2.9,<3.0"
    python3 -c "import sys; print(sys.executable)"
    python3 -c "import nanobind; print(nanobind.__version__)"
    python3 -c "import nanobind; print(nanobind.cmake_dir())"

    cd "$ALLO_ROOT/externals/llvm-project/build"
    cmake -DPython3_EXECUTABLE="$(command -v python3)" ../llvm

The Python path printed by ``python3 -c`` must be the interpreter intended for
both LLVM and Allo.

Linux Wheel Is Rejected on Another Platform
===========================================

**Symptom**

.. code-block:: text

    past-0.7.2-cp312-cp312-linux_x86_64.whl is not a supported wheel on this platform

**Cause**

Allo currently depends on a CPython 3.12 Linux x86-64 wheel for ``past``.
Pip rejects that wheel on Windows, macOS, other architectures, and other
Python versions.

**Fix**

Use the documented Docker installation or a Python 3.12 Linux x86-64
environment. Other platforms require a compatible ``past`` build and are not
covered by the current source-installation commands.

``_py_stats`` Is Undefined When Importing Allo
==============================================

**Symptom**

.. code-block:: text

    ImportError: .../_mlir.cpython-312-x86_64-linux-gnu.so:
    undefined symbol: _py_stats

**Cause**

``_py_stats`` is a private CPython symbol used when Python is built with
internal performance statistics enabled. This error can occur when Allo's
compiled MLIR extension was built against a Python configuration that provided
the symbol, but the current Python runtime does not provide it.

Allo generates the extension under ``mlir/build`` and imports it through
``allo/_mlir``. The build directory is reused by later editable installs. If
Python is rebuilt in place, or the active Python installation changes while
keeping the same extension suffix, the existing MLIR build can be incompatible
with the current runtime.

**Checks**

First confirm the current interpreter and its build configuration:

.. code-block:: bash

    python3 -c "import sys; print(sys.executable)"
    python3 -c "import sysconfig; print(sysconfig.get_config_var('CONFIG_ARGS'))"
    python3 -c "import sysconfig; print('Py_STATS:', sysconfig.get_config_var('Py_STATS'))"
    python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"

A runtime built without performance statistics normally reports
``Py_STATS: 0``. Next locate the generated Allo extension and inspect its
dynamic symbols:

.. code-block:: bash

    MLIR_EXTENSION="$(find \
        "$ALLO_ROOT/mlir/build/tools/allo/_mlir/_mlir_libs" \
        -maxdepth 1 -name '_mlir*.so' -print -quit)"
    echo "MLIR_EXTENSION=$MLIR_EXTENSION"
    test -n "$MLIR_EXTENSION"
    nm -D "$MLIR_EXTENSION" | grep -w _py_stats

An entry such as the following proves that the extension expects the symbol
from the Python runtime:

.. code-block:: text

                     U _py_stats

Check whether the current Python process exports the symbol:

.. code-block:: bash

    python3 -c "import ctypes; getattr(ctypes.CDLL(None), '_py_stats')"

A runtime that does not provide the symbol reports an ``AttributeError``
ending in ``undefined symbol: _py_stats``.

If the extension reports ``U _py_stats``, the current runtime has
``Py_STATS: 0``, and the Python process does not export ``_py_stats``, the
extension and runtime are inconsistent.

**Fix**

Remove only Allo's generated MLIR build, then rebuild it with the active Python
and the existing LLVM build:

.. code-block:: bash

    conda activate allo || exit 1
    cd "$ALLO_ROOT" || exit 1
    test -f setup.py || exit 1
    test -d mlir || exit 1
    test -f "$LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake" || exit 1

    rm -rf -- mlir/build
    python3 -m pip install -v -e .
    python3 -c "import allo; import allo.ir; print('Allo MLIR import passed')"

Do not remove ``LLVM_BUILD_DIR`` for this error. It contains the LLVM/MLIR
backend build and is separate from Allo's generated extension in
``$ALLO_ROOT/mlir/build``. A clean Allo extension rebuild is appropriate only
after the checks above demonstrate a Python/extension mismatch; it is not a
general fix for unrelated import errors.

Vitis Environment Library Conflicts
===================================

**Symptom**

The Allo build fails while CMake is starting:

.. code-block:: console

  running build_ext
    cmake: error while loading shared libraries: libidn.so.11: cannot open shared object file: No such file or directory

**Cause**

The Vitis environment is sourced in the same shell and introduces conflicting
library paths.

**Fix**

Disable the Vitis environment, or start a new shell without sourcing it, and
then build Allo again:

.. code-block:: bash

    cd "$ALLO_ROOT"
    python3 -m pip install -v -e .

Too Many Open Files
===================

**Symptom**

The linker reports that it cannot open an LLVM or MLIR library even though the
file exists:

.. code-block:: console

  /opt/rh/devtoolset-9/root/usr/libexec/gcc/x86_64-redhat-linux/9/ld: cannot find /work/shared/common/llvm-project-main/build/lib/libMLIRLLVMToLLVMIRTranslation.a: Too many open files

**Cause**

The build has reached the shell's limit on simultaneously open file
descriptors.

**Fix**

Increase the limit for the current shell, then build Allo again:

.. code-block:: bash

    ulimit -n 4096
    cd "$ALLO_ROOT"
    python3 -m pip install -v -e .
