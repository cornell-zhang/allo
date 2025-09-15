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

################################
AMD MLIR-AIE (AI Engine)
################################

`AMD AI Engine (AIE) <https://www.amd.com/en/products/adaptive-socs-and-fpgas/technologies/ai-engine.html>`_ is a hardware accelerator specifically designed for AI applications. It is a TPU-like architecture with an array of AIE cores.
In this document, we target the `AMD Ryzen 7040 <https://www.amd.com/en/products/processors/laptop/ryzen-for-business.html>`_ and 8040 Series processors that are built on `AMD XDNA NPUs <https://www.amd.com/en/technologies/xdna.html>`_, which are equipped with the second-generation AI Engine cores.

.. image:: https://riallto.ai/notebooks/images/png/ryzenai_array_5x4.png
   :alt: AMD Ryzen AI Engine Array
   :align: center

Environment Setup
=================

Prerequisites
-------------

Before proceeding with the Allo installation, please follow the instructions on the `MLIR-AIE website <https://github.com/Xilinx/mlir-aie/tree/main?tab=readme-ov-file#getting-started-for-amd-ryzen-ai---linux-quick-setup-instructions>`_ to install the required `Vitis <https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis.html>`_ and XRT environment. Stop when you reach the "Install IRON for AMD Ryzen™ AI AIE Application" section as we need a separate process to install MLIR-AIE under the Allo environment.


Install from Source
-------------------

Please follow the general instructions in :ref:`Install from Source <install-from-source>` to install the LLVM-19 project and the Allo package. In the following, we suppose you have already installed the LLVM-19 project and enable the ``allo`` conda environment.

Below are the exact commands to set up the environment:

Step 1
~~~~~~
Activate the ``allo`` conda environment

   .. code-block:: bash

      conda activate allo

Step 2
~~~~~~
We depend on the `MLIR-AIE <https://github.com/Xilinx/mlir-aie>`_ project to compile the Allo IR to AIE. Install release 1.0

   .. code-block:: bash

      # Install IRON library and mlir-aie from a wheel
      python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.0
      # Install Peano from a llvm-aie wheel
      python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl


   The ``mlir_aie`` wheel requires ``manylinux_2_35``, and some systems (e.g., those with glibc 2.34, confirmed by ``ldd --version``) do not meet this requirement.  
   This results in an installation failure such as:

   ``ERROR: mlir_aie-0.0.1.2025042204+24208c0-cp312-cp312-manylinux_2_35_x86_64.whl is not a supported wheel on this platform.``

Step 3
~~~~~~
Clone the mlir-aie repository and checkout to the commit corresponding to release 1.0

   .. code-block:: bash

      git clone https://github.com/Xilinx/mlir-aie.git
      cd mlir-aie
      git checkout 07320d6

Then, install python requirements, setup environment and add tools to PATHs (under ``mlir-aie``)

   .. code-block:: bash

      # Install basic Python requirements 
      python3 -m pip install -r python/requirements.txt
      # Install the pre-commit hooks defined in .pre-commit-config.yaml
      pre-commit install
      # Install MLIR Python Extras 
      HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
      # Install Torch for ML examples
      python3 -m pip install -r python/requirements_ml.txt

      source utils/env_setup.sh

.. _step4:

Step 4
~~~~~~
Clone the allo repository and install.

   You may want to set up environment variables first to use a custom CMake and LLVM build. For example:

   .. code-block:: bash

      export PATH=/opt/cmake-3.31.5-linux-x86_64/bin:/opt/llvm-project-19.x/build/bin:$PATH
      export LLVM_BUILD_DIR=/opt/llvm-project-19.x/build

   Then clone the allo repository and install by running the following commands

   .. code-block:: bash

      git clone https://github.com/cornell-zhang/allo.git
      cd allo
      python3 -m pip install -v -e .

**Note:** See :ref:`internal_install` for Zhang Group students.

.. _step5:

Step 5
~~~~~~
Setup Vitis and XRT.

**Note:** See :ref:`internal_install` for Zhang Group students.

Lastly, you can verify the AIE backend by running the following command under the ``allo`` folder.

.. code-block:: console

    python3 tests/dataflow/aie/test_vector.py


Patches and Configuration
-------------------------

To use components from the `MLIR-AIE toolchain <https://github.com/Xilinx/mlir-aie>`_ as libraries:

.. note::

   The instructions below are based on `MLIR-AIE release v1.0 <https://github.com/Xilinx/mlir-aie/releases/tag/v1.0>`_, which corresponds to commit `07320d6 <https://github.com/Xilinx/mlir-aie/tree/07320d6831b17e4a4c436d48c3301a17c1e9f1cd>`_.
   For compatibility, make sure to use this commit when copying the following components.

Clone and checkout the specific commit:

.. code-block:: bash

   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie
   git checkout 07320d6

- To use `external kernels <https://github.com/Xilinx/mlir-aie/tree/07320d6831b17e4a4c436d48c3301a17c1e9f1cd/aie_kernels>`_ as an AIE kernel library:

  .. code-block:: bash

     export MLIR_AIE_EXTERNAL_KERNEL_DIR=/your/copied/path/aie_kernels

- To use `runtime_lib <https://github.com/Xilinx/mlir-aie/tree/07320d6831b17e4a4c436d48c3301a17c1e9f1cd/runtime_lib>`_ for the host:

  .. code-block:: bash

     export RUNTIME_LIB_DIR=/your/copied/path/runtime_lib

If you run into issues when using ``aiecc.py`` such as:

.. code-block:: text

   error: expected ')' at end of argument list
   declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
                                                             ^

You can fix this by modifying ``downgrade_ir_for_peano`` in:

.. code-block:: text

   $MLIR_AIE_INSTALL_DIR/python/aie/compiler/aiecc/main.py

Update the function as follows:

**Before:**

.. code-block:: python

   def downgrade_ir_for_peano(llvmir):
       llvmir = llvmir.replace("getelementptr inbounds nuw", "getelementptr inbounds")
       return llvmir

**After:**

.. code-block:: python

   def downgrade_ir_for_peano(llvmir):
       llvmir = llvmir.replace("getelementptr inbounds nuw", "getelementptr inbounds")
       llvmir = llvmir.replace("captures(none)", "")
       return llvmir

.. _internal_install:

Internal Installation (Cornell)
-------------------------------

For Zhang Group students, please set up environment variables in :ref:`step4` with the following commands.

.. code-block:: console

      export PATH=/opt/cmake-3.31.5-linux-x86_64/bin:/opt/llvm-project-19.x/build/bin:$PATH  
      export LLVM_BUILD_DIR=/opt/llvm-project-19.x/build

And set up Vitis and XRT in :ref:`step5`  by running the following commands.

.. code-block:: console

      source /opt/common/setupVitis.sh
      source /opt/common/setupXRT.sh


Lastly, to verify the installation, you can run the following command:

.. code-block:: console

      python3 tests/dataflow/aie/test_vector.py

If the unit tests pass, then the installation is successful. Otherwise, please contact us for help.


Learning Materials
==================

- `IRON AIE Programming Guide <https://github.com/Xilinx/mlir-aie/tree/main/programming_guide>`_
- `MLIR-AIE Programming Examples <https://github.com/Xilinx/mlir-aie/tree/main/programming_examples>`_
- `MLIR-based AI Engine Design Tutorial <https://github.com/Xilinx/mlir-aie/tree/main/tutorial>`_
- `Riallto - an exploration framework for the AMD Ryzen AI NPU <https://riallto.ai/index.html>`_
