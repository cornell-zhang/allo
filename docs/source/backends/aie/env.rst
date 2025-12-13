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

Environment Setup
=================

Prerequisites
-------------

Before proceeding with the Allo installation, please follow the instructions on the `MLIR-AIE website <https://github.com/Xilinx/mlir-aie/tree/main?tab=readme-ov-file#getting-started-for-amd-ryzen-ai---linux-quick-setup-instructions>`_ to install the required `Vitis <https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis.html>`_ and XRT environment. Stop when you reach the "Install IRON for AMD Ryzen™ AI AIE Application" section as we need a separate process to install MLIR-AIE under the Allo environment.


Install from Source
-------------------

Please follow the general instructions in :ref:`Install from Source <install-from-source>` to install the latest LLVM project and the Allo package. In the following, we suppose you have already installed the LLVM project, cloned Allo repository and created the ``allo`` conda environment.

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


.. warning::

   The ``mlir_aie`` wheel requires ``manylinux_2_35``, and some systems (e.g., those with glibc 2.34, confirmed by ``ldd --version``) do not meet this requirement.  
   This results in an installation failure such as:

   ``ERROR: mlir_aie-0.0.1.2025042204+24208c0-cp312-cp312-manylinux_2_35_x86_64.whl is not a supported wheel on this platform.``

.. _step3:

Step 3
~~~~~~
Enter the Allo repository and install.

Enter the ``scripts`` directory

.. code-block:: bash

   cd scripts

Set up MLIR-AIE by running the setup script

.. code-block:: bash

   source aie-setup.sh

.. note::

   This will clone the ``mlir-aie`` repository and checkout to the commit corresponding to release 1.0. 
   
   By default, the repository is cloned under Allo's root directory. To customize the installation directory, 
   use the ``--clone-dir`` option.

   .. code-block:: bash

      source aie-setup.sh --clone-dir /customized/path/

   After running the setup script, you may see the following message in the terminal:

   .. code-block:: text

      >>> Please note: Each time you activate your environment, you need to export the following variables:
      export PATH=/path/to/your/env/lib/python3.12/site-packages/mlir_aie/bin:$PATH
      export MLIR_AIE_INSTALL_DIR=/path/to/your/env/lib/python3.12/site-packages/mlir_aie
      export PEANO_INSTALL_DIR=/path/to/your/env/lib/python3.12/site-packages/llvm-aie
      export MLIR_AIE_EXTERNAL_KERNEL_DIR=/path/to/mlir-aie/aie_kernels/
      export RUNTIME_LIB_DIR=/path/to/mlir-aie/runtime_lib/
      export PYTHONPATH=/path/to/your/env/lib/python3.12/site-packages/mlir_aie/python:$PYTHONPATH

   You can copy the export commands listed here into your own script (e.g., ``/path/to/your/env/etc/conda/activate.d/setup.sh``), so that these environment variables are automatically set whenever you activate your environment.


To build and install Allo, you may want to set up environment variables first to use a custom CMake and LLVM build. For example:

.. code-block:: bash

   export PATH=/opt/cmake-3.31.5-linux-x86_64/bin:/opt/llvm-project-19.x/build/bin:$PATH
   export LLVM_BUILD_DIR=/opt/llvm-project-19.x/build

Next, enter Allo's root directory and install by running the following commands

.. code-block:: bash

   python3 -m pip install -v -e .

.. note::

   See :ref:`internal_install` for Zhang Group students.

.. _step4:

Step 4
~~~~~~
Setup Vitis and XRT.

.. note::

   See :ref:`internal_install` for Zhang Group students.

Lastly, you can verify the AIE backend by running the following command under Allo's root directory.

.. code-block:: console

   python3 tests/dataflow/aie/test_vector.py


.. _internal_install:

Internal Installation (Cornell)
-------------------------------

For Zhang Group students, please set up environment variables in :ref:`step3` with the following commands.

.. code-block:: console

   export PATH=/opt/cmake-3.31.5-linux-x86_64/bin:/opt/llvm-project-19.x/build/bin:$PATH  
   export LLVM_BUILD_DIR=/opt/llvm-project-19.x/build

And set up Vitis and XRT in :ref:`step4` by running the following commands.

.. code-block:: console

   source /opt/common/setupVitis.sh
   source /opt/common/setupXRT.sh


Lastly, to verify the installation, you can run the following command:

.. code-block:: console

   python3 tests/dataflow/aie/test_vector.py

If the unit tests pass, then the installation is successful. Otherwise, please contact us for help.
