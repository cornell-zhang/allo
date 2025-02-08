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

##############
AMD AI Engine
##############

`AMD AI Engine (AIE) <https://www.amd.com/en/products/adaptive-socs-and-fpgas/technologies/ai-engine.html>`_ is a hardware accelerator specifically designed for AI applications. It is a TPU-like architecture with an array of AIE cores.
In this document, we target the `AMD Ryzen 7040 <https://www.amd.com/en/products/processors/laptop/ryzen-for-business.html>`_ and 8040 Series processors that are built on `AMD XDNA NPUs <https://www.amd.com/en/technologies/xdna.html>`_, which are equipped with the second-generation AI Engine cores.

.. image:: https://riallto.ai/notebooks/images/png/ryzenai_array_5x4.png
   :alt: AMD Ryzen AI Engine Array
   :align: center


Prerequisites
-------------

Before proceeding with the Allo installation, please follow the instructions on the `MLIR-AIE website <https://github.com/Xilinx/mlir-aie/tree/main?tab=readme-ov-file#getting-started-for-amd-ryzen-ai---linux-quick-setup-instructions>`_ to install the required `Vitis <https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis.html>`_ and XRT environment. Stop when you reach the "Install IRON for AMD Ryzen™ AI AIE Application" section as we need a separate process to install MLIR-AIE under the Allo environment.


Install from Source
-------------------

Please follow the general instructions in :ref:`Install from Source <install-from-source>` to install the LLVM-19 project and the Allo package. In the following, we suppose you have already installed the LLVM-19 project and enable the ``allo`` conda environment.

We depend on the `MLIR-AIE <https://github.com/Xilinx/mlir-aie>`_ project to compile the Allo IR to AIE, but since we are using a specific LLVM-19 version that is not compatible with the latest MLIR-AIE project, we cannot follow the offical instructions but build the MLIR-AIE project from source.

First, clone the MLIR-AIE project and checkout to the specific commit.

.. code-block:: console

  $ git clone --recursive https://github.com/Xilinx/mlir-aie.git
  $ cd mlir-aie && git checkout fd89c9
  $ export MLIR_AIE_ROOT_DIR=$(pwd)

Then, build the MLIR-AIE project. The second command will install the PEANO backend compiler for AIE. Please make sure the current Python environment is Python 3.12 and you have already set up the Vitis and XRT environment. You can check by running ``which vitis``.

.. code-block:: console

  $ ./utils/build-mlir-aie.sh $LLVM_BUILD_DIR
  $ source env_setup.sh $MLIR_AIE_ROOT_DIR/install $LLVM_BUILD_DIR

As the above setup script will append additional environment variables to your current shell, you need to relaunch a new shell for the following steps.

.. code-block:: console

  $ export PATH=$MLIR_AIE_ROOT_DIR/install/bin:$PATH
  $ export PYTHONPATH=$MLIR_AIE_ROOT_DIR/install/python:$PYTHONPATH
  $ export MLIR_AIE_INSTALL_DIR=$MLIR_AIE_ROOT_DIR/install
  $ export PEANO_INSTALL_DIR=$MLIR_AIE_ROOT_DIR/utils/my_install/llvm-aie
  $ # install python packages under the allo conda environment
  $ python3 -m pip install -r $MLIR_AIE_ROOT_DIR/python/requirements.txt
  $ HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r $MLIR_AIE_ROOT_DIR/requirements_extras.txt

Lastly, you can verify the AIE backend by running the following command under the ``allo`` folder.

.. code-block:: console

  $ python3 tests/dataflow/aie/test_vector.py


Internal Installation (Cornell)
-------------------------------

For Zhang Group students, we have already set up the environment for LLVM and MLIR-AIE, so you do not need to build everything from source. Firstly, you need to create a new conda environment with Python 3.12 and activate it.

.. code-block:: console

  $ conda create -n allo python=3.12
  $ conda activate allo

To set up the environment, all you need to do is to source the following script.

.. code-block:: console

  $ source /opt/common/setup.sh

Then, go through the normal steps to install Allo:

.. code-block:: console

  $ git clone https://github.com/cornell-zhang/allo.git && cd allo
  $ python3 -m pip install -v -e .

Some additional packages are required to run the MLIR-AIE compiler. You can install them by running the following commands:

.. code-block:: console

  $ python3 -m pip install -r /opt/mlir-aie/python/requirements.txt
  $ HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r /opt/mlir-aie/python/requirements_extras.txt

Lastly, to verify the installation, you can run the following command:

.. code-block:: console

  $ python3 tests/dataflow/aie/test_vector.py

If the unit tests pass, then the installation is successful. Otherwise, please contact us for help.


Learning Materials
------------------

- `IRON AIE Programming Guide <https://github.com/Xilinx/mlir-aie/tree/main/programming_guide>`_
- `MLIR-AIE Programming Examples <https://github.com/Xilinx/mlir-aie/tree/main/programming_examples>`_
- `MLIR-based AI Engine Design Tutorial <https://github.com/Xilinx/mlir-aie/tree/main/tutorial>`_
- `Riallto - an exploration framework for the AMD Ryzen AI NPU <https://riallto.ai/index.html>`_
