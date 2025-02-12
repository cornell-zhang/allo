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

.. _setup:

############
Installation
############

To install and use Allo, we need to first install the `LLVM-19 project <https://github.com/cornell-zhang/allo/tree/main/externals>`_. You can choose to use our provided docker or build from source.


Install from Docker
-------------------

To simplify the installation process, we provide a docker image that has already installed the LLVM-19 project. Please pull the image from Docker Hub as described below. The LLVM is installed under the :code:`/root/llvm-project` folder in the docker image.

.. code-block:: console

  $ docker pull chhzh123/allo:llvm-19.x-py3.12
  $ docker run --rm -it chhzh123/allo:llvm-19.x-py3.12
  (docker) $ git clone https://github.com/cornell-zhang/allo.git && cd allo
  (docker) $ python3 -m pip install -v -e .


.. _install-from-source:

Install from Source
-------------------

Please follow the instructions below to build the LLVM-19 project from source. You can also refer to the `official guide <https://mlir.llvm.org/getting_started/>`_ for more details. As the LLVM/MLIR API changes a lot, if you are using a different LLVM version, the Allo package may not work properly. The LLVM version we used can be found in the `externals <https://github.com/cornell-zhang/allo/tree/main/externals>`_ folder.

.. code-block:: bash

    git clone --recursive https://github.com/cornell-zhang/allo.git
    cd allo/externals/llvm-project
    # Apply our patch
    git apply ../llvm_patch
    # Python 3.12 is required
    mkdir -p build && cd build
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_INSTALL_UTILS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE=`which python3`
    ninja
    # export environment variable
    export LLVM_BUILD_DIR=$(pwd)

We recommend creating a new conda environment for Allo. Since we are using the latest Python features, the minimum Python version is **3.12**.

.. code-block:: console

  $ conda create -n allo python=3.12
  $ conda activate allo

You can now install Allo by running the following command.

.. code-block:: console

  $ python3 -m pip install -v -e .


Testing
-------

To make sure the installation is successful, you can run the following command to test the Allo package.

.. code-block:: console

  $ python3 -m pytest tests/


Internal Installation (Cornell)
-------------------------------
For Zhang Group students, we have already prepared a prebuilt version of LLVM on our server, so you do not need to build everything from source. Please follow the instruction below to set up the environment.

Make sure you have the access to the :code:`brg-zhang` or other Zhang group servers. You can log into the server by SSH or use VSCode Remote SSH extension. Please refer to `this website <https://code.visualstudio.com/docs/remote/ssh>`_ for more details on configuring VSCode. Those servers are only accessible from the campus network. Please use the VPN if you are off-campus.

After logging into the server, the first step is to install an Anaconda environment. We recommend you to install your own `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, which is a lightweight version of Anaconda and contains only the necessary packages. You can download the installer from the link above and install it on your system. After the installation, you can create a new environment for Allo by running the following commands:

.. code-block:: console

  $ conda create -n allo python=3.12
  $ conda activate allo

We also provide a script to set up the backend LLVM compiler. You can simply run it

.. code-block:: console

  $ source /work/shared/common/allo/setup-llvm19.sh

.. note::

  You can also add this line to your :code:`~/.bashrc` file so that you don't need to run the setup script every time.

Then, you can pull the latest version of Allo from GitHub and install it by running

.. code-block:: console

  $ git clone https://github.com/cornell-zhang/allo.git
  $ cd allo
  $ python3 -m pip install -v -e .

Now, you can run the following command to test if the installation is successful

.. code-block:: console

  $ python3 -c "import allo as allo; import allo.ir as air"

If you see no error messages, then the installation is successful. Otherwise, please contact us for help.
