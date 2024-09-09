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

To install and use Allo, we need to first install the `LLVM-18 project <https://github.com/llvm/llvm-project/tree/llvmorg-18-init>`_ and the `hcl-mlir dialect <https://github.com/cornell-zhang/hcl-dialect>`_. You can choose to use our provided docker or build from source.

Install from Docker
-------------------

To simplify the installation process, we provide a docker image that has already installed the LLVM-18 project. Please pull the image from Docker Hub as described below. The LLVM is installed under the :code:`/root/llvm-project` folder in the docker image, and a prebuilt hcl-mlir dialect is installed in :code:`/root/hcl-dialect`.

.. code-block:: console

  $ docker pull chhzh123/hcl-dialect:llvm-18.x-py3.12
  $ docker run --rm -it chhzh123/hcl-dialect:llvm-18.x-py3.12
  (docker) $ git clone https://github.com/cornell-zhang/allo.git && cd allo
  (docker) $ python3 -m pip install -e .

Please note that the hcl-mlir dialect is not up-to-date. You can pull the latest hcl-mlir dialect and rebuild it if needed.

.. code-block:: console

  (docker) $ cd /root/hcl-dialect
  (docker) $ git remote update && git fetch
  (docker) $ cd build && make -j8
  (docker) $ cd tools/hcl/python_packages/hcl_core
  (docker) $ python3 -m pip install -e .

Install from Source
-------------------

Please clone the Allo repository to your local machine.

.. code-block:: console
  
  $ git clone https://github.com/cornell-zhang/allo.git
  $ cd allo

We recommend creating a new conda environment for Allo. Since we are using the latest Python features, the minimum Python version is **3.12**.

.. code-block:: console

  $ conda create -n allo python=3.12
  $ conda activate allo

After creating the Python environment, you can install the required packages by running the following command.

.. code-block:: console

  $ bash build.sh

You should see "Installation completed!" if the installation is finished.

Testing
-------

To make sure the installation is successful, you can run the following command to test the Allo package.

.. code-block:: console

  $ python3 -m pytest tests/

Internal Installation
---------------------
For Zhang Group students, we have already prepared a prebuilt version of LLVM on our server, so you do not need to build everything from source. Please follow the instruction below to set up the environment.

Make sure you have the access to the :code:`brg-zhang` or other Zhang group server. You can log into the server by SSH or use VSCode Remote SSH extension. Please refer to `this website <https://code.visualstudio.com/docs/remote/ssh>`_ for more details on configuring VSCode. Those servers are only accessible from the campus network. Please use the VPN if you are off-campus.

After logging into the server, the first step is to install an Anaconda environment. We recommend you to install your own `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, which is a lightweight version of Anaconda and contains only the necessary packages. You can download the installer from the link above and install it on your system. After the installation, you can create a new environment for Allo by running the following commands:

.. code-block:: console

  $ conda create -n allo python=3.12
  $ conda activate allo

We also provide a script to set up the backend LLVM compiler. You can copy the script to your home directory and run it

.. code-block:: console

  $ cp /work/shared/common/allo/setup-py312.sh ~/
  $ source ~/setup.sh

.. note::

  You can also add this line to your :code:`~/.bashrc` file so that you don't need to run the setup script every time.

Then, you can pull the latest version of Allo from GitHub and install it by running

.. code-block:: console

  $ git clone https://github.com/cornell-zhang/allo.git
  $ cd allo
  $ python3 -m pip install -e .

Now, you can run the following command to test if the installation is successful

.. code-block:: console

  $ python3 -c "import allo as allo; import allo.ir as air"

If you see no error message, then the installation is successful. Otherwise, please contact us for help.
