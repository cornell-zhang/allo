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

To install and use Allo, we need to have the LLVM compiler infrastructure installed on our system.
We have already prepared a prebuilt version of LLVM on our server, so you can directly export it to your system path.
Please follow the instruction below to set up the environment.
If you want to install Allo on your local machine, you can refer to the [README](https://github.com/cornell-zhang/allo) guideline in the Allo repository.

Make sure you have the access to the :code:`brg-zhang` or other Zhang group server. You can log into the server by
SSH or use VSCode Remote SSH extension. Please refer to `this website <https://code.visualstudio.com/docs/remote/ssh>`_
for more details on configuring VSCode. Those servers are only accessible from the campus network.

After logging into the server, the first step is to install an Anaconda environment. We recommend you to install your own
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, which is a lightweight version of Anaconda and contains
only the necessary packages. You can download the installer from the link above and install it on your system.
After the installation, you can create a new environment for Allo by running the following commands:

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
