..  Copyright HeteroCL authors. All Rights Reserved.
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

To install and use HeteroCL, we need to have the LLVM compiler infrastructure installed on our system.
We have already prepared a prebuilt version of LLVM on our server, so you can directly export it to your system path.
Please follow the instruction below to set up the environment.

Make sure you have the access to the :code:`brg-zhang` or :code:`zhang-21` server. You can log into the server by
SSH or use VSCode Remote SSH extension. Please refer to `this website <https://code.visualstudio.com/docs/remote/ssh>`_
for more details on configuring VSCode. Those servers are only accessible from the campus network.

After logging into the server, the first step is to install an Anaconda environment. We recommend you to install your own
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, which is a lightweight version of Anaconda and contains
only the necessary packages. You can download the installer from the link above and install it on your system.
After the installation, you can create a new environment and install the required packages by running the following commands:

.. code-block:: console

  $ conda create -n hcl python=3.8
  $ conda activate hcl
  $ python -m pip install -r /work/shared/common/heterocl/requirements.txt

We also provide a script to set up the backend LLVM compiler. You can copy the script to your home directory by running

.. code-block:: console

  $ cp /work/shared/common/heterocl/setup.sh ~/

Then, you can pull the latest version of HeteroCL from GitHub by running

.. code-block:: console

  $ git clone https://github.com/chhzh123/heterocl.git
  $ cd heterocl
  $ git checkout parser

Maintaining your own copy of HeteroCL is important since later on you will probably need to change the source code.
To expose HeteroCL to your environment, please change the **last line** of the setup script :code:`~/setup.sh` to

.. code-block:: bash

  export PYTHONPATH=$PYTHONPATH:~/heterocl

Then, you can run the setup script to set up the environment

.. code-block:: console

  $ source ~/setup.sh

.. note::

  You can also add this line to your :code:`~/.bashrc` file so that you don't need to run the setup script every time.

Now, you can run the following command to test if the installation is successful

.. code-block:: console

  $ python -c "import heterocl as hcl; hcl.init(hcl.Int())"

If you see no error message, then the installation is successful. Otherwise, please contact us for help.


Trouble Shooting
----------------
If you encounter the following error message when running the test command, please check if you have set up the environment correctly.

.. code-block:: console

  >>> import heterocl.ir
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ModuleNotFoundError: No module named 'heterocl.ir'

To be more specific, you can check if your :code:`PYTHONPATH` environment variable **ONLY** contains the repository path of your
own HeteroCL (by typing ``echo $PYTHONPATH`` in the terminal).
Otherwise, you can ``unset PYTHONPATH`` to remove the environment variable and run the setup script again.
