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

.. _developer:

###############
Developer Setup
###############

Depending on which part of Allo you want to contribute to, you may set up the environment differently.

Developer Installation
----------------------

If you only want to change the frontend part of Allo, you can install the Allo package following the `general guide <../setup/index.html>`_.

If you need to change the backend part of Allo, you need to install the LLVM-18 project and make sure your LLVM commit is the same as `the one we referred to <https://github.com/cornell-zhang/hcl-dialect/tree/main/externals>`_.

.. code-block:: bash

    mkdir -p build && cd build
    # Python 3.12 is required
    cmake -G "Unix Makefiles" ../llvm \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_INSTALL_UTILS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE=`which python3`
    make -j8

.. note::

    For Zhang Group students, you can run the following commands to use the pre-installed LLVM-18 project.

    .. code-block:: bash
        export LLVM_HOME=/work/shared/users/common/llvm-project-18.x
        export PATH=$LLVM_HOME/build-patch/bin:$PATH
        export PATH=/work/shared/users/common/cmake-3.27.9/bin/:$PATH
        export LLVM_BUILD_DIR=$LLVM_HOME/build-patch

After installing the LLVM-18 project, you can build the hcl-mlir dialect from source. Need to export ``$LLVM_BUILD_DIR`` to system path first.

.. code-block:: bash

    git clone https://github.com/cornell-zhang/hcl-dialect.git
    cd hcl-dialect
    mkdir -p build && cd build
    cmake -G "Unix Makefiles" .. \
        -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
        -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
        -DPYTHON_BINDING=ON \
        -DOPENSCOP=OFF \
        -DPython3_EXECUTABLE=`which python3` \
        -DCMAKE_CXX_FLAGS="-Wfatal-errors -std=c++17"
    make -j8
    cd tools/hcl/python_packages/hcl_core
    python3 -m pip install -e .

Every time you change the hcl-mlir dialect, you need to run ``make`` again to make the changes effective. Finally, you can install the Allo package:

.. code-block:: bash

    python3 -m pip install -e .

It will connect the frontend with the backend.

Upstream Changes
----------------

It would be good to maintain your own `fork <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_ of
the Allo repository, which helps create a `PR (pull request) <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_ and upstream changes easier.

To create your own fork, click on the "Fork" button on the top right of the `Allo repository <https://github.com/cornell-zhang/allo>`_.
This will create a copy of the repository under your own GitHub account.

Next, you can clone your fork to your local machine OR if you have already cloned the Allo repository, you can add your fork as a remote:

.. code-block:: bash

    # Change OWNER to your GitHub username
    $ git remote add origin https://github.com/OWNER/allo.git
    # Update upstream
    $ git remote add upstream https://github.com/cornell-zhang/allo.git

And then you can verify the remotes:

.. code-block:: bash

    $ git remote -v
    > origin  https://github.com/OWNER/allo.git (fetch)
    > origin  https://github.com/OWNER/allo.git (push)
    > upstream https://github.com/cornell-zhang/allo.git (fetch)
    > upstream https://github.com/cornell-zhang/allo.git (push)

Basically, the development workflow is as follows:

1. Create a new branch from the ``main`` branch (``git checkout -b <branch_name> main``)
2. Make changes to the code
3. Commit the changes to your *local* branch (``git commit -m "commit message"``)
4. Push the changes to your *fork* (``git push origin <branch_name>``)
5. Make sure your changes do not break the existing facilities, see `Integration Tests <#integration-tests>`_ for more details
6. Create a `PR <https://github.com/cornell-zhang/allo/pulls>`_ from your fork to the ``main`` branch of the Allo repository (see `here <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork>`_ for more details)
7. Wait for the PR to be reviewed
8. If there are any changes requested, make the changes and push to your fork
9. Once the PR is approved, it will be merged into the ``main`` branch

Most of the ``git`` features are integrated in VSCode, please refer to the `document <https://code.visualstudio.com/docs/sourcecontrol/intro-to-git>`_ if you want to use the GUI.

Integration Tests
-----------------
We have configured `GitHub Actions <https://github.com/features/actions>`_ to run the tests on every PR, so please
make sure you have passed the tests locally before creating a PR or pushing to your fork.

There are several tests for our project:

1. **License header check**: make sure all the source files have the license header, which is important for open source projects.
2. **Code style check**: make sure the code style is consistent with the `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ standard. We use ``black`` for Python formatting, which should has been installed in your environment during the setup.
3. **Linting check**: make sure the code is `linted <https://www.perforce.com/blog/qac/what-lint-code-and-what-linting-and-why-linting-important>`_ correctly. We use ``pylint`` for Python linting, which should also been installed during the setup.
4. **Unit tests**: make sure the changes will not break the existing facilities. We use ``pytest`` for Python unit tests, and the test cases are under the ``tests`` folder.

If you are making changes to the code, please make sure to run those tests before pushing to your fork.
To run the tests, you can run the following command from the root of the repository:

.. code-block:: bash

    $ bash .circleci/task_lint.sh

Following is an example output:

.. code-block::

    Check license header...
    all checks passed...
    all checks passed...
    Check Python formats using black...
    ./scripts/lint/git-black.sh: line 31: warning: setlocale: LC_ALL: cannot change locale (C.UTF-8): No such file or directory
    ./scripts/lint/git-black.sh: line 31: warning: setlocale: LC_ALL: cannot change locale (C.UTF-8)
    Version Information: black, 24.8.0 (compiled: yes)
    Python (CPython) 3.12.0
    Read returned 0
    Files: allo/ir/types.py
    Running black in checking mode
    All done! ✨ 🍰 ✨
    1 file would be left unchanged.
    ./scripts/lint/git-black.sh: line 31: warning: setlocale: LC_ALL: cannot change locale (C.UTF-8)    : No such file or directory
    ./scripts/lint/git-black.sh: line 31: warning: setlocale: LC_ALL: cannot change locale (C.UTF-8)
    Version Information: black, 24.8.0 (compiled: yes)
    Python (CPython) 3.12.0
    Read returned 0
    Files: allo/ir/types.py
    Running black in checking mode
    All done! ✨ 🍰 ✨
    1 file would be left unchanged.
    Running pylint on allo

    --------------------------------------------------------------------
    Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)

Lastly run the unit tests:

.. code-block:: bash

    $ python3 -m pytest tests

If no error is reported, hurrah, you are good to go!