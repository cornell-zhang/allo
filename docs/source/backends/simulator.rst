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

##############################
Multi-Threaded Simulator (CPU)
##############################

This document demonstrates how to simulate a dataflow design using the simulator backend in Allo. The simulator backend provides a fast and flexible environment for verifying the behavior of dataflow kernels before deploying them to hardware. In this example, a simple producer-consumer model is implemented, where data is produced from an input matrix, sent through a pipe, and then consumed with a basic arithmetic operation.

Dataflow Kernel Definition
--------------------------
The design consists of a top-level region that contains two kernels: a producer and a consumer. The producer reads data from an input matrix and sends each element through a pipe, while the consumer receives the data, increments it by one, and writes the result to an output matrix. The following code illustrates the kernel definitions using Allo's dataflow API:

.. code-block:: python

   import allo
   from allo.ir.types import float32, Stream
   import allo.dataflow as df
   import numpy as np

   Ty = float32
   M, N, K = 16, 16, 16

   @df.region()
   def top():
       # Create a pipe with a depth of 4
       pipe: Stream[Ty, 4]

       @df.kernel(mapping=[1])
       def producer(A: Ty[M, N]):
           for i, j in allo.grid(M, N):
               # Load data from the input matrix
               out: Ty = A[i, j]
               # Send data to the pipe
               pipe.put(out)

       @df.kernel(mapping=[1])
       def consumer(B: Ty[M, N]):
           for i, j in allo.grid(M, N):
               # Receive data from the pipe
               data = pipe.get()
               # Perform a simple computation (increment by 1)
               B[i, j] = data + 1

Simulation and Testing
-----------------------
To verify the correctness of the dataflow design, a simulation is executed using the simulator backend. The test function initializes an input matrix with random values and an output matrix filled with zeros. The simulation is performed by building the module with the target set to `"simulator"`. After running the simulation, the output is compared against the expected result using NumPy's testing utilities.

.. code-block:: python

    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)
    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B)
    np.testing.assert_allclose(B, A + 1, rtol=1e-5, atol=1e-5)
    print("Dataflow Simulator Passed!")

The simulator is implemented using the `OMP dialect <https://mlir.llvm.org/docs/Dialects/OpenMPDialect/>`_ in MLIR, so it can natively support multi-threaded execution on CPU, which greatly speeds up functional testing at the first place.

Controlling OpenMP Threads in Simulator Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running simulations or tests that use the simulator backend, the number of threads created by OpenMP can be controlled via the ``OMP_NUM_THREADS`` environment variable. This variable controls the number of CPU threads OpenMP uses for parallel execution.

Some dataflow designs may require a large number of threads to simulate properly (otherwise, you might experience "hang" or "stuck"). In such cases, please set ``OMP_NUM_THREADS`` to a larger value.

For example:

.. code-block:: bash

    export OMP_NUM_THREADS=64
    python3 tests/dataflow/test_1D_systolic.py


The OpenMP runtime determines the size of its thread pool during initialization based on the value of ``OMP_NUM_THREADS``.  
Once the thread pool has been created, modifying ``OMP_NUM_THREADS`` afterward will **not** affect the existing pool size.  

**Please ensure that the thread pool is large enough when running the simulator.**

You can either export ``OMP_NUM_THREADS`` to a sufficiently large value before running all simulators,  
or adjust it individually within each process as needed.  

**Tip:** If you are using ``pytest`` to run multiple tests and have not pre-exported a large enough ``OMP_NUM_THREADS``,  
be sure to use the ``--forked`` option so that each test runs in a separate process and the ``OMP_NUM_THREADS`` setting in ``setup_env`` takes effect.


Example: To ensure that each test runs in a separate process and the thread pool is recreated according to ``OMP_NUM_THREADS``, you can add the following code to each test file in the target folder:

.. code-block:: python

    import os
    import pytest

    @pytest.fixture(scope="module", autouse=True)
    def setup_env():
        os.environ["OMP_NUM_THREADS"] = "128"
        yield
        del os.environ["OMP_NUM_THREADS"]

    def test_xxx():
        # your test code here
        pass

Then run the tests with:

.. code-block:: bash

    python3 -m pytest --forked /path/to/tests/folder -v
