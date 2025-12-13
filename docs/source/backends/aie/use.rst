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

Getting Started with MLIR-AIE
=============================

To enable the experimental MLIR-AIE codegen, specify ``"aie"`` as the target
in the ``dataflow.build`` function.

Currently, the supported target platforms include ``XDNA1`` and ``XDNA2``.
By default, the target platform is set to ``XDNA1``.  
To switch to ``XDNA2``, please run:

.. code-block:: bash

   export NPU2=1  


Example
-------

Vector addition
~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   import allo
   from allo.ir.types import int32
   import allo.dataflow as df
   import numpy as np

   def _test_vector_scalar_add():
       # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
       Ty = int32
       M = 1024

       @df.region()
       def top():
           @df.kernel(mapping=[1])
           def core(A: Ty[M], B: Ty[M]):
               B[:] = allo.add(A, 1)

       A = np.random.randint(0, 100, M).astype(np.int32)
       if "MLIR_AIE_INSTALL_DIR" in os.environ:
           mod = df.build(top, target="aie")
           B = np.zeros(M).astype(np.int32)
           mod(A, B)
           np.testing.assert_allclose(B, A + 1)
           print("PASSED!")
       else:
           print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


Matrix multiplication
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import allo
   from allo.ir.types import int32
   import allo.dataflow as df
   import numpy as np
   from allo.memory import Layout

   LyA = Layout("S0R")
   LyB = Layout("RS1")
   LyC = Layout("S0S1")


   def _test_gemm_1D():
       Ty = int32
       M, N, K = 16, 16, 16
       P0 = 2

       @df.region()
       def top():
           @df.kernel(mapping=[P0])
           def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N], C: Ty[M, N] @ LyA):
               C[:, :] = allo.matmul(A, B)

       mod = df.build(top, target="aie")
       A = np.random.randint(0, 64, (M, K)).astype(np.int32)
       B = np.random.randint(0, 64, (K, N)).astype(np.int32)
       C = np.zeros((M, N)).astype(np.int32)
       mod(A, B, C)
       np.testing.assert_allclose(C, A @ B, atol=1e-5)
       print("PASSED!")


Producer-consumer
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   import allo
   from allo.ir.types import int32, Stream
   import allo.dataflow as df
   import numpy as np

   Ty = int32
   M, N, K = 16, 16, 16


   @df.region()
   def top():
       pipe: Stream[Ty, 4]

       @df.kernel(mapping=[1])
       def producer(A: Ty[M, N]):
           for i, j in allo.grid(M, N):
               # load data
               out: Ty = A[i, j]
               # send data
               pipe.put(out)

       @df.kernel(mapping=[1])
       def consumer(B: Ty[M, N]):
           for i, j in allo.grid(M, N):
               # receive data
               data = pipe.get()
               # computation
               B[i, j] = data + 1


   def test_producer_consumer():
       A = np.random.randint(0, 64, (M, K)).astype(np.int32)
       B = np.zeros((M, N), dtype=np.int32)

       if "MLIR_AIE_INSTALL_DIR" in os.environ:
           mod = df.build(top, target="aie")
           mod(A, B)
           np.testing.assert_allclose(A + 1, B, atol=1e-5)
           print("Passed!")
       else:
           print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

Learning Materials
==================

- `IRON AIE Programming Guide <https://github.com/Xilinx/mlir-aie/tree/main/programming_guide>`_
- `MLIR-AIE Programming Examples <https://github.com/Xilinx/mlir-aie/tree/main/programming_examples>`_
- `MLIR-based AI Engine Design Tutorial <https://github.com/Xilinx/mlir-aie/tree/main/tutorial>`_
- `Riallto - an exploration framework for the AMD Ryzen AI NPU <https://riallto.ai/index.html>`_
