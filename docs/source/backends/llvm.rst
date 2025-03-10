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

##################
LLVM (CPU)
##################

The CPU backend leverages the `LLVM <https://llvm.org/>`_ infrastructure to compile high-level kernels into LLVM IR for CPU simulation. This example demonstrates the workflow of defining a matrix multiplication (GEMM) kernel in Allo, compiling it into an LLVM executable, and validating the results with NumPy. For more details on kernel customizations and scheduling optimizations, please refer to the `Allo-CPU tutorial <https://cornell-zhang.github.io/allo/gallery/tutorial_01_get_started.html>`_.

Kernel Definition
-----------------
In this section, we define the GEMM kernel. The kernel uses strict type annotations to ensure all variables are of type `int32`. The iteration space is constructed with the `allo.grid` API, which simplifies the creation of nested loops for matrix multiplication.

.. code-block:: python

   import allo
   from allo.ir.types import int32

   def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
       C: int32[32, 32] = 0
       for i, j, k in allo.grid(32, 32, 32):
           C[i, j] += A[i, k] * B[k, j]
       return C

Compilation to LLVM
-------------------
After defining the kernel, the next step is to compile it for the CPU. The Allo framework creates a schedule for the kernel with `allo.customize` and then compiles the kernel into an LLVM module using the `build` method. This process converts the high-level description into LLVM IR, which is used for CPU simulation.

.. code-block:: python

   s = allo.customize(gemm)
   mod = s.build(target="llvm")

Execution and Data Preparation
------------------------------
Once the LLVM module is built, input data must be prepared for execution. In this example, two random 32x32 matrices are generated using NumPy. The data is explicitly cast to `int32` to match the kernel's requirements, ensuring consistency in the computation.

.. code-block:: python

   import numpy as np

   np_A = np.random.randint(0, 100, (32, 32)).astype(np.int32)
   np_B = np.random.randint(0, 100, (32, 32)).astype(np.int32)
   allo_C = mod(np_A, np_B)

Testing
-------
To verify the correctness of the compiled kernel, the output produced by the LLVM module is compared with the result of NumPy's matrix multiplication. The comparison uses NumPy's testing utilities to ensure that the computed result meets the desired numerical precision.

.. code-block:: python

   golden_C = np.matmul(np_A, np_B)
   np.testing.assert_allclose(allo_C, golden_C, rtol=1e-5, atol=1e-5)
   print("Results are correct!")

Conclusion
----------
This example illustrates the process of defining a numerical kernel using the Allo DSL, compiling it with the LLVM backend for CPU simulation. Notice as Allo does not optimize the CPU code, the CPU backend is purely for functional correctness checking but *not* for performance evaluation.
