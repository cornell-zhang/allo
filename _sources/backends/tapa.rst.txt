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

###############################
RapidStream TAPA (FPGA)
###############################

`RapidStream TAPA <https://github.com/rapidstream-org/rapidstream-tapa>`_ is an open-source backend that enables agile synthesis for designing high-frequency FPGA dataflow accelerators. Please refer to their `TRETS papers <https://doi.org/10.1145/3593025>`_ for more details.
It still requires the `Vitis HLS <https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis/vitis-hls.html>`_ toolchain to generate hardware accelerators for FPGA devices, so please set up the environment in advance. This document demonstrates how to define a general matrix multiplication (GEMM) kernel using the Allo ADL and generate TAPA code for FPGA synthesis.

Kernel Definition
-----------------
The GEMM kernel is implemented using `float32` precision and pre-defined constants for the matrix dimensions. The kernel utilizes the `allo.grid` API to iterate over output indices and the `allo.reduction` API to designate the reduction axis for accumulating the dot-product.

.. code-block:: python

   import allo
   from allo.ir.types import float32
   import numpy as np

   # Define matrix dimensions
   M, N, K = 32, 32, 32

   def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
       C: int32[32, 32] = 0
       for i, j, k in allo.grid(32, 32, 32):
           C[i, j] += A[i, k] * B[k, j]
       return C

Code Generation for RapidStream TAPA
-------------------------------------
Allo supports the generation of TAPA code by setting the target to `"tapa"`. Four synthesis modes are available:

- ``csim``: C simulation mode, which compiles the design for software simulation.
- ``fast_hw_emu``: Fast hardware emulation mode developed by RapidStream, offering a rapid emulation of the hardware design.
- ``hw_emu``: Hardware emulation mode, synthesizing the design into an RTL model and simulating it on an FPGA emulator.
- ``hw``: Full hardware synthesis mode, performing complete synthesis, placement, routing, and bitstream generation for execution on FPGA hardware.

.. code-block:: python

   s = allo.customize(gemm)
   mod = s.build(target="tapa", mode="hw_emu", project="gemm.prj")
   print(mod)

Project Structure and Execution
-------------------------------
The generated TAPA project (e.g., in the folder ``gemm.prj``) typically includes:

- **host.cpp**: The host (CPU) code that interfaces with the generated accelerator.
- **kernel.cpp**: The accelerator kernel code generated for TAPA.
- **Makefile**: Build scripts that streamline the project compilation.

To execute the design, prepare the input matrices using NumPy. For instance, generate random matrices for the inputs and allocate an output array:

.. code-block:: python

   np_A = np.random.random((M, K)).astype(np.float32)
   np_B = np.random.random((K, N)).astype(np.float32)
   allo_C = np.zeros((M, N), dtype=np.float32)
   mod(np_A, np_B, allo_C)
   np.testing.assert_allclose(allo_C, np.matmul(np_A, np_B), rtol=1e-5, atol=1e-5)

Note:
  Ensure that the TAPA and required toolchain environments are correctly configured before running the flow. For further environment setup and detailed information on synthesis modes, please consult the `RapidStream TAPA <https://tapa.readthedocs.io/en/main/>`_ documentation.


HBM/DDR Memory Mapping
----------------------
Similar to the Vitis HLS backend, TAPA also supports HBM/DDR memory channel mapping through the ``hbm_mapping`` configuration option. This allows you to specify which memory channels each kernel argument should be mapped to.

.. code-block:: python

   # Define HBM channel mapping using argument names
   hbm_mapping = {
       "A": 0,              # Input A -> HBM channel 0
       "B": "HBM[1]",       # Input B -> HBM channel 1
       "output_0": "DDR[0]", # Return value -> DDR bank 0
   }

   mod = s.build(
       target="tapa",
       mode="hw",
       project="gemm.prj",
       configs={"hbm_mapping": hbm_mapping},
   )

For more details on HBM/DDR memory mapping options, please refer to the :doc:`Vitis HLS documentation <vitis>`.
