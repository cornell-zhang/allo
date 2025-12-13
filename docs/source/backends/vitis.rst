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

############################
AMD Vitis HLS (FPGA)
############################

The `Vitis HLS <https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis/vitis-hls.html>`_ FPGA backend leverages the Vivado/Vitis HLS toolchain to generate hardware accelerators for FPGA devices. This document demonstrates how to define a general matrix multiplication (GEMM) kernel using the Allo ADL and generate HLS code for FPGA synthesis. For more details on kernel customizations and scheduling optimizations, please refer to the `Allo-HLS tutorial <https://cornell-zhang.github.io/allo/gallery/tutorial_02_vhls.html>`_.

Kernel Definition
-----------------
The GEMM kernel is implemented with `float32` precision using pre-defined constants for the matrix dimensions. The kernel utilizes the `allo.grid` API to iterate over output indices and the `allo.reduction` API to designate the reduction axis for accumulating the dot-product.

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

Code Generation for Vivado/Vitis HLS
------------------------------------
Allo supports several approaches to generate HLS code:

1. **Direct HLS Code Generation**:  
   Set the target to `"vhls"` to produce HLS code as a string. This code includes the necessary headers and pragmas for Vitis HLS synthesis.

    .. code-block:: python

       code = s.build(target="vhls")
       print(code)

2. **HLS Emulation, Synthesis, and Execution**:  
   Specify the target as `"vitis_hls"` along with a synthesis mode and project name to generate a complete HLS project. The supported modes are:

   - ``sw_emu``: Software emulation mode, which is similar to C simulation that compiles the program using C compiler and runs it on the CPU. Depending on the size of your input data, this mode may take within one minute.
   - ``hw_emu``: Hardware emulation mode, which is similar to co-simulation that compiles the program into RTL design using HLS compiler and runs the RTL with the test bench on the FPGA emulator. Since it needs to go through the HLS synthesis flow, it may take several minutes to finish.
   - ``hw``: Full hardware synthesis mode, which compiles the program into RTL design using HLS, goes through placement and routing, generates the bitstream, and finally executes on FPGA. This mode may take several hours to finish.

    .. code-block:: python

       mod = s.build(target="vitis_hls", mode="hw_emu", project="gemm.prj")

3. **(Legacy) HLS Synthesis**:  
   Set the target to `"vivado_hls"` to generate a legacy HLS project. This approach is similar to the previous one but uses ``run.tcl`` for the project script. The supported modes are:

   - ``csim``: C simulation mode, using the gcc compiler to compile the program and runs it on the CPU.
   - ``csyn``: C synthesis mode, using Vivado HLS compiler to synthesize the program. **Note: This mode only synthesize the program and generate the RTL design but does not execute the program!**
   - ``cosim``: Co-simulation mode, using Vivado HLS compiler to synthesize the program and generate the RTL design, then runs the RTL with the test bench on the CPU.
   - ``impl``: Implementation mode, using Vivado HLS compiler to synthesize the program, generate the RTL design, and go through placement and routing to the bitstream.

   .. code-block:: python

      mod = s.build(target="vivado_hls", mode="csim", project="gemm.prj")
      # For csim
      mod(np_A, np_B, allo_C)
      # For csyn
      mod()


Project Structure and Execution
-------------------------------
The generated HLS project (e.g., in the folder ``gemm.prj``) typically includes:

- **host.cpp**: The host-side (CPU) code that invokes the accelerator.
- **kernel.cpp**: The accelerator kernel code.
- **Makefile**: Build scripts to compile the project.

To run the design, prepare the input matrices using NumPy and allocate an output array for the result:

.. code-block:: python

   np_A = np.random.random((M, K)).astype(np.float32)
   np_B = np.random.random((K, N)).astype(np.float32)
   allo_C = np.zeros((M, N), dtype=np.float32)
   mod(np_A, np_B, allo_C)
   np.testing.assert_allclose(allo_C, np.matmul(np_A, np_B), rtol=1e-5, atol=1e-5)

Note:
  Ensure that the Vitis HLS and XRT environments are correctly configured before running the HLS flow. For further environment setup and synthesis mode details, please consult the `Vitis HLS <https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis/vitis-hls.html>`_ documentation.


HBM/DDR Memory Mapping
----------------------
For designs targeting platforms with High Bandwidth Memory (HBM) or multiple DDR banks (such as Alveo U280), Allo provides an easy way to specify memory channel mappings for kernel arguments. This is done through the ``hbm_mapping`` configuration option in the ``configs`` dictionary.

**Basic Usage**

The ``hbm_mapping`` dictionary maps function argument names to memory channels. You can specify:

- **Integer values**: Interpreted as HBM channel numbers (e.g., ``0`` → ``HBM[0]``)
- **String values**: Full memory specification (e.g., ``"HBM[0]"``, ``"DDR[1]"``)

.. code-block:: python

   from allo.ir.types import int32

   def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
       C: int32[32, 32] = 0
       for i, j, k in allo.grid(32, 32, 32):
           C[i, j] += A[i, k] * B[k, j]
       return C

   s = allo.customize(gemm)

   # Define HBM channel mapping using argument names
   hbm_mapping = {
       "A": 0,              # Input A -> HBM channel 0
       "B": "HBM[1]",       # Input B -> HBM channel 1
       "output_0": "DDR[0]", # Return value -> DDR bank 0
   }

   mod = s.build(
       target="vitis_hls",
       mode="hw",
       project="gemm.prj",
       configs={"hbm_mapping": hbm_mapping},
   )

**Argument Naming Convention**

- **Input arguments**: Use the same names as in your function definition (e.g., ``"A"``, ``"B"``)
- **Return values**: Use ``"output_0"``, ``"output_1"``, etc. for functions with return values

**Generated Configuration File**

Allo automatically generates a ``.cfg`` file in the project directory with the connectivity settings. For example, the above configuration would generate a file like:

.. code-block:: text

   [connectivity]

   sp=gemm.v15:HBM[0]
   sp=gemm.v16:HBM[1]
   sp=gemm.v17:DDR[0]

Note that Allo automatically translates your user-friendly argument names (``A``, ``B``, ``output_0``) to the actual HLS-generated argument names (``v15``, ``v16``, ``v17``).

**Complex Example with Multiple Arguments**

For larger designs with many memory interfaces, you can specify different HBM channels for each argument:

.. code-block:: python

   hbm_mapping = {
       "inp_0": 0,           # HBM[0]
       "inp_1": 0,           # HBM[0] - can share channels
       "weight_0": 1,        # HBM[1]
       "weight_1": 2,        # HBM[2]
       "bias": "HBM[3]",     # HBM[3]
       "output_0": "DDR[0]", # DDR bank 0
   }

This feature is particularly useful for:

- **Memory bandwidth optimization**: Distribute data across multiple HBM channels to maximize bandwidth
- **Bank conflict avoidance**: Place frequently accessed data on separate memory banks
- **Platform-specific tuning**: Match memory assignments to your target FPGA platform's memory architecture


Device and Frequency Configuration
----------------------------------
You can specify the target device and clock frequency through the ``configs`` dictionary:

.. code-block:: python

   mod = s.build(
       target="vitis_hls",
       mode="hw",
       project="gemm.prj",
       configs={
           "device": "u280",     # Target device (default: "u280")
           "frequency": 300,     # Target frequency in MHz (default: 300)
       },
   )

**Supported Devices**

- **Alveo**: ``u200``, ``u250``, ``u280``
- **Zynq UltraScale+**: ``zcu102``, ``zcu104``, ``zcu106``, ``zcu111``
- **Versal**: ``vck190``, ``vhk158``
- **Embedded**: ``ultra96v2``, ``pynqz2``, ``zedboard``


Conclusion
----------
This example illustrates the process of defining a GEMM kernel using the Allo ADL and generating HLS code for FPGA acceleration with the Vitis HLS backend. The approach supports various synthesis modes (sw_emu, hw_emu, hw) to cater to different design and verification needs.
