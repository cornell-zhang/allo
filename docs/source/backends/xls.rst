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
Google XLS (ASIC)
############################

The `Google XLS <https://google.github.io/xls/>`_ ASIC backend leverages the Google Accelerated Hardware Synthesis toolchain to generate 
hardware accelerators for the ASIC flow. This document demonstrates how to define a general matrix multiplication (GEMM) 
kernel using the Allo ADL and generate HLS code for place and route. For more details on kernel customizations and scheduling optimizations, 
please refer to the `Allo-HLS tutorial <https://cornell-zhang.github.io/allo/gallery/tutorial_02_vhls.html>`_.

Kernel Definition
-----------------
The GEMM kernel is implemented with `int32` precision using pre-defined constants for the matrix dimensions. The kernel utilizes the 
`allo.grid` API to iterate over output indices and the `allo.reduction` API to designate the reduction axis for accumulating the 
dot-product.

.. code-block:: python

   import allo
   from allo.ir.types import float32
   import numpy as np

   # Define matrix dimensions
   M, N, K = 4, 4, 4

   def gemm(A: int32[4, 4], B: int32[4, 4]) -> int32[4, 4]:
       C: int32[4, 4] = 0
       for i, j, k in allo.grid(4, 4, 4):
           C[i, j] += A[i, k] * B[k, j]
       return C

Code Generation for Google XLS
------------------------------------
Allo supports several approaches to generate HLS code:

1. **Direct HLS Code Generation**:  
   Set the target to `"xlscc"` to produce XLS C++ frontend code as a string. This code includes the necessary headers and pragmas for 
   Google XLS synthesis. We expose a `use_memory` flag to control whether to use the memory API in the XLS backend.

    .. code-block:: python

       code = s.build(target="xlscc", use_memory=True)
       print(code)

   When `use_memory` is set to `True`, the generated code will use the memory API in the XLS backend. When `use_memory` is set to `False`, 
   the generated code will use the C-style arrays which become registers clusters in the XLS backend.

2. **Software Emulation, Synthesis, and Execution**:  
   We also provide a way to emulate the generated code in software. This software emulation mode is similar to C simulation that compiles the program 
   using C compiler and runs it on the CPU. Specify the target as `"xlscc"` along with the `sw_emu` synthesis mode and project name to generate a 
   complete HLS project. Depending on the size of your input data, this mode may take some time to build and run.

    .. code-block:: python

       mod = s.build(target="xlscc", mode="sw_emu", project="gemm.prj")

Project Structure and Execution
-------------------------------
The generated XLS project (e.g., in the folder ``gemm.prj``) typically includes:

- **test_block.cpp**: The accelerator kernel code. 

If you set `use_memory` to `True`, the generated project will also include:

- **rewrites.textproto**: The memory configuration file that is required by the XLS backend to map any memory arrays to the hardware.

When in software emulation mode, the above items will be generated along with:

- **test_harness.cpp**: The host-side (CPU) code that invokes the accelerator.
- **input*.dat**: The input data files that are used to test the accelerator.
- **Makefile**: Build scripts to compile the project.

To run the design in software emulation mode, prepare the input matrices using NumPy:

.. code-block:: python

   # Prepare input matrices using NumPy (must use int32 for XLS backend)
   np_A = np.random.randint(0, 10, size=(M, K)).astype(np.int32)
   np_B = np.random.randint(0, 10, size=(K, N)).astype(np.int32)

   # Run the sw_emu and get the result
   allo_C = mod(np_A, np_B)

   # Verify the result
   expected_C = np.matmul(np_A, np_B).astype(np.int32)
   np.testing.assert_array_equal(allo_C, expected_C)
   print("sw_emu verification passed!")

Note:
  The XLS backend only supports integer and fixed-point types. Floating-point types (f16, f32, f64) are not supported.

Memory Implementation Customization
------------------------------------
Allo provides the ``Memory`` class to specify on-chip memory implementation details for arrays, similar to the `Vitis HLS bind_storage pragma <https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-bind_storage>`_. This allows fine-grained control over how arrays are mapped to FPGA memory resources (BRAM, URAM, LUTRAM, etc.).

**Basic Usage**

Use the ``@`` operator to annotate function arguments or local variables with memory specifications:

.. code-block:: python

   from allo import Memory
   from allo.ir.types import int32, float32

   # Define memory specifications
   MemUram = Memory(resource="URAM")
   MemBram = Memory(resource="BRAM", storage_type="RAM_2P")

   def kernel(a: int32[32] @ MemUram, b: float32[16, 16] @ MemBram) -> int32[32]:
       # Local variable with memory annotation
       buf: int32[32] @ Memory(resource="BRAM")
       for i in range(32):
           buf[i] = a[i] * 2
       c: int32[32]
       for i in range(32):
           c[i] = buf[i] + 1
       return c

   s = allo.customize(kernel)
   mod = s.build(target="vhls")
   print(mod.hls_code)

This generates HLS code with ``bind_storage`` pragmas:

.. code-block:: cpp

   void kernel(int32_t v0[32], float v1[16][16], int32_t v2[32]) {
     #pragma HLS bind_storage variable=v0 impl=uram
     #pragma HLS bind_storage variable=v1 type=ram_2p impl=bram

     int32_t buf[32];
     #pragma HLS bind_storage variable=buf impl=bram
     // ... kernel body ...
   }

**Memory Class Parameters**

The ``Memory`` class accepts the following parameters:

- **resource** (str): Memory resource type

  - ``"BRAM"``: Block RAM - the most common on-chip memory
  - ``"URAM"``: Ultra RAM - larger capacity, available on UltraScale+ devices
  - ``"LUTRAM"``: LUT-based RAM - faster but smaller
  - ``"SRL"``: Shift Register LUT - efficient for FIFOs
  - ``"AUTO"``: Let the HLS tool decide (default)

- **storage_type** (str, optional): RAM access pattern

  - ``"RAM_1P"``: Single-port RAM
  - ``"RAM_2P"``: Simple dual-port RAM (one read, one write port)
  - ``"RAM_T2P"``: True dual-port RAM (two read/write ports)
  - ``"RAM_1WNR"``: Single write, N read ports
  - ``"RAM_S2P"``: Simple dual-port (alias for RAM_2P)
  - ``"ROM_1P"``: Single-port ROM
  - ``"ROM_2P"``: Dual-port ROM
  - ``"ROM_NP"``: N-port ROM

- **latency** (int, optional): Memory access latency in cycles
- **depth** (int, optional): Depth of the memory (useful for streams/FIFOs)

**Examples**

1. **URAM for large buffers**:

   .. code-block:: python

      # URAM is ideal for large arrays on UltraScale+ FPGAs
      LargeBuffer = Memory(resource="URAM")

      def process(data: float32[1024, 1024] @ LargeBuffer):
          ...

2. **Dual-port BRAM for concurrent access**:

   .. code-block:: python

      # RAM_2P allows simultaneous read and write
      DualPort = Memory(resource="BRAM", storage_type="RAM_2P")

      def pipeline(inp: int32[256] @ DualPort) -> int32[256]:
          ...

3. **LUTRAM for small, fast buffers**:

   .. code-block:: python

      # LUTRAM is faster but uses more LUTs
      FastBuffer = Memory(resource="LUTRAM")

      def compute(weights: int8[64] @ FastBuffer):
          ...

4. **Multiple memory types in one kernel**:

   .. code-block:: python

      InputMem = Memory(resource="BRAM", storage_type="RAM_1P")
      WeightMem = Memory(resource="URAM")
      OutputMem = Memory(resource="BRAM", storage_type="RAM_2P")

      def neural_layer(
          inp: float32[128] @ InputMem,
          weights: float32[128, 64] @ WeightMem,
          out: float32[64] @ OutputMem
      ):
          ...

**Best Practices**

- Use **URAM** for large arrays (>36Kb) on UltraScale+ devices to save BRAM resources
- Use **BRAM with RAM_2P** when you need concurrent read/write access
- Use **LUTRAM** for small lookup tables that require low latency
- Use **ROM** types for constant data that never changes
- Let the tool decide (``resource="AUTO"``) when you don't have specific requirements


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
