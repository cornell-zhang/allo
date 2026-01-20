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
   from allo.ir.types import int32
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

2. **HLS Emulation, Synthesis, and Execution**:  
   Users may also specify the target as `"xlscc"` but run the designs in our software emulation mode:

   - ``sw_emu``: Software emulation mode, which is similar to C simulation that compiles the program using C compiler 
   and runs it on the CPU. Depending on the size of your input data, this mode may take within one minute.

    .. code-block:: python

       mod = s.build(target="xlscc", mode="sw_emu", project="gemm.prj")

Project Structure and Execution
-------------------------------
The generated XLS project (e.g., in the folder ``gemm.prj``) typically includes:

- **test_block.cpp**: The accelerator kernel code. 

If you set `use_memory` to `True`, the generated project will also include:

- **rewrites.textproto**: The memory configuration file that is required by the XLS 
backend to map any memory arrays to the hardware. Without this file or if this file is
configured wrong, the XLS backend will error out during its IR optimization pass.

If you are using the software emulation mode, you will also obtain the following files:

- **test_harness.cpp**: The host-side (CPU) code that invokes the accelerator.
- **Makefile**: Build scripts to compile the project.
- **input*.data**: The binary input data files for the design.

To run the design, prepare the input matrices using NumPy and invoke the generated module to obtain the result:

.. code-block:: python

   np_A = np.random.random((M, K)).astype(np.int32)
   np_B = np.random.random((K, N)).astype(np.int32)
   result = mod(np_A, np_B)
   np.testing.assert_allclose(result, np.matmul(np_A, np_B), rtol=1e-5, atol=1e-5)
   
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
   MemUram = Memory(resource="URAM", storage_type="RAM_1P")
   MemBram = Memory(resource="BRAM", storage_type="RAM_2P")

   def kernel(a: int32[32] @ MemUram, b: int32[16, 16] @ MemBram) -> int32[32]:
       # Local variable with memory annotation
       buf: int32[32] @ Memory(resource="BRAM")
       for i in range(32):
           buf[i] = a[i] * 2
       c: int32[32]
       for i in range(32):
           c[i] = buf[i] + 1
       return c

   s = allo.customize(kernel)
   mod = s.build(target="xls", use_memory=True)
   print(mod.hls_code)

This generates a rewrite textproto file that maps buffer ``a`` (``RAM_1P``) to the XLS ``RAM_1RW`` type and input buffer ``b`` (``RAM_2P``) to the XLS ``RAM_1R1W`` type:

**Memory Class Parameters**
The Allo ``Memory`` class accepts the following parameters:

- **resource** (str): Memory resource type

  - ``"BRAM"``: Block RAM - the most common on-chip memory
  - ``"URAM"``: Ultra RAM - larger capacity, available on UltraScale+ devices
  - ``"LUTRAM"``: LUT-based RAM - faster but smaller
  - ``"SRL"``: Shift Register LUT - efficient for FIFOs
  - ``"AUTO"``: Let the HLS tool decide (default)

These all default to RAM_ABSTRACT in XLS.

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

XLS currently only supports RAM_1RW and RAM_1R1W, so we can only support RAM_1P, RAM_2P, and ROM_1P. The
other storage types are currently not supported.

Conclusion
----------
This example illustrates the process of defining a GEMM kernel using the Allo ADL and generating XLS code 
for the ASIC flow. The approach supports the software synthesis modes which is wonderful for quick verifications and debugging.
