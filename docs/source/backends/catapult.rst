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
Siemens Catapult HLS (FPGA)
############################

The `Catapult HLS <https://eda.sw.siemens.com/en-US/ic/catapult-high-level-synthesis/>`_ backend enables Allo to generate hardware accelerators using Siemens' high-level synthesis toolchain. Catapult HLS uses Algorithmic C (AC) data types (``ac_int``, ``ac_fixed``, ``ac_channel``) and provides industry-leading quality of results for ASIC and FPGA designs.

Prerequisites
-------------
To use the Catapult HLS backend, you need:

1. **Siemens Catapult HLS** installed and licensed
2. **MGC_HOME** environment variable set to your Catapult installation directory
3. **AC Datatypes** headers available (typically included with Catapult)

.. code-block:: bash

   # Example environment setup
   export MGC_HOME=/path/to/catapult
   export PATH=$MGC_HOME/bin:$PATH

Kernel Definition
-----------------
Define your kernel using Allo's Python-embedded DSL. Here's an example of a vector addition kernel:

.. code-block:: python

   import allo
   from allo.ir.types import int32

   def vvadd(a: int32[100], b: int32[100]) -> int32[100]:
       c: int32[100]
       for i in range(100):
           c[i] = a[i] + b[i]
       return c

   s = allo.customize(vvadd)

Code Generation for Catapult HLS
--------------------------------
Allo supports two modes for Catapult HLS:

1. **C Simulation (csim)**:
   Compiles the generated C++ code with g++ and runs functional simulation. This mode is useful for verifying the correctness of your design before synthesis.

   .. code-block:: python

      import numpy as np

      mod = s.build(target="catapult", mode="csim", project="vvadd.prj")

      # Prepare test data
      np_a = np.random.randint(0, 100, size=(100,)).astype(np.int32)
      np_b = np.random.randint(0, 100, size=(100,)).astype(np.int32)
      np_c = np.zeros((100,), dtype=np.int32)

      # Run simulation
      mod(np_a, np_b, np_c)

      # Verify results
      np.testing.assert_array_equal(np_c, np_a + np_b)

2. **C Synthesis (csyn)**:
   Runs Catapult HLS synthesis to generate RTL. This mode invokes the Catapult tool to synthesize your design.

   .. code-block:: python

      mod = s.build(target="catapult", mode="csyn", project="vvadd.prj")

      # Run synthesis (no arguments needed for csyn mode)
      mod()

Generated Code Features
-----------------------
The Catapult backend generates C++ code with Catapult-specific features:

**AC Datatypes**

Allo automatically maps data types to Catapult's AC datatypes:

- Integer types map to ``ac_int<W, S>`` for non-standard widths
- Fixed-point types map to ``ac_fixed<W, I, S>``
- Streams map to ``ac_channel<T>``

.. code-block:: cpp

   // Generated headers
   #include <ac_int.h>
   #include <ac_fixed.h>
   #include <ac_channel.h>

**Catapult Pragmas**

Allo's scheduling primitives are translated to Catapult-specific pragmas:

.. code-block:: python

   s = allo.customize(kernel)
   s.pipeline("i")      # Generates: #pragma hls_pipeline_init_interval 1
   s.unroll("j")        # Generates: #pragma hls_unroll
   s.unroll("k", 4)     # Generates: #pragma hls_unroll 4

Project Structure
-----------------
The generated project (e.g., ``vvadd.prj``) includes:

- **kernel.cpp**: The synthesizable kernel code with AC datatypes
- **kernel.h**: Header file for the kernel interface
- **host.cpp**: Host code for C simulation (csim mode only)
- **run.tcl**: TCL script for Catapult synthesis
- **Makefile**: Build scripts for the project

Example: Matrix Multiplication
------------------------------
Here's a complete example of matrix multiplication with Catapult HLS:

.. code-block:: python

   import allo
   from allo.ir.types import int32
   import numpy as np

   def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
       C: int32[32, 32] = 0
       for i, j, k in allo.grid(32, 32, 32):
           C[i, j] += A[i, k] * B[k, j]
       return C

   s = allo.customize(gemm)

   # Apply optimizations
   s.pipeline("j")
   s.unroll("k", 4)

   # Build for Catapult
   with tempfile.TemporaryDirectory() as tmpdir:
       mod = s.build(target="catapult", mode="csim", project=tmpdir)

       # Test the design
       np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
       np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
       np_C = np.zeros((32, 32), dtype=np.int32)

       mod(np_A, np_B, np_C)
       np.testing.assert_array_equal(np_C, np.matmul(np_A, np_B))

Configuration Options
---------------------
You can customize the synthesis through the ``configs`` dictionary:

.. code-block:: python

   mod = s.build(
       target="catapult",
       mode="csyn",
       project="gemm.prj",
       configs={
           "frequency": 500,  # Target frequency in MHz (default: 300)
       },
   )

The frequency setting affects the clock period constraint in the generated TCL script.

Comparison with Vitis HLS
-------------------------
While both Catapult and Vitis HLS are high-level synthesis tools, they have different characteristics:

.. list-table::
   :header-rows: 1

   * - Feature
     - Catapult HLS
     - Vitis HLS
   * - Data Types
     - AC datatypes (``ac_int``, ``ac_fixed``)
     - AP datatypes (``ap_int``, ``ap_fixed``)
   * - Streams
     - ``ac_channel<T>``
     - ``hls::stream<T>``
   * - Pipeline Pragma
     - ``#pragma hls_pipeline_init_interval``
     - ``#pragma HLS pipeline``
   * - Unroll Pragma
     - ``#pragma hls_unroll``
     - ``#pragma HLS unroll``
   * - Vendor
     - Siemens
     - AMD/Xilinx

Troubleshooting
---------------

**MGC_HOME not set**

If you see an error about MGC_HOME not being set, ensure the environment variable points to your Catapult installation:

.. code-block:: bash

   export MGC_HOME=/path/to/catapult

**AC headers not found**

If compilation fails due to missing AC headers, verify that the AC datatypes are installed:

.. code-block:: bash

   ls $MGC_HOME/shared/include/ac_int.h

**Synthesis fails**

Check the Catapult log files in your project directory for detailed error messages. Common issues include:

- Unsupported C++ constructs
- Memory access patterns that cannot be synthesized
- Timing constraints that cannot be met

Conclusion
----------
The Catapult HLS backend provides an alternative synthesis path for Allo designs, leveraging Siemens' industry-leading HLS technology. It supports both functional simulation (csim) and RTL synthesis (csyn), making it suitable for designs targeting both ASIC and FPGA implementations.
