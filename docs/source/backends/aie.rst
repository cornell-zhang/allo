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

################################
AMD MLIR-AIE (AI Engine)
################################

`AMD AI Engine (AIE) <https://www.amd.com/en/products/adaptive-socs-and-fpgas/technologies/ai-engine.html>`_ is a hardware accelerator specifically designed for AI applications. It is a TPU-like architecture with an array of AIE cores.
In this document, we target the `AMD Ryzen 7040 <https://www.amd.com/en/products/processors/laptop/ryzen-for-business.html>`_ and 8040 Series processors that are built on `AMD XDNA NPUs <https://www.amd.com/en/technologies/xdna.html>`_, which are equipped with the second-generation AI Engine cores.

.. image:: https://riallto.ai/notebooks/images/png/ryzenai_array_5x4.png
   :alt: AMD Ryzen AI Engine Array
   :align: center

Environment Setup
=================

Prerequisites
-------------

Before proceeding with the Allo installation, please follow the instructions on the `MLIR-AIE website <https://github.com/Xilinx/mlir-aie/tree/main?tab=readme-ov-file#getting-started-for-amd-ryzen-ai---linux-quick-setup-instructions>`_ to install the required `Vitis <https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis.html>`_ and XRT environment. Stop when you reach the "Install IRON for AMD Ryzen™ AI AIE Application" section as we need a separate process to install MLIR-AIE under the Allo environment.


Install from Source
-------------------

Please follow the general instructions in :ref:`Install from Source <install-from-source>` to install the LLVM-19 project and the Allo package. In the following, we suppose you have already installed the LLVM-19 project and enable the ``allo`` conda environment.

Below are the exact commands to set up the environment:

Step 1
~~~~~~
Activate the ``allo`` conda environment

   .. code-block:: bash

      conda activate allo

Step 2
~~~~~~
We depend on the `MLIR-AIE <https://github.com/Xilinx/mlir-aie>`_ project to compile the Allo IR to AIE. Install release 1.0

   .. code-block:: bash

      # Install IRON library and mlir-aie from a wheel
      python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.0
      # Install Peano from a llvm-aie wheel
      python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl


.. warning::

   The ``mlir_aie`` wheel requires ``manylinux_2_35``, and some systems (e.g., those with glibc 2.34, confirmed by ``ldd --version``) do not meet this requirement.  
   This results in an installation failure such as:

   ``ERROR: mlir_aie-0.0.1.2025042204+24208c0-cp312-cp312-manylinux_2_35_x86_64.whl is not a supported wheel on this platform.``

Step 3
~~~~~~
Clone the mlir-aie repository and checkout to the commit corresponding to release 1.0

   .. code-block:: bash

      git clone https://github.com/Xilinx/mlir-aie.git
      cd mlir-aie
      git checkout 07320d6

Then, install python requirements, setup environment and add tools to PATHs (under ``mlir-aie``)

   .. code-block:: bash

      # Install basic Python requirements 
      python3 -m pip install -r python/requirements.txt
      # Install the pre-commit hooks defined in .pre-commit-config.yaml
      pre-commit install
      # Install MLIR Python Extras 
      HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
      # Install Torch for ML examples
      python3 -m pip install -r python/requirements_ml.txt

      source utils/env_setup.sh

.. _step4:

Step 4
~~~~~~
Clone the allo repository and install.

   You may want to set up environment variables first to use a custom CMake and LLVM build. For example:

   .. code-block:: bash

      export PATH=/opt/cmake-3.31.5-linux-x86_64/bin:/opt/llvm-project-19.x/build/bin:$PATH
      export LLVM_BUILD_DIR=/opt/llvm-project-19.x/build

   Then clone the allo repository and install by running the following commands

   .. code-block:: bash

      git clone https://github.com/cornell-zhang/allo.git
      cd allo
      python3 -m pip install -v -e .

.. note::

   See :ref:`internal_install` for Zhang Group students.

.. _step5:

Step 5
~~~~~~
Setup Vitis and XRT.

.. note::

   See :ref:`internal_install` for Zhang Group students.

Lastly, you can verify the AIE backend by running the following command under the ``allo`` folder.

.. code-block:: console

    python3 tests/dataflow/aie/test_vector.py


Patches and Configuration
-------------------------

To use components from the `MLIR-AIE toolchain <https://github.com/Xilinx/mlir-aie>`_ as libraries:

.. note::

   The instructions below are based on `MLIR-AIE release v1.0 <https://github.com/Xilinx/mlir-aie/releases/tag/v1.0>`_, which corresponds to commit `07320d6 <https://github.com/Xilinx/mlir-aie/tree/07320d6831b17e4a4c436d48c3301a17c1e9f1cd>`_.
   For compatibility, make sure to use this commit when copying the following components.

Clone and checkout the specific commit:

.. code-block:: bash

   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie
   git checkout 07320d6

- To use `external kernels <https://github.com/Xilinx/mlir-aie/tree/07320d6831b17e4a4c436d48c3301a17c1e9f1cd/aie_kernels>`_ as an AIE kernel library:

  .. code-block:: bash

     export MLIR_AIE_EXTERNAL_KERNEL_DIR=/your/copied/path/aie_kernels

- To use `runtime_lib <https://github.com/Xilinx/mlir-aie/tree/07320d6831b17e4a4c436d48c3301a17c1e9f1cd/runtime_lib>`_ for the host:

  .. code-block:: bash

     export RUNTIME_LIB_DIR=/your/copied/path/runtime_lib

If you run into issues when using ``aiecc.py`` such as:

.. code-block:: text

   error: expected ')' at end of argument list
   declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
                                                             ^

You can fix this by modifying ``downgrade_ir_for_peano`` in:

.. code-block:: text

   $MLIR_AIE_INSTALL_DIR/python/aie/compiler/aiecc/main.py

Update the function as follows:

**Before:**

.. code-block:: python

   def downgrade_ir_for_peano(llvmir):
       llvmir = llvmir.replace("getelementptr inbounds nuw", "getelementptr inbounds")
       return llvmir

**After:**

.. code-block:: python

   def downgrade_ir_for_peano(llvmir):
       llvmir = llvmir.replace("getelementptr inbounds nuw", "getelementptr inbounds")
       llvmir = llvmir.replace("captures(none)", "")
       return llvmir

.. _internal_install:

Internal Installation (Cornell)
-------------------------------

For Zhang Group students, please set up environment variables in :ref:`step4` with the following commands.

.. code-block:: console

      export PATH=/opt/cmake-3.31.5-linux-x86_64/bin:/opt/llvm-project-19.x/build/bin:$PATH  
      export LLVM_BUILD_DIR=/opt/llvm-project-19.x/build

And set up Vitis and XRT in :ref:`step5`  by running the following commands.

.. code-block:: console

      source /opt/common/setupVitis.sh
      source /opt/common/setupXRT.sh


Lastly, to verify the installation, you can run the following command:

.. code-block:: console

      python3 tests/dataflow/aie/test_vector.py

If the unit tests pass, then the installation is successful. Otherwise, please contact us for help.

Usage
=====

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
   from allo.memory import Layout

   Ly = Layout("S0")

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
   from allo.ir.types import int32
   import allo.dataflow as df
   import numpy as np

   Ty = int32
   M, N, K = 16, 16, 16


   @df.region()
   def top():
       pipe = df.pipe(dtype=Ty, shape=(), depth=4)

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


New Feature
===========

Profiling
---------

A new timing-based profiling feature has been added to help measure the
performance of the module during execution.

To enable profiling, use the ``profile`` flag in the ``build`` method in
``allo/dataflow.py``:

.. code-block:: python

   def build(
       func,
       target="vitis_hls",
       mode="csim",
       project="top.prj",
       configs=None,
       wrap_io=True,
       opt_default=True,
       enable_tensor=False,
       mapping_primitives: list[tuple[str, list]] = [],
       profile=False,
       warmup=20,
       num_iters=100,
       trace: list[tuple[str, tuple[int, ...]]] = None,
       trace_size: int = 4096,
       device_type: str = None,
   )

**Related Parameters:**

- ``profile`` (``bool``): Set to ``True`` to enable profiling. When enabled, the
  module performs extra warm-up and test iterations.
- ``warmup`` (``int``): Number of initial iterations to warm up the system.
  These iterations are **excluded** from the timing measurements. Default is
  ``20``.
- ``num_iters`` (``int``): Number of timed iterations used to compute execution
  time. Default is ``100``.

Example
~~~~~~~

.. code-block:: python

   import allo
   from allo.ir.types import int16, int32, float32
   import allo.dataflow as df
   import numpy as np
   from allo.memory import Layout

   Ty = int16
   M, N, K = 128, 128, 32
   Pm, Pn, Pk = 4, 4, 1
   Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

   LyA = Layout("S1S2")
   LyB = Layout("S2S0")
   LyC = Layout("S1S0")

   @df.region()
   def top1():
       @df.kernel(mapping=[Pk, Pm, Pn])
       def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: int32[M, N] @ LyC):
           C[:, :] = allo.matmul(A, B)

   mod = df.build(
       top1,
       target="aie",
       profile=True,
       warmup=200,
       num_iters=1000,
   )
   A = np.random.randint(0, 32, (M, K)).astype(np.int16)
   B = np.random.randint(0, 32, (K, N)).astype(np.int16)
   C = np.zeros((M, N)).astype(np.int32)
   tmp_C = np.zeros((M, N)).astype(np.int32)
   mod(A, B, C)


Profiling with Trace
--------------------

AIEs are equipped with tracing hardware that provides a cycle-accurate view of
hardware events. This enables more precise profiling, especially for analyzing
the performance of computation on each compute tile (AIE) and the associated
data transfers.

However, configuring the trace unit can be complex. This new feature simplifies
the process, making trace-based profiling easier to use.

Trace-based profiling requires configuring the compute tile and routing the
trace data as packets through the shim tile to external memory. This places
additional pressure on the DMA ports of the shim tile, making it unsuitable for
large-scale computation tasks where DMA bandwidth is already a constrained
resource. As a result, trace support is currently provided mainly for small-
scale computations.

To use trace, users can configure the options in the ``build`` method in
``allo/dataflow.py``:

.. code-block:: python

   def build(
       func,
       target="vitis_hls",
       mode="csim",
       project="top.prj",
       configs=None,
       wrap_io=True,
       opt_default=True,
       enable_tensor=False,
       mapping_primitives: list[tuple[str, list]] = [],
       profile=False,
       warmup=20,
       num_iters=100,
       trace: list[tuple[str, tuple[int, ...]]] = None,
       trace_size: int = 4096,
       device_type: str = None,
   )

**Related Parameters:**

- ``trace``: a list of tiles from the ``allo.dataflow.kernel`` users wish to
  trace. Each element consists of the kernel’s name as a string and a tuple
  representing the tile index. This index does not necessarily correspond to the
  final physical compute tile index in the 2D AIE array. Tracing is enabled on a
  best-effort basis: if resources (DMA ports or buffer descriptors) are limited,
  tracing may not be applied to all specified tiles in the list.
- ``trace_size``: the size of the trace buffer. If a large amount of trace
  information is expected, users may increase this accordingly.

After ``build``, running the generated module produces a file named
``trace.txt`` under the ``project`` directory.

The ``trace.txt`` file should contain multiple lines of non-zero values. If all
entries are zero, first check whether the ``top.mlir`` file contains any
``aie.packet_flow`` operations:

- If not, it indicates that tracing for the specified tiles was skipped due to
  resource constraints.
- If such operations are present but entries in ``trace.txt`` are all zero,
  please submit a bug report.

Users can use multiple tools to parse the ``trace.txt`` and convert it into a
more human-readable format. Useful parsers are provided in the ``mlir-aie``
repository. For example,
:download:`parse_trace.py <https://github.com/Xilinx/mlir-aie/blob/v1.0/programming_examples/utils/parse_trace.py>`
parses it into a JSON file that can be viewed in
`Perfetto <http://ui.perfetto.dev>`_. See the
`trace parser README <https://github.com/Xilinx/mlir-aie/blob/v1.0/programming_examples/utils/README.md#trace-parser-parse_tracepy>`_
for details.

.. note::

   The unit of timing reported in Perfetto should be interpreted as cycle count.
   See `issue #2214 <https://github.com/Xilinx/mlir-aie/issues/2214>`_ for more
   information.

Example
~~~~~~~

Tracing tile ``(0, 0)`` of the ``allo.dataflow.kernel`` named ``gemm``.

.. code-block:: python

   TyI, TyO = int16, int32
   M, N, K = 32, 32, 32
   P0, P1 = 2, 4

   @df.region()
   def top():
       @df.kernel(mapping=[P0, P1])
       def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
           C[:, :] = allo.matmul(A, B)

   # trace tile (0, 0) of gemm df.kernel
   mod = df.build(
       top,
       target="aie",
       trace=[
           ("gemm", (0, 0)),
       ],
       trace_size=65536,
   )
   A = np.random.randint(0, 64, (M, K)).astype(np.int16)
   B = np.random.randint(0, 64, (K, N)).astype(np.int16)
   C = np.zeros((M, N)).astype(np.int32)
   mod(A, B, C)
   np_C = A.astype(np.int32) @ B.astype(np.int32)
   np.testing.assert_allclose(C, np_C, atol=1e-5)
   print("PASSED!")


Using Trace to Measure the Performance of External Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trace is useful for evaluating the performance of an external kernel running on
a single compute tile. This is especially important when profiling optimizations
such as vectorization of external kernels. The following example demonstrates
how to use trace profiling on some convolution kernels.

In this case, due to the relatively small computation scale, the difference
between the vectorized (``allo/library/aie/conv_small_vector.cc``) and
scalar (``allo/library/aie/conv_small_scalar.cc``) versions of the kernel is not
clearly observable using timing-based profiling. Instead, one can insert event
markers (``event0();`` and ``event1();``) directly into the external C++ code
and run the trace on the compute tile executing the external kernel. Sample code
is available in ``tests/dataflow/aie/test_trace_conv.py``.

Process the generated trace (in ``top.prj/trace.txt``) with
:download:`parse_trace.py <https://github.com/Xilinx/mlir-aie/blob/v1.0/programming_examples/utils/parse_trace.py>`:

.. code-block:: bash

   # sample processing cmds
   cd top.prj
   path/to/parse_trace.py --filename trace.txt --mlir top.mlir --colshift 1 > trace_scalar.json

Use `Perfetto <http://ui.perfetto.dev>`_ to view the timeline.

- Scalar version:

  .. image:: https://github.com/user-attachments/assets/4cc92e2b-4b4c-495d-8718-0c5d32d22c00
     :width: 80%
     :alt: scalar

- Vector version:

  .. image:: https://github.com/user-attachments/assets/4c5b558d-c84d-4c16-aef2-3c626b62bbee
     :width: 80%
     :alt: vector

From the timeline screenshot, you can observe a clear difference in the
computation cycle count between the two kernels within the regions marked by the
event markers. Additionally, you can see that the vectorized version makes use
of vector instructions, which are absent in the scalar version.

If you need more precise cycle counts or additional profiling information, you
can write your own processing script to analyze the generated JSON file, or
directly parse the ``trace.txt``.


Support for User-Defined External Kernels
-----------------------------------------

Originally, complex computations on AIE cores were implemented using a limited
set of `external kernels provided in the mlir-aie repository
<https://github.com/Xilinx/mlir-aie/tree/v1.0/aie_kernels>`_. However, this
external kernel library supports only a narrow range of operations and leaves
room for performance improvement. To address these limitations, support has been
added for user-defined external kernels.

Users can now register and invoke external kernels implemented in C++ and
exposed via ``extern "C"`` interfaces. These kernels can be written using the
AIE API and integrated into the programming model workflow.

Suppose the external kernel is implemented in the :file:`norm.cc` file:

.. code-block:: cpp

   #include <aie_api/aie.hpp>
   #include <stdint.h>
   #include <stdio.h>
   #include <stdlib.h>
   #include <type_traits>

   #define NOCPP

   #define EPS 1e-6f // epsilon

   template <typename T_in, typename T_out, const int SEQ_LEN, const int HIDDEN>
   void rms_norm_single_batch(T_in *input_tensor, T_in *weight,
                              T_out *output_tensor) {
     constexpr int vec_factor = 16;
     using vec_t = aie::vector<T_in, vec_factor>;
     event0();
     for (int iter = 0; iter < SEQ_LEN; iter++) {
       T_in *__restrict input_ptr = input_tensor;
       T_in *__restrict weight_ptr = weight;
       T_out *__restrict output_ptr = output_tensor;
       float square_sum = 0.0f;
       const int F = HIDDEN / vec_factor;
       for (int i = 0; i < F; i++) {
         vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
         input_ptr += vec_factor;
         vec_t square_vec = aie::mul(input_vec, input_vec);
         square_sum += aie::reduce_add(square_vec);
       }
       vec_t square_sum_vec =
           aie::broadcast<T_in, vec_factor>(square_sum / HIDDEN + EPS);
       vec_t rms = aie::invsqrt(square_sum_vec);
       input_ptr = input_tensor;
       for (int i = 0; i < F; i++) {
         vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
         input_ptr += vec_factor;
         vec_t normed = aie::mul(input_vec, rms);
         vec_t weight_vec = aie::load_v<vec_factor>(weight_ptr);
         weight_ptr += vec_factor;
         vec_t result = aie::mul(normed, weight_vec);
         aie::store_v(output_ptr, result);
         output_ptr += vec_factor;
       }
       input_tensor += HIDDEN;
       output_tensor += HIDDEN;
     }
     event1();
   }

and exposed via ``extern "C"`` interfaces:

.. code-block:: cpp

   extern "C" {
     void layer_norm(float A_in[4][512], float B_in[512], float C_out[4][512]) {
       rms_norm_single_batch<float, float, 4, 512>(&A_in[0][0], B_in, &C_out[0][0]);
     }
   }

.. warning::

   External kernel function arguments must have fully specified constant
   shapes. Pointer types are not allowed.

Users can create an ``ExternalModule`` to wrap the
kernel and use it in computation on an AIE core.

Register the ``ExternalModule`` in the context:

.. code-block:: python

   norm = ExternalModule(
       top="layer_norm",       # Name of the top-level function defined with `extern "C"`
       impl_path="norm.cc",    # Path to the user-provided source file
       input_idx=[0, 1],       # Indices of input arguments
       output_idx=[2],         # Indices of output arguments
   )

The external module can then be used in an Allo kernel:

.. code-block:: python

   @df.kernel(mapping=[1])
   def core(A: Ty[M, N] @ LyA, B: Ty[N] @ Ly, C: Ty[M, N] @ LyA):
       norm(A, B, C)

An example can be found in ``tests/dataflow/aie/test_norm.py``.

Learning Materials
==================

- `IRON AIE Programming Guide <https://github.com/Xilinx/mlir-aie/tree/main/programming_guide>`_
- `MLIR-AIE Programming Examples <https://github.com/Xilinx/mlir-aie/tree/main/programming_examples>`_
- `MLIR-based AI Engine Design Tutorial <https://github.com/Xilinx/mlir-aie/tree/main/tutorial>`_
- `Riallto - an exploration framework for the AMD Ryzen AI NPU <https://riallto.ai/index.html>`_
