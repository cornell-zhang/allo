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

Trace-based Profiling
---------------------

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
   def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
       @df.kernel(mapping=[P0, P1], args=[A, B, C])
       def gemm(local_A: TyI[M, K] @ LyA, local_B: TyI[K, N] @ LyB, local_C: TyO[M, N] @ LyC):
           local_C[:, :] = allo.matmul(local_A, local_B)

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
between the vectorized (``allo/library/aie/kernels/conv_small_vector.cc``) and
scalar (``allo/library/aie/kernels/conv_small_scalar.cc``) versions of the kernel is not
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
