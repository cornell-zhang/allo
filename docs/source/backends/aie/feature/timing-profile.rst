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

Timing-based Profiling
----------------------

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

