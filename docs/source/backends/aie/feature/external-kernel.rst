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

User-Defined External Kernels
-----------------------------

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
