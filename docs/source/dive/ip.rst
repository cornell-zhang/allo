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

##############
IP Integration
##############

Apart from directly writing Allo kernels in Python, we also support integrating existing C++ HLS kernels into Allo. This feature is useful when you have a existing optimized C++ HLS code that wants to be integrated into Allo. The following example shows how to integrate a simple vector addition kernel written in C++ into Allo.

Suppose the C++ kernel is implemented in the ``vadd.cpp`` file:

.. code-block:: cpp

    void vadd(int A[32], int B[32], int C[32]) {
        for (int i = 0; i < 32; ++i) {
            C[i] = A[i] + B[i];
        }
    }

In Allo, we can create an *IP module* to wrap the C++ kernel. Basically, we need to provide the top-level function name and the implementation file. Allo will automatically parse the C++ kernel signature, compile the kernel, and generate the corresponding Python wrapper based on the provided files. Multi-dimensional arrays and pointers are supported in this C++ function definition. The last argument ``link_hls`` of the ``IPModule`` determines whether the C++ compiler should link the Vitis HLS libraries (e.g., ``ap_int``), which is only available when your machine has installed Vitis HLS.

.. code-block:: python

    vadd = allo.IPModule(top="vadd", impl="vadd.cpp", link_hls=False)

After creating the IP module, we can use it in Allo as a normal Python function. For example, we can directly call the ``vadd`` function to perform vector addition. The inputs and outputs will be automatically wrapped and unwrapped as NumPy arrays, which greatly simplies the burden of complex C-Python interface management. This is also very useful when you want to debug the HLS kernels with the Python data.

.. code-block:: python

    np_A = np.random.randint(0, 100, (32,)).astype(np.int32)
    np_B = np.random.randint(0, 100, (32,)).astype(np.int32)
    np_C = np.zeros((32,), dtype=np.int32)
    vadd(np_A, np_B, np_C)
    np.testing.assert_allclose(np_A + np_B, np_C, atol=1e-6)

Moreover, the IP module can also be called in a normal Allo kernel. In the following example, we wrap the ``vadd`` function into an Allo ``kernel`` and use it to perform vector addition. The Allo kernel can then be further customized and compiled with the external C++ HLS kernel.

.. code-block:: python

    def kernel(A: int32[32], B: int32[32]) -> int32[32]:
        C: int32[32] = 0
        vadd(A, B, C)
        return C

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(0, 100, (32,)).astype(np.int32)
    np_B = np.random.randint(0, 100, (32,)).astype(np.int32)
    allo_C = mod(np_A, np_B)
    np.testing.assert_allclose(np_A + np_B, allo_C, atol=1e-6)

Similar to other Allo kernels, we can also change the build target to invoke the HLS compiler to generate the hardware design with the external C++ kernel. Make sure you have installed Vitis environment before running the following code.

.. code-block:: python

    mod = s.build(target="vitis_hls", mode="csyn", project="vadd.prj")
    mod()

Supported data types
====================

The element types allowed in the C++ IP signature are the standard C arithmetic types (``int``, ``float``, ``double``, and the fixed-width ``stdint`` types such as ``int8_t``/``uint16_t``), as well as the arbitrary-precision HLS integer types ``ap_int<N>`` and ``ap_uint<N>``. Because ``ap_int``/``ap_uint`` are provided by the Vitis HLS headers, an IP module that uses them must be created with ``link_hls=True``.

At the Python boundary, ``ap_int<N>`` and ``ap_uint<N>`` are mapped to the matching fixed-width integer type (e.g., ``ap_int<8>`` behaves like ``int8_t`` and ``ap_uint<16>`` like ``uint16_t``), so callers simply pass NumPy arrays of the corresponding ``dtype``. For example, given the following kernel in ``vadd_ap_int.cpp``:

.. code-block:: cpp

    #include <ap_int.h>
    void vadd_ap_int(ap_int<8> A[32], ap_int<8> B[32], ap_int<16> C[32]) {
        for (int i = 0; i < 32; ++i)
            C[i] = A[i] + B[i];
    }

it can be wrapped and invoked with NumPy arrays whose dtypes match the ``ap_int`` widths:

.. code-block:: python

    vadd_ap_int = allo.IPModule(top="vadd_ap_int", impl="vadd_ap_int.cpp", link_hls=True)
    np_A = np.random.randint(-64, 64, (32,)).astype(np.int8)
    np_B = np.random.randint(-64, 64, (32,)).astype(np.int8)
    np_C = np.zeros((32,), dtype=np.int16)
    vadd_ap_int(np_A, np_B, np_C)
    np.testing.assert_allclose(np_A.astype(np.int16) + np_B.astype(np.int16), np_C, atol=1e-6)
