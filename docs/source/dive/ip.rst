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

Suppose the C++ kernel header is defined in the ``vadd.h`` file:

.. code-block:: cpp

    #ifndef VADD_H
    #define VADD_H

    void vadd(int A[32], int B[32], int C[32]);

    #endif // VADD_H

And the corresponding implementation is defined in the ``vadd.cpp`` file:

.. code-block:: cpp

    #include "vadd.h"
    using namespace std;

    void vadd(int A[32], int B[32], int C[32]) {
        for (int i = 0; i < 32; ++i) {
            C[i] = A[i] + B[i];
        }
    }

In Allo, we can create an *IP module* to wrap the C++ kernel. Basically, we need to provide the top-level function name, the header files, and the implementation files. Also, currently an Allo signature is required to specify the input and output types of the kernel. Allo will automatically compile the C++ kernel and generate the corresponding Python wrapper based on the provided files and signature. The last argument ``link_hls`` determines whether the C++ compiler should link the Vitis HLS libraries (e.g., ``ap_int``), which is only available when your machine has installed Vitis HLS.

.. code-block:: python

    vadd = allo.IPModule(
        top="vadd",
        headers=["vadd.h"],
        impls=["vadd.cpp"],
        signature=["int32[32]", "int32[32]", "int32[32]"],
        link_hls=False,
    )

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
