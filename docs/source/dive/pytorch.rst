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

###################
PyTorch Integration
###################

In this document, we will show how to directly compile PyTorch models to Allo.
First, users can define a PyTorch module as usual:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x, y):
            x = x + y
            x = F.relu(x)
            return x

    model = Model()
    model.eval()

Then, users can compile the PyTorch model to Allo by using the ``allo.frontend.from_pytorch`` API:

.. code-block:: python

    import allo
    example_inputs = [torch.rand(1, 3, 10, 10), torch.rand(1, 3, 10, 10)]
    llvm_mod = allo.frontend.from_pytorch(model, example_inputs=example_inputs)

Then, we can use the generated Allo LLVM module as usual by passing in the NumPy inputs:

.. code-block:: python

    golden = model(*example_inputs)
    np_inputs = [x.detach().numpy() for x in example_inputs]
    res = llvm_mod(*np_inputs)
    torch.testing.assert_close(res, golden.detach().numpy())
    print("Passed!")

The process should be very similar to the original Allo workflow.
The default target is LLVM. We can also change the backend to other compilers such as Vitis HLS by specifying the ``target``:

.. code-block:: python

    mod = allo.frontend.from_pytorch(model, example_inputs=example_inputs, target="vhls")
    print(mod.hls_code)
