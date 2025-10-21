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

.. toctree::
   :maxdepth: 1
   :caption: Setup

   env.rst


.. toctree::
   :maxdepth: 1
   :caption: Usage

   use.rst

.. toctree::
   :maxdepth: 2
   :caption: Additional Features

   feature/index
