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

#######################
Frontend Syntax Guide
#######################

This document provides a comprehensive reference for the Allo frontend syntax and semantics.
Allo uses a Python-based domain-specific language (DSL) that requires **strict type annotations**
to enable hardware synthesis and optimization.

.. contents:: Table of Contents
   :local:
   :depth: 2

Function Definition
===================

Basic Function Signature
------------------------

Allo kernels are defined as Python functions with explicit type annotations for all arguments and return types.
Arguments and return types can be scalar or tensor types. The syntax follows Python's type hint notation:

.. code-block:: python

    def kernel(arg1: Type1[Shape1], arg2: ScalarType) -> ReturnType[Shape]:
        # function body
        return result

**Example: Scalar Arguments**

.. code-block:: python

    def kernel(A: int32) -> int32:
        return A + 1

**Example: Matrix Multiplication**

.. code-block:: python

    from allo.ir.types import int32

    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32):
            C[i, j] += A[i, k] * B[k, j]
        return C

Multiple Return Values
----------------------

Functions can return multiple values as a tuple:

.. code-block:: python

    def kernel(A: int32[M], B: int32[M]) -> (int32[M], int32[M]):
        res0: int32[M] = 0
        res1: int32[M] = 0
        for i in range(M):
            res0[i] = A[i] + 1
            res1[i] = B[i] + 1
        return res0, res1

The caller can unpack the returned tuple:

.. code-block:: python

    C, D = callee(A[i], B[i])

To ignore certain return values, use underscore:

.. code-block:: python

    C, _ = callee(A[0], B[0])  # Ignore second return value


No Return Value
---------------

Functions that don't return a value can omit the return type annotation, use ``-> None``,
or have an empty return:

.. code-block:: python

    def kernel(A: int32[32]):
        pass  # No return

    def kernel(A: int32[32]) -> None:
        return

    def kernel(A: int32[32]):
        return None


Variable Declaration and Assignment
===================================

Scalar Variables
----------------

Scalar variables are declared using Python's type annotation syntax:

.. code-block:: python

    # Declaration with initialization
    x: int32 = 0
    y: float32 = 3.14

    # Declaration without initialization
    z: int32

    # Assignment after declaration
    z = x + y

Tensor Variables
----------------

Tensors are declared with their shape in the type annotation:

.. code-block:: python

    # 1D tensor
    A: int32[10] = 0

    # 2D tensor initialized to zero
    B: int32[32, 32] = 0

    # 4D tensor
    C: float32[M, M, M, M] = 0

Initialization from Lists and NumPy Arrays
------------------------------------------

Tensors can be initialized from Python lists or NumPy arrays:

.. code-block:: python

    # From nested list (compile-time constant)
    tmp: int32[2, 2] = [[1, 2], [3, 4]]

    # From NumPy array (global constant)
    arr = np.array([[1, 2], [3, 4]])
    def kernel() -> int32:
        tmp: int32[2, 2] = arr
        return tmp[0, 0]

    # Constant tensor slicing
    np_A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    def kernel() -> int32[4]:
        A: int32[4] = np_A[1]  # Load second row as constant
        return A

Copy Semantics
--------------

Tensors can be copied by assignment:

.. code-block:: python

    temp: int32[M, N] = 0
    outp: int32[M, N] = temp  # Copy temp to outp

    # Copy from argument
    def kernel(inp: int32[M, N]) -> int32[M, N]:
        outp: int32[M, N] = inp
        return outp


Loop Constructs
===============

Range Loops
-----------

Standard Python ``range`` loops are supported with one, two, or three arguments:

.. code-block:: python

    # range(end)
    for i in range(10):
        A[i] = i

    # range(start, end)
    for i in range(10, 20):
        A[i] = i

    # range(start, end, step)
    for i in range(0, 20, 2):
        A[i] = i * 2

.. note::

   ``break`` and ``continue`` are **not supported** in Allo.

Variable Loop Bounds
--------------------

Loop bounds can be runtime variables:

.. code-block:: python

    def kernel(A: int32[10]):
        for i in range(10):
            for j in range(i + 1, 10):  # Variable lower bound
                for k in range(j * 2, 10):  # Variable lower bound
                    A[k] += i - j

    # Bounds from array elements
    def kernel(A: int32[10], B: int32[10]):
        for i in range(10):
            for j in range(A[i], 10, A[i]):  # Bounds from array
                B[j] += i

Grid Loops
----------

``allo.grid`` provides a shorthand for nested loops:

.. code-block:: python

    # Equivalent to three nested for loops
    for i, j, k in allo.grid(32, 32, 32):
        C[i, j] += A[i, k] * B[k, j]

    # 2D grid
    for i, j in allo.grid(M, M):
        res[i, j] = C[i, j] + 1

Named grids are useful for applying schedule optimizations:

.. code-block:: python

    for i, j, k in allo.grid(32, 32, 32, name="C"):
        C[i, j] += A[i, k] * B[k, j]


While Loops
-----------

While loops with runtime conditions:

.. code-block:: python

    from allo.ir.types import index

    def kernel(A: int32[10]):
        i: index = 0
        while i < 10:
            A[i] = i
            i += 1


Conditional Statements
======================

If-Elif-Else
------------

Standard Python conditional syntax:

.. code-block:: python

    def kernel(a: int32, b: int32) -> int32:
        r: int32 = 0
        if a == 0:
            r = 1
        elif a == 1:
            r = 2
            if b == 2:  # Nested conditional
                r = 3
        else:
            r = 4
        return r

Logical Operators
-----------------

Conditions can use ``and``, ``or``, and ``not``:

.. code-block:: python

    if A[0] > 0 and b < 0:
        r = 1
    elif A[1] * 2 <= 1 or b + 1 >= 1:
        r = 2
    elif not flag:
        r = 3

Multiple conditions can be chained:

.. code-block:: python

    if A[0] > 0 and A[1] > 0 and A[2] > 0 and b > 0 and c > 0:
        r = 1

Select Expression (Ternary)
---------------------------

Python's ternary expression for conditional assignment:

.. code-block:: python

    B[i] = 1 if A[i] % 2 == 0 else 0

    # With type casting
    B[i] = (i * 2) if A[i] % 2 == 0 else 0


Operators
=========

Arithmetic Operators
--------------------

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
     - Example
   * - ``+``
     - Addition
     - ``a + b``
   * - ``-``
     - Subtraction
     - ``a - b``
   * - ``*``
     - Multiplication
     - ``a * b``
   * - ``/``
     - Division (float)
     - ``a / b``
   * - ``//``
     - Floor division
     - ``a // b``
   * - ``%``
     - Modulo
     - ``a % b``

Unary Operators
---------------

.. code-block:: python

    vi: int32 = -(v + 1)  # Negation
    result = +(vi + vf)   # Unary plus

Comparison Operators
--------------------

All standard comparison operators are supported: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``

Bitwise Operators
-----------------

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
     - Example
   * - ``<<``
     - Left shift
     - ``1 << v``
   * - ``>>``
     - Right shift
     - ``64 >> v``
   * - ``&``
     - Bitwise AND
     - ``1 & v``
   * - ``|``
     - Bitwise OR
     - ``1 | v``
   * - ``^``
     - Bitwise XOR
     - ``a ^ b``

Augmented Assignment
--------------------

All augmented assignment operators work on both scalars and tensor elements:

.. code-block:: python

    C[i, j] += A[i, k] * B[k, j]
    A[i] *= 2
    A[i] -= 1


Array and Tensor Operations
===========================

Indexing
--------

Standard multi-dimensional indexing:

.. code-block:: python

    value = A[i, j, k]
    A[i, j] = value

Subviews
--------

Accessing a sub-array by partial indexing:

.. code-block:: python

    def kernel(A: int32[10, 10]) -> int32[10]:
        return A[5]  # Returns row 5 as 1D array

    def kernel(A: float32[5, 10, 15]) -> float32[15]:
        return A[3, 2]  # Returns a 1D slice

Dynamic subviews with variable indices:

.. code-block:: python

    def kernel(A: float32[5, 10, 15], i: index, j: index) -> float32[15]:
        return A[i, j]

Slicing
-------

Sub-tensor assignment using slices:

.. code-block:: python

    def slice_copy(A: int32[6, 6]) -> int32[6, 6]:
        B: int32[2, 3] = 0
        B[0, 0] = 1
        A[0:2, 0:3] = B  # Copy B into a slice of A
        return A

Bit Operations on Integers
--------------------------

Access individual bits or bit ranges:

.. code-block:: python

    B[i] = A[i][0]      # Access bit 0
    B[i][0:2] = A[i]    # Assign to bits 0-1 (upper bound exclusive)

Dynamic Shapes
--------------

For functions that accept tensors of unknown size at compile time:

.. code-block:: python

    def kernel(A: float32[...], B: float32[...], size: int32):
        for i in range(size):
            B[i] = A[i]


Nested Functions
================

Functions can be defined inside kernels:

.. code-block:: python

    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0

        def foo(x: int32) -> int32:
            return x + 1

        for i in range(10):
            B[i] = foo(A[i])
        return B

Index Arguments
---------------

Use the ``index`` type for loop indices passed to functions:

.. code-block:: python

    from allo.ir.types import index

    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0

        def foo(A_: int32[10], x: index) -> int32:
            C: int32[10] = 0
            for i in range(10):
                C[i] = A_[i] + 1
            return C[x]

        for i in range(10):
            B[i] = foo(A, i)
        return B


Built-in Functions
==================

Min and Max
-----------

Element-wise minimum and maximum:

.. code-block:: python

    min_val = min(min_val, A[i])
    max_val = max(max_val, A[i])

Type promotion is handled automatically:

.. code-block:: python

    res[0] = min(A[0], 0)      # int8 with int
    res[1] = max(A[1], 0.0)    # int8 with float -> float comparison

Broadcast Binary Operations
---------------------------

Apply element-wise operations across tensors:

.. code-block:: python

    # Chained broadcast operations
    result = allo.div(allo.mul(allo.sub(allo.add(A, 3), 1), 2), 2)

    # Nested operations
    result = allo.sub(50, allo.mul(2, allo.add(3, allo.div(10, A))))


ConstExpr (Compile-Time Constants)
==================================

``ConstExpr`` declares compile-time constant values that can be used in loop bounds:

.. code-block:: python

    from allo.ir.types import ConstExpr, int32

    M = 10

    def kernel(A: int32[10]) -> int32[10]:
        limit: ConstExpr[int32] = M // 2
        B: int32[10]
        for i in range(limit):  # Loop bound is constant 5
            B[i] = A[i] + 1
        for i in range(limit, 10):
            B[i] = A[i]
        return B

ConstExpr Arithmetic
--------------------

ConstExpr values can be computed from other ConstExpr values:

.. code-block:: python

    base: ConstExpr[int32] = 2
    mult: ConstExpr[int32] = 3
    offset: ConstExpr[int32] = base * mult  # Computed at compile time: 6

Dependent ConstExpr
-------------------

ConstExpr can depend on previously defined ConstExpr:

.. code-block:: python

    N: ConstExpr[int32] = 4
    M: ConstExpr[int32] = N + 2  # 6
    K: ConstExpr[int32] = M + 2  # 8

Using Helper Functions
----------------------

Python helper functions can compute ConstExpr values at compile time:

.. code-block:: python

    import math

    def compute_coefficient(i):
        return math.cos(2.0 * math.pi * i / 8)

    def kernel(A: float32[8], B: float32[8]):
        with allo.meta_for(8) as i:
            coef: ConstExpr[float32] = compute_coefficient(i)
            B[i] = A[i] * coef

.. note::

   ConstExpr variables **must** be initialized at declaration time. Uninitialized ConstExpr will raise an error.


Scoping Rules
=============

Allo enforces C++-style **Block Scoping** rules, which differs from standard Python.

*   **Scope Boundaries**: ``if``, ``elif``, ``else``, ``for``, ``while``, ``meta_if``, ``meta_for``, ``meta_else``.
*   **Rule**: A variable declared for the first time inside a block is **local** to that block. It is not visible after the block exits.
*   **Access**: Inner blocks can read/write variables defined in outer blocks.

Reassignment Validity
---------------------

*   A variable can be reassigned.
*   The new value must match the declared type of the variable.
*   **Immutable Constants**: ``ConstExpr`` variables and values returned by ``df.get_pid()`` are compile-time constants and cannot be reassigned.

Valid Scoping
-------------

Variables should be declared in the scope where they are used:

.. code-block:: python

    def kernel(a: int32) -> int32:
        r: int32 = 0  # Declare outside conditional
        if a == 0:
            r = 1
        else:
            r = 4
        return r

Local variables within a branch are allowed:

.. code-block:: python

    def kernel(a: int32) -> int32:
        r: int32 = 0
        if a > 0:
            t: int32 = 1  # Local to if-branch
            r = r + t
        return r

Invalid Scoping
---------------

The following patterns will raise errors:

**Declaring the same variable in multiple branches:**

.. code-block:: python

    # ERROR: r is not accessible outside branches
    def kernel(a: int32) -> int32:
        if a == 0:
            r: int32 = 1
        else:
            r: int32 = 4
        return r  # Error: r not in scope

**Using loop-local variables outside the loop:**

.. code-block:: python

    # ERROR: tmp is not accessible outside loop
    def kernel(n: int32) -> int32:
        for i in range(n):
            tmp: int32 = i
        return tmp  # Error: tmp not in scope

**Redefining loop variables in nested loops:**

.. code-block:: python

    # ERROR: Cannot redefine i in nested loop
    def kernel(n: int32) -> int32:
        s: int32 = 0
        for i in range(n):
            for i in range(n):  # Error: i already defined
                s = s + i
        return s


Meta-Programming Constructs
===========================

Allo provides compile-time meta-programming constructs that are evaluated during compilation,
enabling conditional code generation and advanced optimizations.

Meta If/Elif/Else
-----------------

Compile-time conditionals that generate different code based on conditions known at compile time. The conditions must be compile-time constants:

.. code-block:: python

    with allo.meta_if(condition1):
        # Code generated only when condition1 is true
        pass

    with allo.meta_elif(condition2):
        # Code generated only when condition1 is false and condition2 is true
        pass

    with allo.meta_else():
        # Code generated when all previous conditions are false
        pass

These are useful for:

- Selecting different implementations based on compile-time parameters
- Specializing kernels for specific data types or array sizes
- Eliminating dead code at compile time

Meta For (Compile-Time Loop Unrolling)
--------------------------------------

``allo.meta_for`` supports multiple argument formats similar to Python's ``range``. The loop bounds and step must be compile-time constants:

.. code-block:: python

    # Single argument: meta_for(upper)
    with allo.meta_for(10) as i:
        A[i] = i

    # Two arguments: meta_for(lower, upper)
    with allo.meta_for(5, 10) as i:
        A[i] = i

    # Three arguments: meta_for(lower, upper, step)
    with allo.meta_for(0, 10, 2) as i:
        A[i] = i * 2


Tensor Attributes and Methods
=============================

Allo provides several built-in attributes and methods for tensor manipulation.

Transpose (.T)
--------------

Transpose a tensor by reversing its dimensions:

.. code-block:: python

    def kernel(A: float32[3, 4]) -> float32[4, 3]:
        return A.T  # Transpose: shape becomes [4, 3]

Copy (.copy)
------------

Create a copy of a tensor:

.. code-block:: python

    B = A.copy()

Bit Reverse (.reverse)
----------------------

Reverse the bits of an integer value (useful for FFT algorithms):

.. code-block:: python

    reversed_bits = x.reverse


Type Conversion Functions
=========================

Explicit Type Casting
---------------------

Use Python built-in functions for explicit type casting:

.. code-block:: python

    # Cast to float32
    b: float32 = float(a)

    # Cast to int32
    c: int32 = int(b)

Fixed-Point Type Attributes
---------------------------

Access type metadata for fixed-point types:

.. code-block:: python

    from allo.ir.types import Fixed

    def kernel(A: Fixed[16, 8]) -> int32:
        return A.bits   # Returns 16 (total bitwidth)
        # A.fracs would return 8 (fractional bits)

Bitcast
-------

Reinterpret the bit pattern of a value as a different type (preserves bits, changes interpretation):

.. code-block:: python

    # Reinterpret float32 bits as int32
    int_bits = float_val.bitcast()

.. note::

   ``bitcast`` preserves the bit pattern but changes the type interpretation.
   This is different from type casting which preserves the value but may change the bits.


Library Operations
==================

Allo provides high-level library operations that map to optimized implementations.

Matrix Operations
-----------------

.. code-block:: python

    # Matrix multiplication
    C = allo.matmul(A, B)

    # Batch matrix multiplication
    C = allo.bmm(A, B)

    # Linear layer: X @ A.T + B
    Y = allo.linear(X, A, B)

Tensor Manipulation
-------------------

.. code-block:: python

    # Transpose with custom permutation
    B = allo.transpose(A, permutation=(1, 0, 2))

    # Reshape/view tensor
    B = allo.view(A, new_shape)

    # Concatenate tensors along an axis
    C = allo.concat(A, B, axis=0)

Element-wise Operations
-----------------------

.. code-block:: python

    # Exponential
    B = allo.exp(A)

    # Logarithm
    B = allo.log(A)

    # Absolute value
    B = allo.abs(A)

Neural Network Operations
-------------------------

.. code-block:: python

    # 2D Convolution (NCHW format)
    output = allo.conv2d(input, kernel)

    # Max pooling
    output = allo.maxpool(input, kernel)

    # Sum pooling
    output = allo.sumpool(input, kernel)

    # ReLU activation
    output = allo.relu(input)

    # Softmax
    output = allo.softmax(input)


Templates and Type Parameters
=============================

Allo supports parameterized kernels using type parameters:

.. code-block:: python

    def kernel[Ty](flag: bool) -> "Ty":
        X: Ty
        if not flag:
            X = 1
        else:
            X = 0
        return X

    s = allo.customize(kernel, instantiate=[int8])

For more details on templates, see the :doc:`../gallery/dive_02_template` documentation.


Building and Execution
======================

After defining a kernel, create a schedule and build the executable:

.. code-block:: python

    import allo
    import numpy as np

    s = allo.customize(gemm)
    mod = s.build()  # Default: LLVM backend

    # Prepare inputs
    np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)

    # Execute
    np_C = mod(np_A, np_B)

For HLS code generation:

.. code-block:: python

    mod = s.build(target="vhls")
    print(mod.hls_code)

