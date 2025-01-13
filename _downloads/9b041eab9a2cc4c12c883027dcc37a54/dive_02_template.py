# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Template Kernels
================

**Author**: Hongzheng Chen (hzchen@cs.cornell.edu)

This document explains how to write a template kernel in Allo.
Template kernels are useful when we need to reuse a kernel with different data types or when certain computation patterns depend on specific constants.
By leveraging template kernels, we can achieve greater flexibility and reusability in the code.
"""

import allo
from allo.ir.types import int32, float32

# %%
# We follow Python's convention to use *type variable* to define a template kernel.
# Specifically, the type variable is specified after the function name using square brackets: ``def kernel[T](...)``, and the type variable can be used in the function signature and body.
# Importantly, as the native Python interpreter does not support Allo's type declaration (i.e., base type + shape), we need to use string annotations like ``"T[10]"`` to specify the type of the variables.
# Otherwise, it will raise a type error.
#
# In the following, we define a simple addition function that adds 1 to each element of the input array.
# To invoke the kernel with a specific data type, we can use the ``instantiate`` argument in the ``allo.customize`` function.


def kernel[T](A: "T[10]") -> "T[10]":
    B: T[10]
    for i in range(10):
        B[i] = A[i] + 1
    return B


s = allo.customize(kernel, instantiate=[int32])
print(s.module)

# %%
# We can see that the kernel is specialized with the given ``int32`` data type.
# Similarly, we can directly declare a new kernel by specifying ``float32`` as the data type.

s = allo.customize(kernel, instantiate=[float32])
print(s.module)

# %%
# If we not only want to specialize the data type but also the shape of the array, we can provide another type variable, and pass it to the ``instantiate`` argument.
# Note that here we also use the ``<type_var>: base_type`` notation to constrain the type of the type variable. Here we constrain the type variable ``M`` to be an integer.


def kernel2[T, M: int32](A: "T[M]") -> "T[M]":
    B: T[M]
    for i in range(M):
        B[i] = A[i] + 1
    return B


s = allo.customize(kernel2, instantiate=[int32, 20])
print(s.module)

# %%
# Furthermore, Allo's template also enables metaprogramming that can evaluate type variables at compile time.
# Specifically, we can use the ``allo.meta_if``, ``allo.meta_elif``, and ``allo.meta_else`` to conditionally generate code based on the type variables.
# Just to make sure the conditions can be determined at compile time.


def kernel3[T, M: int32](A: "T[M]") -> "T[M]":
    B: T[M]
    for i in range(M):
        with allo.meta_if(T == int32):
            B[i] = A[i] + 1
        with allo.meta_else():
            B[i] = A[i] - 1
    return B


# %%
# In final generated code, we can see that only a single branch is generated based on the given data type.

s = allo.customize(kernel3, instantiate=[int32, 20])
print(s.module)
s = allo.customize(kernel3, instantiate=[float32, 20])
print(s.module)
