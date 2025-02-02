# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Kernel Composition
==================

**Author**: Hongzheng Chen (hzchen@cs.cornell.edu)

This document will discuss kernel composition.
In the previous tutorials, we have seen how to write a simple kernel.
However, in real applications, we often need to compose multiple kernels together.

In the following example, we define a ``matrix_add`` and a ``gemm`` kernel, and wrap them into a ``top``-level function.
"""

import allo
from allo.ir.types import int32, float32

M, K, N = 32, 32, 32


def matrix_add(A: int32[M, N]) -> int32[M, N]:
    B: int32[M, N] = 0
    for i, j in allo.grid(M, N):
        B[i, j] = A[i, j] + 1
    return B


def gemm(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
    C: int32[M, N] = 0
    for i, j in allo.grid(M, N):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C


def top(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
    C = gemm(A, B)
    D = matrix_add(C)
    return D


# %%
# Different teams or people can then work on different parts of the code and optimize each kernel.
# We first create a schedule for the ``matrix_add`` kernel, and add several optimizations.

s1 = allo.customize(matrix_add)
s1.pipeline("j")
print(s1.module)

# %%
# Then we create a schedule for the ``gemm`` kernel and optimize it.

s2 = allo.customize(gemm)
s2.reorder("k", "j")
s2.buffer_at(s2.C, axis="i")
s2.pipeline("j")
print(s2.module)

# %%
# Notice that now we only optimize the separate kernels but do not incorporate them into the top-level function, as shown in the following printed module.

s = allo.customize(top)
print(s.module)

# %%
# Therefore, after each part has been optimized, we need to explicitly *compose* them together.
# In Allo, we can use the ``.compose()`` primitive to compose the schedules together into the parent function.

s.compose([s1, s2])
print(s.module)

# %%
# We can see that the schedules for the ``matrix_add`` and ``gemm`` kernels are both correctly optimized in the top-level function.

##############################################################################
# Template Composition
# --------------------
# Sometimes we may define template kernels and invoke the kernel with different template arguments. Allo provides an *id* option to specify the exact kernel to be composed.


def kernel[T_in, T_out, S](A: "T_in[S]") -> "T_out[S]":
    B: T_out[S] = 0
    for i in range(S):
        with allo.meta_if(T_out == int32):
            B[i] = A[i] + 1
        with allo.meta_else():
            B[i] = A[i] * 2
    return B


def top2(A: int32[M]) -> float32[M]:
    C = kernel[int32, int32, M, "K1"](A)
    D = kernel[int32, float32, M, "K2"](C)
    return D


# %%
# Specifically, the last argument of the template kernel is the *id* of the kernel. Later on we can use this ID for distinguishing different kernels during composition.
# We also customize the two template kernels with different optimizations first.

s1 = allo.customize(kernel, instantiate=[int32, int32, M])
s1.unroll("i", factor=4)
print(s1.module)

s2 = allo.customize(kernel, instantiate=[int32, float32, M])
s2.pipeline("i")
print(s2.module)

# %%
# Finally, we compose the two template kernels into the top-level function with the ID specified.

s = allo.customize(top2)
s.compose(s1, id="K1")
s.compose(s2, id="K2")
print(s.module)

# %%
# We can see from the printed module that the loop in the first kernel is unrolled by a factor of 4, and the loop in the second kernel is pipelined.
