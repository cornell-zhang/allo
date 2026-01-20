# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Other Features
==============

**Author**: Hongzheng Chen (hzchen@cs.cornell.edu)

This document will discuss other features that are not covered in the previous tutorials.
"""

##############################################################################
# Dynamic Shapes
# --------------
# In some cases, the shape of the tensor is not known at compile time, so we can use ``[...]`` to represent the dynamic shape.
# From the generated MLIR module, we can see it has a ``"?"`` in the shape of the tensor, which means the shape is not predefined,
# but we can still run the LLVM module with arbitrary shapes of NumPy arrays.

import allo
from allo.ir.types import int32, float32
import numpy as np


def kernel(A: float32[...], B: float32[...], size: int32):
    for i in range(size):
        B[i] = A[i]


s = allo.customize(kernel)
print(s.module)
np_A = np.random.random((256,)).astype(np.float32)
allo_A = np.zeros((256,)).astype(np.float32)
mod = s.build()
mod(np_A, allo_A, 256)
np.testing.assert_allclose(np_A, allo_A)

# %%
# We can also check the generated HLS code that the arguments are declared as pointers.

code = s.build(target="vhls")
print(code)

##############################################################################
# Tuple Return
# ------------
# Another feature is the tuple support. As in Python, we can return multiple values from a function, Allo
# also supports this by explicitly specifying the return type as a tuple.


def callee(a: float32, b: float32) -> (float32, float32):
    c: float32 = a + b
    d: float32 = a - b
    return c, d


def kernel(A: float32[10], B: float32[10]) -> (float32[10], float32[10]):
    C: float32[10] = 0
    D: float32[10] = 0
    for i in range(10):
        C[i], D[i] = callee(A[i], B[i])
    return C, D


s = allo.customize(kernel)
print(s.module)
mod = s.build()
np_A = np.random.random((10,)).astype(np.float32)
np_B = np.random.random((10,)).astype(np.float32)
np_C, np_D = mod(np_A, np_B)
np_C_ref = np.zeros((10,), dtype=np.float32)
np_D_ref = np.zeros((10,), dtype=np.float32)
for i in range(10):
    np_C_ref[i], np_D_ref[i] = callee(np_A[i], np_B[i])
np.testing.assert_allclose(np_C, np_C_ref)
np.testing.assert_allclose(np_D, np_D_ref)

##############################################################################
# Compile-Time Constant Expressions (ConstExpr)
# ----------------------------------------------
# ``ConstExpr`` allows you to declare variables that are evaluated at Python
# level during compilation. This enables using Python helper functions for
# compile-time computations like computing coefficients or lookup indices.

from allo.ir.types import ConstExpr


# Python helper functions - evaluated at compile time
def compute_coefficient(i):
    """Compute a coefficient based on index."""
    import math

    return math.cos(2.0 * math.pi * i / 8)


def compute_index(i, offset):
    """Compute a transformed index."""
    return (i + offset) % 8


def kernel_with_constexpr(A: float32[8], B: float32[8]):
    with allo.meta_for(8) as i:
        # ConstExpr values are computed at Python level during compilation
        coef: ConstExpr[float32] = compute_coefficient(i)
        idx: ConstExpr[int32] = compute_index(i, 3)

        # Use the compile-time constants in expressions
        B[i] = A[idx] * coef


s = allo.customize(kernel_with_constexpr)
print(s.module)

# %%
# From the generated MLIR, you can see that the ``coef`` and ``idx`` values
# are embedded as constants in the code, not computed at runtime.
#
# Key points about ``ConstExpr``:
#
# - The RHS is evaluated at Python level, not compiled as Allo code
# - You can use arbitrary Python functions (math, etc.)
# - Use ``ConstExpr[int32]`` or ``ConstExpr[float32]`` for the type annotation
