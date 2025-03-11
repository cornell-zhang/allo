# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Equivalence Checking
====================

**Author**: Hongzheng Chen (hzchen@cs.cornell.edu)

In this tutorial, we demonstrate how to use Allo's verifier facility to check the equivalence of different scheduling transformations. The verifier ensures that various optimizations applied to the same algorithm do not alter its functional behavior.

First, we import the necessary packages:
"""

import allo
from allo.ir.types import float32

##############################################################################
# Create the Schedule
# -------------------
# We define a general matrix multiplication (GEMM) kernel that takes two 32x32 matrices as inputs
# and produces a 32x32 output matrix. The reduction loop is used to accumulate the multiplication results.


def gemm(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
    C: float32[32, 32] = 0
    for i, j in allo.grid(32, 32):
        for k in allo.reduction(32):
            C[i, j] += A[i, k] * B[k, j]
    return C


# %%
# We create two schedules for the GEMM kernel using different transformations.
# The first schedule, **s1**, applies a loop reordering transformation, while the second schedule, **s2**, applies a buffering transformation on the output tensor.

s1 = allo.customize(gemm)
s1.reorder("gemm:i", "gemm:j")
print(s1.module)

# %%
# In the code above, **s1** is customized by reordering the loops corresponding to indices ``i`` and ``j``.
# The printed intermediate representation (IR) shows the effect of this transformation.

s2 = allo.customize(gemm)
s2.buffer_at(s2.C, axis="i")
print(s2.module)

# %%
# Here, a buffering transformation is applied on tensor ``C`` along the ``i`` axis.
# The IR output confirms that the transformation has been incorporated.
# Although the schedules differ in structure, they should implement equivalent functionality.

##############################################################################
# Verifying Equivalence
# ---------------------
# Next, we use the verifier facility to check whether the two schedules, **s1** and **s2**,
# are equivalent. The ``allo.verify`` function compares the schedules and returns a truthy
# value if they are functionally identical.

verifier = allo.verify(s1, s2)
assert verifier, "Failed to verify the equivalence of two schedules!"
print("s1 and s2 are equivalent!")

# %%
# The assertion confirms that the transformations applied in **s1** and **s2** preserve the semantics of the original GEMM algorithm.

##############################################################################
# Introducing a Non-equivalent Schedule
# -------------------------------------
# To illustrate the effectiveness of the verifier, we define an alternative GEMM kernel, **gemm_wrong**,
# which incorrectly implements the multiplication by overwriting the output instead of
# accumulating the results. The schedule derived from **gemm_wrong** (named **s3**)
# should not be equivalent to **s1**.


def gemm_wrong(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
    C: float32[32, 32] = 0
    for i, j in allo.grid(32, 32):
        for k in allo.reduction(32):
            C[i, j] = A[i, k] * B[k, j]
    return C


s3 = allo.customize(gemm_wrong)
print(s3.module)
verifier = allo.verify(s1, s3)
assert not verifier, "Failed to verify the equivalence of two schedules!"
print("s1 and s3 are not equivalent!")

# %%
# The verifier correctly detects that **s3** does not preserve the intended accumulation,
# thus confirming that **s1** and **s3** are not equivalent.

##############################################################################
# Conclusion
# ----------
# This tutorial has demonstrated how to use Allo's verifier facility to ensure that different
# scheduling transformations yield equivalent computational behavior. By verifying the equivalence
# of various schedules, you can confidently apply optimizations without compromising the
# functional correctness of your algorithms.
