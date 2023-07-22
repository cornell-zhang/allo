# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Getting Started
===============

**Author**: Hongzheng Chen (hzchen@cs.cornell.edu)

In this tutorial, we demonstrate the basic usage of HeteroCL-MLIR with the new DSL frontend.

Import HeteroCL
---------------
We usually use ``hcl`` as the acronym of HeteroCL.
"""

import heterocl as hcl


##############################################################################
# Algorithm Definition
# --------------------
# HeteroCL leverages a algorithm-optimization decoupled paradigm, which means
# users can first define the algorithm in a high-level language and then
# optimize the program with various hardware customization techniques (i.e.,
# schedule primitives). Here we show how to define a general matrix multiplication
# (GEMM) in the new HeteroCL DSL.
#
# We first import the necessary data types from HeteroCL. In this example, we
# use ``int32`` as the data type for all the variables.

from heterocl.ir.types import int32

# We then define a function that takes two 32x32 matrices as inputs and
# returns a 32x32 matrix as output. The variable declaration is defined
# as ``<name>: <type>[<shape>]``. We require **strict type annotation** in
# HeteroCL's kernels, which is different from directly programming in Python.
#
# Inside the kernel, we provide a shorthand for the loop iterator. For example,
# ``for i, j, k in hcl.grid(32, 32, 32)`` is equivalent to the following
# nested for-loop:
#
# .. code-block:: python
#
#    for i in range(32):
#        for j in range(32):
#            for k in range(32):
#                # body
#
# The ``hcl.grid`` API is used to define the iteration space of the loop.
# The arguments denote the upper bounds of the loop iterators.


def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    C: int32[32, 32] = 0
    for i, j, k in hcl.grid(32, 32, 32):
        C[i, j] += A[i, k] * B[k, j]
    return C


##############################################################################
# Create the Schedule
# -------------------
# After defining the algorithm, we can start applying transformations to the
# kernel in order to achieve high performance. We call ``hcl.customize`` to
# create a schedule for the kernel, where **schedule** denotes the set of
# transformations.

s = hcl.customize(gemm)

##############################################################################
# Inspect the Intermediate Representation (IR)
# --------------------------------------------
# HeteroCL leverage the `MLIR <https://mlir.llvm.org/>`_ infrastructure to
# represent the program, and we can directly print out the IR by using
# ``s.module``.

print(s.module)

# Let's take a close look at the generated IR. Basically an MLIR program is
# a set of operations in different dialects, and the operations are referred
# to as **<dialect>.<ops>**. In this example, we can see that the generated IR
# contains the following dialects:
#
# - ``func``: Used to define the function signature and the return of the function.
# - ``memref``: Used to define the shape and memory layout of the tensors.
# - ``affine``: Used to define the loop structure.
# - ``arith``: Used to conduct actual arithmetic operations.
# - ``linalg``: Currently only used to initialize the tensors.
#
# And the inner-most dot-product is explicitly represented by a sequence of load/store
# operations and some arithmetic operations.
# HeteroCL also attaches some attributes to the operations, including the tensor
# names, loop names, and operation names, which are further used for optimization.

##############################################################################
# Apply Transformations
# ---------------------
# Next, we start transforming the program by using the schedule primitives.
# We can refer to the loops by using the loop names. For example, to split
# the outer-most loop into two, we can call the ``.split()`` primitive as follows:

s.split("i", factor=8)

# We can print out the IR again to see the effect of the transformation.
#
# .. note::
#
#   In the new HeteroCL DSL, all the transformations are applied **immediately**,
#   so users can directly see the changes after they apply the transformations.

print(s.module)

# We can see that the outer-most loop is split into two loops, and the
# original loop is replaced by the two new loops. The new loops are named
# as ``i.outer`` and ``i.inner``.
#
# Similarly, we can split the ``j`` loop:

s.split("j", factor=8)
print(s.module)

# We can further reorder the loops by using ``.reorder()``. For example, we
# can move the splitted outer loops together, and move the splitted inner
# loops together.

s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
print(s.module)

# We can see the changes from the loop names in the generated IR.

##############################################################################
# Create the Executable
# ---------------------
# The next step is to generate the executable from the schedule. We can
# directly call ``.build()`` function on the schedule and specify the target
# hardware as ``llvm``. By default, HeteroCL will generate a LLVM program that
# can be executed on the CPU. Otherwise, you can also specify the target as
# ``vhls`` to generate a Vivado HLS program that can be synthesized to an FPGA
# accelerator.

mod = s.build(target="llvm")

##############################################################################
# Prepare the Inputs/Outputs for the Executable
# ---------------------------------------------
# To run the executable, we can generate random NumPy arrays as input data, and
# directly feed them into the executable. HeteroCL will automatically handle the
# input data and generate corresponding internal wrappers for LLVM to execute.

import numpy as np

np_A = np.random.randint(100, size=(32, 32))
np_B = np.random.randint(100, size=(32, 32))
np_C = np.zeros((32, 32), dtype=np.int32)

##############################################################################
# Run the Executable
# ------------------
# With the prepared inputs/outputs, we can feed them to our executable.
# Notice the output is also passed into the HeteroCL as a function argument,
# and the result will be directly written into the output array.

mod(np_A, np_B, np_C)

##############################################################################
# Finally, we can do a sanity check to see if the results are correct.

golden_C = np.matmul(np_A, np_B)

assert np.array_equal(np_C, golden_C)
