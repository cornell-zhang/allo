# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Getting Started
===============

**Author**: Hongzheng Chen (hzchen@cs.cornell.edu)

In this tutorial, we demonstrate the basic usage of Allo.

Import Allo
-----------
First we import the necessary packages.
"""

import allo


##############################################################################
# Algorithm Definition
# --------------------
# Allo leverages an algorithm-optimization decoupled paradigm, which means
# users can first define the algorithm in a high-level language and then
# optimize the program with various hardware customization techniques (i.e.,
# schedule primitives). Here we show how to define a general matrix multiplication
# (GEMM) in the Allo DSL.
#
# We first import the necessary data types from Allo. In this example, we
# use ``int32`` as the data type for all the variables.

from allo.ir.types import int32

# %%
# We then define a function that takes two 32x32 matrices as inputs and
# returns a 32x32 matrix as output. The variable declaration is defined
# as ``<name>: <type>[<shape>]``, and the function type is defined as
# ``(<in_type0>, <in_type1>, ...) -> <out_type>``.
# We require **strict type annotation** in Allo's kernels, which is different
# from directly programming in Python.
#
# Inside the kernel, we provide a shorthand for the loop iterator. For example,
# ``for i, j, k in allo.grid(32, 32, 32)`` is equivalent to the following
# nested for-loop:
#
# .. code-block:: python
#
#    for i in range(32):
#        for j in range(32):
#            for k in range(32):
#                # body
#
# The ``allo.grid`` API is used to define the iteration space of the loop.
# The arguments denote the upper bounds of the loop iterators.
# Notice the above range-loop is also supported in the new Allo, so
# users have more flexibility to define the loop structure.


def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    C: int32[32, 32] = 0
    for i, j, k in allo.grid(32, 32, 32):
        C[i, j] += A[i, k] * B[k, j]
    return C


##############################################################################
# Create the Schedule
# -------------------
# After defining the algorithm, we can start applying transformations to the
# kernel in order to achieve high performance. We call ``allo.customize`` to
# create a schedule for the kernel, where **schedule** denotes the set of
# transformations.

s = allo.customize(gemm)

##############################################################################
# Inspect the Intermediate Representation (IR)
# --------------------------------------------
# Allo leverage the `MLIR <https://mlir.llvm.org/>`_ infrastructure to
# represent the program, and we can directly print out the IR by using
# ``s.module``.

print(s.module)

# %%
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
# Allo also attaches some attributes to the operations, including the tensor
# names, loop names, and operation names, which are further used for optimization.

##############################################################################
# Apply Transformations
# ---------------------
# Next, we start transforming the program by using the schedule primitives.
# We can refer to the loops by using the loop names. For example, to split
# the outer-most loop into two, we can call the ``.split()`` primitive as follows:

s.split("i", factor=8)

# %%
# We can print out the IR again to see the effect of the transformation.
#
# .. note::
#
#   In the Allo DSL, all the transformations are applied **immediately**,
#   so users can directly see the changes after they apply the transformations.

print(s.module)

# %%
# We can see that the outer-most loop is split into two loops, and the
# original loop is replaced by the two new loops. The new loops are named
# as ``i.outer`` and ``i.inner``.
#
# Similarly, we can split the ``j`` loop:

s.split("j", factor=8)
print(s.module)

# %%
# We can further reorder the loops by using ``.reorder()``. For example, we
# can move the splitted outer loops together, and move the splitted inner
# loops together.

s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
print(s.module)

# %%
# We can see the changes from the loop names in the generated IR.

##############################################################################
# Create the Executable
# ---------------------
# The next step is to generate the executable from the schedule. We can
# directly call ``.build()`` function on the schedule and specify the target
# hardware as ``llvm``. By default, Allo will generate a LLVM program that
# can be executed on the CPU. Otherwise, you can also specify the target as
# ``vhls`` to generate a Vivado HLS program that can be synthesized to an FPGA
# accelerator.

mod = s.build(target="llvm")

# %%
# .. note::
#
#   ``s.build(target="llvm")`` is equivalent to ``s.build()``.

##############################################################################
# Prepare the Inputs/Outputs for the Executable
# ---------------------------------------------
# To run the executable, we can generate random NumPy arrays as input data, and
# directly feed them into the executable. Allo will automatically handle the
# input data and generate corresponding internal wrappers for LLVM to execute,
# but we still need to make sure the data types are consistent. By default,
# ``np.random.randint`` will generate ``np.int64`` data type, while we use ``int32``
# when defining our kernel function, so we need to explicitly cast the data type
# to ``np.int32``.

import numpy as np

np_A = np.random.randint(0, 100, (32, 32)).astype(np.int32)
np_B = np.random.randint(0, 100, (32, 32)).astype(np.int32)

##############################################################################
# Run the Executable
# ------------------
# With the prepared inputs/outputs, we can feed them to our executable.
# Notice our module can return a new array as output, so we can directly
# assign the output to a new variable.

np_C = mod(np_A, np_B)

##############################################################################
# Finally, we can do a sanity check to see if the results are correct.

golden_C = np.matmul(np_A, np_B)
np.testing.assert_allclose(np_C, golden_C, rtol=1e-5, atol=1e-5)
print("Results are correct!")
