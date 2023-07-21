# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Getting Started
===============

**Author**: Yi-Hsiang Lai (seanlatias@github)

In this tutorial, we demonstrate the basic usage of HeteroCL.

Import HeteroCL
---------------
We usually use ``hcl`` as the acronym of HeteroCL.
"""

import heterocl as hcl

##############################################################################
# Initialize the Environment
# --------------------------
# We need to initialize the environment for each HeteroCL application. We can
# do this by calling the API ``hcl.init()``. We can also set the default data
# type for every computation via this API. The default data type is **32-bit**
# integers.
#
# .. note::
#
#    For more information on the data types, please see
#    :ref:`sphx_glr_gallery_tutorial_05_dtype.py`.

hcl.init()

##############################################################################
# Algorithm Definition
# --------------------
# After we initialize, we define the algorithm by using a Python function
# definition, where the arguments are the input tensors. The function can
# optionally return tensors as outputs. In this example, the two inputs are a
# scalar `a` and a tensor `A`, and the output is also a tensor `B`. The main
# difference between a scalar and a tensor is that *a scalar cannot be updated*.
#
# Within the algorithm definition, we use HeteroCL APIs to describe the
# operations. In this example, we use a tensor-based declarative-style
# operation ``hcl.compute``. We also show the equivalent  Python code.
#
# .. note::
#
#    For more information on the APIs, please see
#    :ref:`sphx_glr_gallery_tutorial_03_api.py`


def simple_compute(a, A):
    B = hcl.compute(A.shape, lambda x, y: A[x, y] + a.v, "B")
    """
    The above API is equivalent to the following Python code.

    for x in range(0, 10):
        for y in range(0, 10):
            B[x, y] = A[x, y] + a
    """

    return B


##############################################################################
# Inputs/Outputs Definition
# -------------------------
# One of the advantages of such *modularized algorithm definition* is that we
# can reuse the defined function with different input settings. We use
# ``hcl.placeholder`` to set the inputs, where we specify the shape, name,
# and data type. The shape must be specified and should be in the form of a
# **tuple**. If it is empty (i.e., `()`), the returned object is a *scalar*.
# Otherwise, the returned object is a *tensor*. The rest two fields are
# optional. In this example, we define a scalar input `a` and a
# two-dimensional tensor input `A`.
#
# .. note::
#
#    For more information on the interfaces, please see
#    :obj:`heterocl.placeholder`

a = hcl.placeholder((), "a")
A = hcl.placeholder((10, 10), "A")

##############################################################################
# Apply Hardware Customization
# ----------------------------
# Usually, our next step is apply various hardware customization techniques to
# the application. In this tutorial, we skip this step which will be discussed
# in the later tutorials. However, we still need to build a default schedule
# by using ``hcl.create_schedule`` whose inputs are a list of inputs and
# the Python function that defines the algorithm.

s = hcl.create_schedule([a, A], simple_compute)

##############################################################################
# Inspect the Intermediate Representation (IR)
# --------------------------------------------
# A HeteroCL program will be lowered to an IR before backend code generation.
# HeteroCL provides an API for users to inspect the lowered IR. This could be
# helpful for debugging.

print(hcl.lower(s))

##############################################################################
# Create the Executable
# ---------------------
# The next step is to build the executable by using ``hcl.build``. You can
# define the target of the executable, where the default target is `llvm`.
# Namely, the executable will be run on CPU. The input for this API is the
# schedule we just created.

f = hcl.build(s)

##############################################################################
# Prepare the Inputs/Outputs for the Executable
# ---------------------------------------------
# To run the generated executable, we can feed it with Numpy arrays by using
# ``hcl.asarray``. This API transforms a Numpy array to a HeteroCL container
# that is used as inputs/outputs to the executable. In this tutorial, we
# randomly generate the values for our input tensor `A`. Note that since we
# return a new tensor at the end of our algorithm, we also need to prepare
# an input array for tensor `B`.

import numpy as np

hcl_a = 10
np_A = np.random.randint(100, size=A.shape)
hcl_A = hcl.asarray(np_A)
hcl_B = hcl.asarray(np.zeros(A.shape))

##############################################################################
# Run the Executable
# ------------------
# With the prepared inputs/outputs, we can finally feed them to our executable.

f(hcl_a, hcl_A, hcl_B)

##############################################################################
# View the Results
# ----------------
# To view the results, we can transform the HeteroCL tensors back to Numpy
# arrays by using ``asnumpy()``.

np_A = hcl_A.asnumpy()
np_B = hcl_B.asnumpy()

print(hcl_a)
print(np_A)
print(np_B)

##############################################################################
# Let's run a test

assert np.array_equal(np_B, np_A + 10)
