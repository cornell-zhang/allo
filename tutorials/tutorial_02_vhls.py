# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Vivado HLS Backend
==================

**Author**: Hongzheng Chen (hzchen@cs.cornell.edu)


In this tutorial, we will demonstrate how to leverage the Allo DSL to generate
Vivado HLS code for FPGA.

Import Allo
-----------
First, we import the necessary packages.
"""

import allo
from allo.ir.types import float32

##############################################################################
# Algorithm Definition
# --------------------
# We again define a general matrix multiplication (GEMM) in this tutorial.
# However, we will make some changes to demonstrate more features of the DSL.
#
# We can define the constants as follows, which denotes the matrix sizes:

M, N, K = 1024, 1024, 1024

# %%
# Here, we define the main computation of the GEMM but use ``float32`` as the
# data type. Notice that users can easily leverage the previously defined arguments
# (e.g., ``M``, ``N``, and ``K``) to construct the matrices, and Allo will
# automatically captures the global variables.
#
# Since Allo has a strict type system, we need to be careful about the
# data types of the variables. To initialize matrix ``C`` with all zeros, we
# need to pass in a floating-point value ``0.0`` instead of an integer.
#
# We also use the ``allo.reduction`` API to denote the reduction axis. The
# reduction axis is the loop iterator that is used to accumulate the result.
# In this example, we use ``k`` as the reduction axis, which means the
# computation of ``C[i, j]`` will be accumulated along the ``k`` dimension.
# This annotation is necessary for later optimizations, since Allo leverages
# this information to generate correct intermediate buffers.


def gemm(A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
    C: float32[M, N] = 0.0
    for i, j in allo.grid(M, N):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C


##############################################################################
# Scalar-Vector Product for GEMM
# ------------------------------
#
# Next, we create a schedule for the GEMM and start to optimize the program.
# We try to implement the **interleaving accumulation** technique presented in
# `this paper <https://arxiv.org/abs/1805.08288>`_, which is also viewed as
# the **scalar-vector product** since it changes the computation order of the
# original dot-product.
#
# .. image:: ../_static/scalar-vector-product.png
#    :width: 600
#
# .. note::
#
#    To get more rational of this technique, please refer to the above mentioned
#    paper from Torsten Hoefler's group.

s = allo.customize(gemm)

# %%
# We first reorder the inner reduction loop with the middle loop.
# This is used to change the computation order of matrix multiplication.

s.reorder("k", "j")
print(s.module)

# %%
# .. note::
#
#    This reordering seems to be easy, but it is impossible in the old Allo,
#    since the previous Allo directly generate reduction variables which make
#    the ``j`` loop becomes imperfect, while MLIR only supports reordering perfect
#    loops.

# %%
# Next, we create a new buffer for the output tensor ``C``.
# We provide a ``.buffer_at()`` primitive for users to quickly create a new buffer
# along a specific axis. Since Allo has attached all the tensors to the function,
# we can directly use ``<schedule>.<tensor>`` to access a specific tensor in the schedule.

s.buffer_at(s.C, axis="i")
print(s.module)

# %%
# From the above generated code, we can see that Allo automatically
# creates an intermediate buffer ``%1`` for ``C`` and attach it inside the ``i`` loop.
# Also two additional loop nested named ``j_init`` and ``j_back`` are created to
# initialize and write the intermediate buffer back to output tensor.

# %%
# Lastly, we pipeline the ``j`` loop in order to achieve the best performance.

s.pipeline("j")
print(s.module)

##############################################################################
# Codegen for Vivado HLS
# ----------------------
# Similar to the CPU execution, we only need to change the target of the ``.build()`` function
# in order to target different backends. Here, we use ``vhls`` as the target to generate
# Vivado HLS code, which will returns the generated code as a string.

code = s.build(target="vhls")
print(code)

# %%
# We can see that the generated code preserves the same structure as the IR, and inserts
# necessary headers and pragmas for Vivado HLS. The generated code can be directly passed
# to Vivado HLS to generate RTL designs.

# %%
# We also provide an easy way to invoke Vivado HLS from Allo. Users can simply provide
# the synthesis mode that are supported by Vivado HLS (e.g., ``csim``, ``csyn``, ``cosim``,
# and ``impl``), and the target project folder name. Allo will automatically generate
# the HLS project and invoke the compiler to generate the RTL design.

mod = s.build(target="vhls", mode="csyn", project="gemm.prj")

# %%
# You will see a ``gemm.prj`` folder is generated in the current directory:
#
# - ``host.cpp``: The host (CPU) code that invokes the generated accelerator.
# - ``kernel.cpp``: The generated accelerator code.
# - ``run.tcl``: The Vivado HLS script that can be used to generate the Vivado HLS project.
# - ``Makefile``: Defined some shorthands for compiling the project.
#
# To run Vivado HLS, you can simply invoke the built module without passing any arguments into it.
#
# .. note::
#
#    You need to configure the Vivado HLS environment before running the generated code.
#    We have the Vivado environment configured in the ``brg-zhang`` server, so you can directly
#    ``source /work/shared/common/allo/vitis_2019.2_opt.sh`` to set up the environment.
#
# .. code-block:: python
#
#    mod()

# %%
# After executing the above command, you will see the following output:
#
# .. code-block:: python
#
#    +-------------------+-----------------------------------+
#    | HLS Version       | Vivado HLS 2019.2.1               |
#    | Product family    | zynq                              |
#    | Target device     | xc7z020-clg484-1                  |
#    | Top Model Name    | gemm                              |
#    +-------------------+-----------------------------------+
#    | Target CP         | 10.00 ns                          |
#    | Estimated CP      | 8.052 ns                          |
#    | Latency (cycles)  | Min 1077958658; Max 1077958658    |
#    | Interval (cycles) | Min 1077958659; Max 1077958659    |
#    | Resources         | Type        Used    Total    Util |
#    |                   | --------  ------  -------  ------ |
#    |                   | BRAM_18K       2      280      1% |
#    |                   | DSP48E         5      220      2% |
#    |                   | FF           862   106400      1% |
#    |                   | LUT         1375    53200      3% |
#    +-------------------+-----------------------------------+
#    +---------------+--------------+------------+---------------------+---------------+------------------+
#    |               |   Trip Count |    Latency |   Iteration Latency |   Pipeline II |   Pipeline Depth |
#    |---------------+--------------+------------+---------------------+---------------+------------------|
#    | Loop1         |         1024 |    2099200 |                2050 |           N/A |              N/A |
#    | + Loop1.1     |         1024 |       2048 |                   2 |           N/A |              N/A |
#    | l_S_i_j_i     |         1024 | 1075859456 |             1050644 |           N/A |              N/A |
#    | + l_j_init    |         1024 |       1024 |                 N/A |             1 |                1 |
#    | + l_S_k_k_l_j |      1048576 |    1048588 |                 N/A |             1 |               14 |
#    | + l_j_back    |         1024 |       1025 |                 N/A |             1 |                3 |
#    +---------------+--------------+------------+---------------------+---------------+------------------+
#    * Units in clock cycles
#
# From the above output, we can clearly see that all the loops inside the GEMM kernel are pipelined
# with II=1.
#
# .. note::
#
#   The results are also printed to a file named ``report.json`` for further analysis.
