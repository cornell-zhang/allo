# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Vivado/Vitis HLS Backend
========================

**Author**: Hongzheng Chen (hzchen@cs.cornell.edu)


In this tutorial, we will demonstrate how to leverage the Allo DSL to generate
`Vivado/Vitis HLS <https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis/vitis-hls.html>`_ C++ code for FPGA.

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
# Codegen for Vivado/Vitis HLS
# ----------------------------
# Similar to the CPU execution, we only need to change the target of the ``.build()`` function
# in order to target different backends. Here, we use ``vhls`` as the target to generate
# Vivado/Vitis HLS code, which will directly returns the generated code as a string.

code = s.build(target="vhls")
print(code)

# %%
# We can see that the generated code preserves the same structure as the IR, and inserts
# necessary headers and pragmas for Vivado/Vitis HLS. The generated code can be directly passed
# to Vivado/Vitis HLS to generate RTL designs.
#
# .. note::
#
#    Vivado HLS was the previous name of Vitis HLS (before 2020.1). The previous HLS code
#    can still run on the latest Vitis HLS, but the performance of the generated RTL design
#    and the estimated reports may be different, as the newer version of Vitis HLS provides better
#    automatic optimizations.

# %%
# We also provide an easy way to invoke Vitis HLS from Allo. Users can simply provide
# the synthesis mode that are supported by Vitis HLS (e.g., ``csim``, ``csyn``, ``sw_emu``,
# ``hw_emu``, and ``hw``), and the target project folder name. Allo will automatically generate
# the HLS project and invoke the compiler to generate the RTL design.

mod = s.build(target="vitis_hls", mode="csyn", project="gemm.prj")

# %%
# You will see a ``gemm.prj`` folder is generated in the current directory:
#
# - ``host.cpp``: The host (CPU) code that invokes the generated accelerator.
# - ``kernel.cpp``: The generated accelerator code.
# - ``run.tcl``: The Vivado HLS script that can be used to run the Vivado HLS project.
# - ``Makefile``: Defined some shorthands for compiling the project.
#
# To run Vitis HLS, you can simply invoke the built module without passing any arguments into it.
#
# .. note::
#
#    You need to configure the Vitis HLS environment before running the generated code.
#    We have the Vitis environment configured on the Zhang group server, so you can directly
#    ``source /work/shared/common/allo/vitis_2022.1_opt.sh`` to set up the environment.
#
# .. code-block:: python
#
#    mod()

# %%
# After executing the above command, you can check the following report under ``gemm.prj/out.prj/solution1/syn/report/csynth.rpt``.
#
# .. code-block:: python
#
#    +--------------------------------------------------+------------+-----------+----------+------------+---------+
#    |                      Modules                     |  Latency   |  Latency  | Iteration|            |   Trip  |
#    |                      & Loops                     |  (cycles)  |    (ns)   |  Latency |  Interval  |  Count  |
#    +--------------------------------------------------+------------+-----------+----------+------------+---------+
#    |+ gemm                                            |  1080059934|  3.597e+09|         -|  1080059935|        -|
#    | + gemm_Pipeline_VITIS_LOOP_44_1_VITIS_LOOP_45_2  |     1048578|  3.492e+06|         -|     1048578|        -|
#    |  o VITIS_LOOP_44_1_VITIS_LOOP_45_2               |     1048576|  3.492e+06|         2|           1|  1048576|
#    | o l_S_buf0_buf0_l_0_l_buf0_l_1                   |     1048576|  3.492e+06|         2|           1|  1048576|
#    | o l_S_buf1_buf1_l_0_l_buf1_l_1                   |     1048576|  3.492e+06|         2|           1|  1048576|
#    | o l_S_i_j_0_i                                    |  1075865600|  3.583e+09|   1050650|           -|     1024|
#    |  + gemm_Pipeline_l_j_init                        |        1026|  3.417e+03|         -|        1026|        -|
#    |   o l_j_init                                     |        1024|  3.410e+03|         1|           1|     1024|
#    |  + gemm_Pipeline_l_S_k_0_k_l_j                   |     1048591|  3.492e+06|         -|     1048591|        -|
#    |   o l_S_k_0_k_l_j                                |     1048589|  3.492e+06|        15|           1|  1048576|
#    |  + gemm_Pipeline_l_j_back                        |        1027|  3.420e+03|         -|        1027|        -|
#    |   o l_j_back                                     |        1025|  3.413e+03|         3|           1|     1024|
#    | o l_S_result2_result2_l_0_l_result2_l_1          |     1048578|  3.492e+06|         4|           1|  1048576|
#    +--------------------------------------------------+------------+-----------+----------+------------+---------+
#
# From the above output, we can clearly see that all the loops inside the GEMM kernel (marked as ``o``) are pipelined
# with Initiation Interval (II) equal to 1. You can also find more detailed information under the ``report`` folder.
