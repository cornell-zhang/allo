# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data Types and Type Casting
===========================

**Author**: Hongzheng Chen (hzchen@cs.cornell.edu)

This document will discuss the Allo-supported data types in detail.
All the data types are defined in the ``allo.ir.types`` module.
"""

import allo
from allo.ir.types import int16, int32, float32, Int, UInt, Float, Fixed

##############################################################################
# Currently, Allo supports three base data types for mathematical operations:
#
# - Integers: ``Int(bitwdith)``, ``UInt(bitwidth)``
# - Floating points: ``Float(bitwidth)`` (only support 16, 32, and 64 bits)
# - Fixed points: ``Fixed(bitwidth, frac)``, ``UFixed(bitwidth, frac)``
#
# For example, one can declare a 15-bit integer as ``Int(15)`` and an unsigned 8-bit fixed-point number with 3 fractional bits as ``UFixed(8, 3)``.
# For all the C/C++ supported data types, we provide shorthands like ``float32`` and ``int16`` to easily declare them.

# %%
# Notice different from native Python, Allo requires the program to be **strongly and statically typed**.
# The variable types are either declared explicitly or inferred from the context.
# For a variable that first appears in the program, we should declare it with an expected data type using Python's type hint notation:

a: int32

# %%
# Once the data types are defined, an important consideration is how to handle
# operations between variables of different types. Allo supports two types of casting:
# (1) implicit casting that is automatically done by the Allo compiler;
# and (2) explicit casting that is manually done by the user.

##############################################################################
# Implicit Casting
# ----------------
# Allo has a strong type system that follows the `MLIR convention <https://mlir.llvm.org/docs/Dialects/ArithOps/>`_ to enforce the operand types are the same for the arithmetic operations.
# However, it is burdensome for users to cast the variables every time, and it is also error-prone to avoid overflow when performing computations.
# Therefore, Allo is equipped with builtin casting rules to automatically cast the variables to the same type before the operation, which is called *implicit casting*.
# An example is shown below:


def add(a: int32, b: int32) -> int32:
    return a + b


s = allo.customize(add)
print(s.module)

# %%
# We can see that ``a`` and ``b`` are firstly casted to ``int33``, added
# together, and converted back to ``int32``.
# This is to avoid overflow and is automatically inferred by the Allo compiler.


##############################################################################
# Explicit Casting
# ----------------
# One can also explicitly cast the variable to a specific type by creating an intermediate variable,
# or use Python-builtin functions like ``float()`` and ``int()`` to explicitly cast a variable to ``float32`` or ``int32``.
# Another example is shown below:


def cast(a: int32) -> int16:
    b: float32 = a  # explicit
    c: float32 = b * 2
    d: float32 = float(a) * 2
    e: int16 = c + d
    return e


s = allo.customize(cast)
print(s.module)

# %%
# By explicitly creating an intermediate variable ``b``, we can cast the ``int32`` variable ``a`` to the desired floating-point type.
# Similarly, calling ``float(a)`` can also cast ``a`` to a floating-point type.
#
# .. note::
#
#    The above stated explicit casting between integers and floating points preserves the value but the precision may be changed.
#    If you want to use a union type to represent both integers and floating points, please use the `.bitcast()` API instead. For example, ``a.bitcast()`` can convert ``int32`` to ``float32`` representation with the bit pattern preserved.

##############################################################################
# Bit Operations
# --------------
# As hardware accelerators have ability to manipulate each bit of the data, Allo supports bit operations on
# those integer types. For example, we can access a specific bit in an integer ``a`` using the indexing operator:
#
# .. code-block:: python
#
#   a[15]

# %%
# We can also extract a chunk of bits from an integer using the slicing operator:
#
# .. code-block:: python
#
#   a[0:16]
#
# .. note::
#
#    Allo follows the Python convention that the upper bound is not included, so ``[0:16]`` means
#    extracting the first 16 bits, which is different from the Xilinx HLS convention that uses ``[0:15]``
#    to indicate the first 16 bits.

# %%
# Not only constant values are supported, but also variables can be used as the index or the slice range.
