# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin, unused-argument

import numpy as np


def grid(*args, name=""):
    return np.ndindex(*args)


def reduction(*args, name=""):
    return np.ndindex(*args)


def matmul(lhs, rhs):
    return np.matmul(lhs, rhs)


def bmm(lhs, rhs):
    return np.einsum("ijk,ikn->ijn", lhs, rhs)


def add(lhs, rhs):
    return lhs + rhs


def sub(lhs, rhs):
    return lhs - rhs


def div(lhs, rhs):
    return lhs / rhs


def exp(x):
    return np.exp(x)


def log(x):
    return np.log(x)


def log2(x):
    return np.log2(x)


def log10(x):
    return np.log10(x)


def abs(x):
    return np.abs(x)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sqrt(x):
    return np.sqrt(x)


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def tan(x):
    return np.tan(x)


def tanh(x):
    return np.tanh(x)


def power(x, y):
    return np.power(x, y)
