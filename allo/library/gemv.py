"""Performs a matrix vector multiply."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unused-import, unsupported-assignment-operation, missing-function-docstring

from .. import dsl
from ..ir.types import int8, int16, int32, index, Int, UInt
from ..ir.utils import MockBuffer


def pe_kernel[
    L, D, P
](x_in: "int8[P, D//P*L]", y_in: "int8[P, D//P*L]", o: "int8[P, D//P]", p: "index"):
    # """Finds the dot product of D//P vectors within given column of inputs."""
    for d in dsl.grid(D // P, name="l_loop"):
        total: int8 = 0
        for l in dsl.grid(L):
            total += x_in[p, d * L + l] * y_in[p, d * L + l]
        o[p, d] = total


def int8xint8_mat_vec[
    L, D, P
](x_in: "Int(8 * P)[D // P, L]", y_in: "int8[L]", out: "int8[D]",):
    # """Gets the matrix vector multiply result."""
    x: int8[P, D // P * L]
    y_buff: int8[L]
    y: int8[P, D // P * L]
    o: int8[P, D // P]

    # load and unpack X-Values
    for d, l in dsl.grid(D // P, L, name="x_load"):
        x_in_tmp: Int(8 * P)
        x_in_tmp = x_in[d, l]
        for p in range(P):
            x[p, d * L + l] = x_in_tmp[p * 8 : p * 8 + 8]

    # buffer Y values
    for d, l in dsl.grid(D // P, L, name="y_load"):
        if d == 0:
            y_buff[l] = y_in[l]
        for p in range(P):
            y[p, d * L + l] = y_buff[l]

    # array of kernels
    for p in dsl.grid(P, name="D_L"):
        pe_kernel[L, D, P](x, y, o, p)

    # collect results
    for d, p in dsl.grid(D // P, P, name="o_store"):
        out[d * P + p] = o[p, d]


def schedule_int8xint8_mat_vec(s):
    # """Schedule matrix vector multiply."""
    assert s.top_func_name == "int8xint8_mat_vec"
    kernel_name = "pe_kernel"

    s.partition(s.x, dim=1)
    s.partition(s.y, dim=1)
    s.partition(s.o, dim=1)

    # Pipeline data access loops
    load_x_loop = s.get_loops(s.top_func_name)["x_load"]["l"]
    s.pipeline(load_x_loop, rewind=True)
    load_y_loop = s.get_loops(s.top_func_name)["y_load"]["l"]
    s.pipeline(load_y_loop, rewind=True)
    store_o_loop = s.get_loops(s.top_func_name)["o_store"]["p"]
    s.pipeline(store_o_loop, rewind=True)

    pe_loops = s.get_loops(kernel_name)["l_loop"]["l"]
    pe_loops.path = kernel_name
    s.pipeline(pe_loops, rewind=True)

    # Unfold tiles
    pe = s.unfold(f"{s.top_func_name}:D_L", [0])  # specify which are spatial loops
    s.to(MockBuffer(s.top_func_name, "x"), pe, axis=0, depth=4)
    s.to(MockBuffer(s.top_func_name, "y"), pe, axis=0, depth=4)
    s.to(MockBuffer(s.top_func_name, "o"), pe, axis=0, depth=4)

    return s
