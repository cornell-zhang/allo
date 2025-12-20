# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import int32, stateful, uint8

MEM_SIZE = 4
OP_H2D = 0
OP_D2H = 1
OP_ADD = 2
OP_MUL = 3


def int32_add(op1: int32, op2: int32) -> int32:
    return op1 + op2


def int32_mul(op1: int32, op2: int32) -> int32:
    return op1 * op2


def test_tpu(op: uint8, inval: int32, addr: uint8) -> int32:
    mem: stateful(int32[MEM_SIZE]) = 0
    retval: int32
    if op == OP_H2D:
        mem[addr] = inval
        retval = 99  # random value
    if op == OP_D2H:
        retval = mem[addr]
    if op == OP_ADD:
        mem[addr] = int32_add(mem[addr], mem[addr + 1])
        retval = mem[addr]
    if op == OP_MUL:
        mem[addr] = int32_mul(mem[addr], mem[addr + 1])
        retval = mem[addr]
    return retval


s = allo.customize(test_tpu)
# x = s.build(target="llvm")

mod = s.build(target="vitis_hls", mode="sw_emu", project="/tmp/tpu.prj")
# print(mod)
