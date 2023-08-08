# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32
import numpy as np


def same(A: int32[32, 1]) -> int32[32, 1]:
    return A


def outzero() -> int32:
    C: int32 = 0
    return C


# test_td_outzero = """
# func.func @outzero() -> tensor<i32> {
#   %0 = tensor.generate  {
#     %c0_i32 = arith.constant 0 : i32
#     tensor.yield %c0_i32 :i32
#   } : tensor<i32>
#   return %0 : tensor<i32>
# }
# """
# mod = allo.invoke_mlir_parser(test_td_outzero)
# print(mod)


def half(A: int32[2, 2]) -> int32[1, 1]:
    return A[:1, :1]


s = allo.customize(outzero, verbose=True, enable_tensor=True)
print(s.module)


# test_td_half = """
# func.func @half(%A: tensor<32x32xi32>) -> tensor<16x16xi32> {
#    %01 = tensor.extract_slice %A[0, 0][16, 16][1, 1] : tensor<32x32xi32> to tensor<16x16xi32>
#    return %0 : tensor<16x16xi32>
#  }
# """
# mod = allo.invoke_mlir_parser(test_td_half)
# print(mod)

# test_td_same_1 = """
# func.func @same(%arg0: tensor<i32>) -> tensor<i32> {
#     return %arg0 : tensor<i32>
#   }
# """
# mod = allo.invoke_mlir_parser(test_td_same_1)
# print(mod)
