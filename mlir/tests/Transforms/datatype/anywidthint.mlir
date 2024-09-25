// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --lower-anywidth-integer
module {
  func.func @top_vadd(%arg0: memref<10xi10>, %arg1: memref<10xi10>) -> memref<10xi10> attributes {"top"} {
    %0 = memref.alloc() {name = "compute_2"} : memref<10xi10>
    affine.for %arg2 = 0 to 10 {
      %1 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<10xi10>
      %2 = affine.load %arg1[%arg2] {from = "compute_1"} : memref<10xi10>
      %3 = arith.addi %1, %2 : i10
      affine.store %3, %0[%arg2] {to = "compute_2"} : memref<10xi10>
    } {loop_name = "x", op_name = "compute_2"}
    return %0 : memref<10xi10>
  }
}