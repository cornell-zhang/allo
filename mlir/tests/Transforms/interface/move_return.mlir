// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --return-to-input
module {
  func.func @top(%arg0: memref<10xi32>) -> (memref<10xi32>, memref<10xi32>) attributes {itypes = "s", otypes = "ss", top} {
    %0 = memref.alloc() {name = "compute_1"} : memref<10xi32>
    affine.for %arg1 = 0 to 10 {
      %2 = affine.load %arg0[%arg1] {from = "compute_0"} : memref<10xi32>
      %c1_i32 = arith.constant 1 : i32
      %3 = arith.addi %2, %c1_i32 : i32
      affine.store %3, %0[%arg1] {to = "compute_1"} : memref<10xi32>
    } {loop_name = "x", op_name = "compute_1"}
    %1 = memref.alloc() {name = "compute_2"} : memref<10xi32>
    affine.for %arg1 = 0 to 10 {
      %2 = affine.load %arg0[%arg1] {from = "compute_0"} : memref<10xi32>
      %c2_i32 = arith.constant 2 : i32
      %3 = arith.addi %2, %c2_i32 : i32
      affine.store %3, %1[%arg1] {to = "compute_2"} : memref<10xi32>
    } {loop_name = "x", op_name = "compute_2"}
    return %0, %1 : memref<10xi32>, memref<10xi32>
  }
}