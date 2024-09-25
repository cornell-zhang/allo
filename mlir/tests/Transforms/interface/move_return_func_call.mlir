// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --return-to-input
module {
  func.func @gemm(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi32>) -> memref<32x32xi32> {
    %0 = memref.alloc() {name = "C"} : memref<32x32xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%0 : memref<32x32xi32>)
    affine.for %arg2 = 0 to 32 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 32 {
          %1 = affine.load %arg0[%arg2, %arg4] {from = "A"} : memref<32x32xi32>
          %2 = affine.load %arg1[%arg4, %arg3] {from = "B"} : memref<32x32xi32>
          %3 = arith.muli %1, %2 : i32
          %4 = affine.load %0[%arg2, %arg3] {from = "C"} : memref<32x32xi32>
          %5 = arith.addi %4, %3 : i32
          affine.store %5, %0[%arg2, %arg3] {to = "C"} : memref<32x32xi32>
        } {loop_name = "k"}
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "S_i_j_k"}
    return %0 : memref<32x32xi32>
  }
  // CHECK-LABEL: func @top(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi32>, %arg2: memref<32x32xi32>) attributes {top} {
  func.func @top(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi32>) -> memref<32x32xi32> attributes {top} {
    %0 = call @gemm(%arg0, %arg1) : (memref<32x32xi32>, memref<32x32xi32>) -> memref<32x32xi32>
    // CHECK: memref.copy
    return %0 : memref<32x32xi32>
    // CHECK: return
  }
}