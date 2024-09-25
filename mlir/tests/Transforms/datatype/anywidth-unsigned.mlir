// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --lower-anywidth-integer
module {
  func.func @top(%arg0: memref<10xi32>) -> memref<10xi2> attributes {itypes = "u", otypes = "u", llvm.emit_c_interface, top} {
    %0 = memref.alloc() {name = "B", unsigned} : memref<10xi2>
    affine.for %arg1 = 0 to 10 {
      %1 = affine.load %arg0[%arg1] {from = "compute_0", unsigned} : memref<10xi32>
      %c1_i32 = arith.constant 1 : i32
      %2 = arith.addi %1, %c1_i32 {unsigned} : i32
      %3 = arith.trunci %2 : i32 to i2
      affine.store %3, %0[%arg1] {to = "B"} : memref<10xi2>
    } {loop_name = "x", op_name = "B"}
    return %0 : memref<10xi2>
  }
}