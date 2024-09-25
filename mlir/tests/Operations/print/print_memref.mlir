// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --lower-print-ops --jit | FileCheck %s
// Get bit 0,1,2 from a integer, the output should be 3
module {
  memref.global "private" @gv0 : memref<1xi32> = dense<[11]>
  func.func @top() -> () attributes {bit, itypes = "s", otypes = "s", top} {
    %0 = memref.get_global @gv0 : memref<1xi32>
    %res =memref.alloc() : memref<1xi32>
    affine.for %arg1 = 0 to 1 {
      %1 = affine.load %0[%arg1] : memref<1xi32>
      %c1_i32 = arith.constant 1 : i32
      %low = arith.constant 0 : index
      %high = arith.constant 2 : index
      %3 = hcl.get_slice(%1 : i32, %high, %low) -> i3
      // CHECK: 3
      hcl.print(%3) {format="%d\n"} : i3
      %4 = arith.extui %3 : i3 to i32
      affine.store %4, %res[%arg1] : memref<1xi32>
    } 
    hcl.print_memref(%res) : memref<1xi32>
    return
  }
}