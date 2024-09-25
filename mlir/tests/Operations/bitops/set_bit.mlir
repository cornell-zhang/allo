// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --lower-print-ops --jit | FileCheck %s
// Input: 0x0000
// By setting the third bit to 1, we get
// Output: 0x0004
module {
  memref.global "private" @gv0 : memref<1xi32> = dense<[0]>
  func.func @top() -> () attributes {bit, itypes = "s", otypes = "s", top} {
    %0 = memref.get_global @gv0 : memref<1xi32>
    %res =memref.alloc() : memref<1xi32>
    affine.for %arg1 = 0 to 1 {
      %1 = affine.load %0[%arg1] : memref<1xi32>
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %val = arith.constant 1 : i1
      %2 = hcl.set_bit(%1 : i32, %c2, %val : i1) -> i32
      affine.store %2, %res[%arg1] : memref<1xi32>
    } 
// CHECK: 4
    %v = affine.load %res[0] : memref<1xi32>
    hcl.print(%v) {format="%d\n"}: i32
    return
  }
}