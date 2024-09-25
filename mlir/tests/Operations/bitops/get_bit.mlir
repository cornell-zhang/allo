// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --lower-print-ops --jit 
module {
  memref.global "private" @gv0 : memref<10xi8> = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]>
  func.func @top() -> () attributes {bit, itypes = "s", otypes = "s", top} {
    %0 = memref.get_global @gv0 : memref<10xi8>
    affine.for %arg1 = 0 to 10 {
      %1 = affine.load %0[%arg1] : memref<10xi8>
      affine.for %arg2 = 0 to 8 {
        %3 = hcl.get_bit(%1 : i8, %arg2) -> i1
        hcl.print(%3) {format="d", unsigned} : i1
      }
    } 
    // LSB, MSB 
    // 00000000
    // 10000000
    // 01000000
    // 11000000
    // 00100000
    // 10100000
    // 01100000
    // 11100000
    // 00010000
    // 10010000
    return
  }
}
