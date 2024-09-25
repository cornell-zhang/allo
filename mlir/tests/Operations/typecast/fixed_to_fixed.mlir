// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --fixed-to-integer --lower-print-ops --jit | FileCheck %s
module {
  // 2.25, 2.50, 3.25, 6.25 
  memref.global "private" @fixed_gv : memref<2x2xi64> = dense<[[9, 10], [13, 25]]>
  func.func @top() -> () {
    %0 = hcl.get_global_fixed @fixed_gv : memref<2x2x!hcl.Fixed<32,2>>
    %1 = memref.alloc() : memref<2x2x!hcl.Fixed<16,1>>
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %3 = affine.load %0[%arg0, %arg1] : memref<2x2x!hcl.Fixed<32,2>>
        %4 = hcl.fixed_to_fixed (%3) : !hcl.Fixed<32, 2> -> !hcl.Fixed<16, 1>
        affine.store %4, %1[%arg0, %arg1] : memref<2x2x!hcl.Fixed<16,1>>
      }
    }
    hcl.print_memref(%1) {format = "%.2f \n"} : memref<2x2x!hcl.Fixed<16,1>> 
    return
  }
}

// CHECK: 2,   2.5
// CHECK: 3,   6