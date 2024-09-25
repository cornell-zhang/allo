// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --fixed-to-integer --lower-print-ops --jit | FileCheck %s
module {
  memref.global "private" @fixed_gv : memref<2x2xi64> = dense<[[8, 0], [10, 20]]>
  func.func @top() -> () {
    %0 = hcl.get_global_fixed @fixed_gv : memref<2x2x!hcl.Fixed<32,2>>
    %1 = memref.alloc() : memref<2x2xf32>
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %3 = affine.load %0[%arg0, %arg1] : memref<2x2x!hcl.Fixed<32,2>>
        %4 = hcl.fixed_to_float (%3) : !hcl.Fixed<32, 2> -> f32
        affine.store %4, %1[%arg0, %arg1] : memref<2x2xf32>
      }
    }
    hcl.print_memref(%1) : memref<2x2xf32> 
    return
  }
}

// CHECK: 2,   0
// CHECK: 2.5,   5