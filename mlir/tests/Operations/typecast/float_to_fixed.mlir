// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --fixed-to-integer --lower-print-ops --jit | FileCheck %s
module {
  memref.global "private" @float_gv : memref<2x2xf32> = dense<[[2.234, 1.223], [5.261, 1.2]]>
  func.func @top() -> () {
    %0 = memref.get_global @float_gv : memref<2x2xf32>
    %1 = memref.alloc() : memref<2x2x!hcl.Fixed<32, 2>>
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %3 = affine.load %0[%arg0, %arg1] : memref<2x2xf32>
        %4 = hcl.float_to_fixed (%3) : f32 -> !hcl.Fixed<32, 2>
        affine.store %4, %1[%arg0, %arg1] : memref<2x2x!hcl.Fixed<32, 2>>
      }
    }
    hcl.print_memref(%1) : memref<2x2x!hcl.Fixed<32,2>> 
    return
  }
}

// CHECK: 2,   1
// CHECK: 5.25,   1