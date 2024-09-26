// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt %s --fixed-to-integer --lower-print-ops --jit | FileCheck %s
module {
  memref.global "private" @fixed_gv : memref<2x2xi64> = dense<[[8, 0], [10, 20]]>
  func.func @top() -> () {
    %0 = allo.get_global_fixed @fixed_gv : memref<2x2x!allo.Fixed<32,2>>
    %1 = memref.alloc() : memref<2x2xi64>
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 2 {
        %3 = affine.load %0[%arg0, %arg1] : memref<2x2x!allo.Fixed<32,2>>
        %4 = allo.fixed_to_int (%3) : !allo.Fixed<32, 2> -> i64
        affine.store %4, %1[%arg0, %arg1] : memref<2x2xi64>
      }
    }
    allo.print_memref(%1) : memref<2x2xi64> 
    return
  }
}

// CHECK: 2,   0
// CHECK: 2,   5