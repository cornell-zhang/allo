// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt --fixed-to-integer --lower-print-ops --jit %s | FileCheck %s
module {
  memref.global "private" @gv_cst : memref<2x2xi64> = dense<[[8, 0], [10, 20]]>

  func.func @top() -> () {
    %0 = allo.get_global_fixed @gv_cst : memref<2x2x!allo.Fixed<32,2>>
    allo.print_memref (%0) {format = "%.1f \n"} : memref<2x2x!allo.Fixed<32,2>> 
    return
  }
}

// CHECK:  2,   0 
// CHECK:  2.5,   5