// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt --lower-print-ops --jit %s | FileCheck %s

module {

  memref.global "private" @gv0 : memref<4x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]>

  func.func @top() -> () {
    %0 = memref.get_global @gv0 : memref<4x4xf32>
// CHECK: 1,   2,   3,   4
// CHECK: 1,   2,   3,   4
// CHECK: 1,   2,   3,   4
// CHECK: 1,   2,   3,   4
    hcl.print_memref(%0) : memref<4x4xf32>
    return
  }
}
