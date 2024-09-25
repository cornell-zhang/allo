// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt -opt %s | FileCheck %s

module {
  func.func @top(%arg0: memref<10x32xi32>) -> memref<10x32xi32> attributes {itypes = "s", otypes = "s"} {
    %0 = memref.alloc() {name = "C"} : memref<10x32xi32>
    %3 = hcl.create_op_handle "C"
    %1 = hcl.create_loop_handle %3, "i"
    %2 = hcl.create_loop_handle %3, "j"
    affine.for %arg1 = 0 to 10 {
      affine.for %arg2 = 0 to 32 {
        %12 = affine.load %arg0[%arg1, %arg2] {from = "A"} : memref<10x32xi32>
        %c1_i32 = arith.constant 1 : i32
        %13 = arith.addi %12, %c1_i32 : i32
        affine.store %13, %0[%arg1, %arg2] {to = "C"} : memref<10x32xi32>
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "C"}
    %4 = memref.alloc() {name = "D"} : memref<10x32xi32>
    %7 = hcl.create_op_handle "D"
    %5 = hcl.create_loop_handle %7, "i"
    %6 = hcl.create_loop_handle %7, "j"
    affine.for %arg1 = 0 to 10 {
      affine.for %arg2 = 0 to 32 {
        %12 = affine.load %0[%arg1, %arg2] {from = "C"} : memref<10x32xi32>
        %c2_i32 = arith.constant 2 : i32
        %13 = arith.muli %12, %c2_i32 : i32
        affine.store %13, %4[%arg1, %arg2] {to = "D"} : memref<10x32xi32>
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "D"}
    %8 = memref.alloc() {name = "E"} : memref<10x32xi32>
    %11 = hcl.create_op_handle "E"
    %9 = hcl.create_loop_handle %11, "i"
    %10 = hcl.create_loop_handle %11, "j"
    affine.for %arg1 = 0 to 10 {
      affine.for %arg2 = 0 to 32 {
        %12 = affine.load %4[%arg1, %arg2] {from = "D"} : memref<10x32xi32>
        %c3_i32 = arith.constant 3 : i32
        %13 = arith.muli %12, %c3_i32 : i32
        affine.store %13, %8[%arg1, %arg2] {to = "E"} : memref<10x32xi32>
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "E"}
    // CHECK: call @Stage_C
    hcl.outline (%3)
    // CHECK: call @Stage_D_E
    hcl.outline (%7, %11)
    return %8 : memref<10x32xi32>
  }
  func.func @top2(%arg0: memref<10x32xi32>) -> memref<10x32xi32> attributes {itypes = "s", otypes = "s"} {
    %0 = memref.alloc() {name = "C1"} : memref<10x32xi32>
    %3 = hcl.create_op_handle "C1"
    %1 = hcl.create_loop_handle %3, "i"
    %2 = hcl.create_loop_handle %3, "j"
    affine.for %arg1 = 0 to 10 {
      affine.for %arg2 = 0 to 32 {
        %12 = affine.load %arg0[%arg1, %arg2] {from = "A1"} : memref<10x32xi32>
        %c1_i32 = arith.constant 1 : i32
        %13 = arith.addi %12, %c1_i32 : i32
        affine.store %13, %0[%arg1, %arg2] {to = "C1"} : memref<10x32xi32>
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "C1"}
    %4 = memref.alloc() {name = "D"} : memref<10x32xi32>
    %7 = hcl.create_op_handle "D1"
    %5 = hcl.create_loop_handle %7, "i"
    %6 = hcl.create_loop_handle %7, "j"
    affine.for %arg1 = 0 to 10 {
      affine.for %arg2 = 0 to 32 {
        %12 = affine.load %0[%arg1, %arg2] {from = "C1"} : memref<10x32xi32>
        %c2_i32 = arith.constant 2 : i32
        %13 = arith.muli %12, %c2_i32 : i32
        affine.store %13, %4[%arg1, %arg2] {to = "D1"} : memref<10x32xi32>
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "D1"}
    %8 = memref.alloc() {name = "E1"} : memref<10x32xi32>
    %11 = hcl.create_op_handle "E1"
    %9 = hcl.create_loop_handle %11, "i"
    %10 = hcl.create_loop_handle %11, "j"
    affine.for %arg1 = 0 to 10 {
      affine.for %arg2 = 0 to 32 {
        %12 = affine.load %4[%arg1, %arg2] {from = "D1"} : memref<10x32xi32>
        %c3_i32 = arith.constant 3 : i32
        %13 = arith.muli %12, %c3_i32 : i32
        affine.store %13, %8[%arg1, %arg2] {to = "E1"} : memref<10x32xi32>
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "E1"}
    // CHECK: call @Stage_C1
    hcl.outline (%3)
    // CHECK: call @Stage_C1
    hcl.outline (%7) {unify="Stage_C1"}
    // CHECK: affine.for %[[ARG:.*]] = 0 to 10 {
    // CHECK:   affine.for %[[ARG1:.*]] = 0 to 32 {
    // CHECK:     call @Stage_E1
    hcl.outline (%11) {axis = "j"}
    return %8 : memref<10x32xi32>
  }
}