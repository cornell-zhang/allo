// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s
module {
  func.func @top(%arg0: memref<16x22xf32>, %arg1: memref<22x18xf32>, %arg2: memref<18x24xf32>, %arg3: memref<16x24xf32>) -> memref<16x24xf32> attributes {llvm.emit_c_interface} {
    %0 = memref.alloc() {name = "out_AB"} : memref<16x18xf32>
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 18 {
        %3 = memref.alloc() {name = "sum_rv"} : memref<1xf32>
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        affine.store %cst, %3[%c0] {to = "sum_rv"} : memref<1xf32>
        affine.for %arg6 = 0 to 22 {
          %5 = affine.load %arg0[%arg4, %arg6] {from = "A"} : memref<16x22xf32>
          %6 = affine.load %arg1[%arg6, %arg5] {from = "B"} : memref<22x18xf32>
          %7 = arith.mulf %5, %6 : f32
          %8 = affine.load %3[%c0] {from = "sum_rv"} : memref<1xf32>
          %9 = arith.addf %7, %8 : f32
          affine.store %9, %3[%c0] {to = "sum_rv"} : memref<1xf32>
        } {loop_name = "r"}
        %c0_0 = arith.constant 0 : index
        %4 = affine.load %3[%c0_0] {from = "sum_rv"} : memref<1xf32>
        affine.store %4, %0[%arg4, %arg5] {to = "out_AB"} : memref<16x18xf32>
      } {loop_name = "y"}
    } {loop_name = "x", op_name = "out_AB"}
    %1 = memref.alloc() {name = "out_ABC"} : memref<16x24xf32>
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 24 {
        %3 = memref.alloc() {name = "sum_rv"} : memref<1xf32>
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        affine.store %cst, %3[%c0] {to = "sum_rv"} : memref<1xf32>
        affine.for %arg6 = 0 to 18 {
          %5 = affine.load %0[%arg4, %arg6] {from = "out_AB"} : memref<16x18xf32>
          %6 = affine.load %arg2[%arg6, %arg5] {from = "C"} : memref<18x24xf32>
          %7 = arith.mulf %5, %6 : f32
          %8 = affine.load %3[%c0] {from = "sum_rv"} : memref<1xf32>
          %9 = arith.addf %7, %8 : f32
          affine.store %9, %3[%c0] {to = "sum_rv"} : memref<1xf32>
        } {loop_name = "k"}
        %c0_0 = arith.constant 0 : index
        %4 = affine.load %3[%c0_0] {from = "sum_rv"} : memref<1xf32>
        affine.store %4, %1[%arg4, %arg5] {to = "out_ABC"} : memref<16x24xf32>
      } {loop_name = "y"}
    } {loop_name = "x", op_name = "out_ABC"}
    %2 = memref.alloc() {name = "E"} : memref<16x24xf32>
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 24 {
        %cst = arith.constant 1.000000e-01 : f32
        %3 = affine.load %1[%arg4, %arg5] {from = "out_ABC"} : memref<16x24xf32>
        %4 = arith.mulf %cst, %3 : f32
        %cst_0 = arith.constant 1.000000e-01 : f32
        %5 = affine.load %arg3[%arg4, %arg5] {from = "D"} : memref<16x24xf32>
        %6 = arith.mulf %cst_0, %5 : f32
        %7 = arith.addf %4, %6 : f32
        affine.store %7, %2[%arg4, %arg5] {to = "E"} : memref<16x24xf32>
      } {loop_name = "y"}
    } {loop_name = "x", op_name = "E"}
    return %2 : memref<16x24xf32>
  }
}