// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt --lower-composite --fixed-to-integer --lower-to-llvm %s
module {

  func.func @basic () -> () {
    %1 = arith.constant 0 : i32
    %2 = arith.constant 1 : i32
    %3 = allo.struct_construct(%1, %2) : i32, i32 -> !allo.struct<i32, i32>
    %4 = allo.struct_get %3[0] : !allo.struct<i32, i32> -> i32
    %5 = allo.struct_get %3[1] : !allo.struct<i32, i32> -> i32
    %6 = arith.addi %4, %5 : i32
    return
  }

  func.func @nested_struct() -> () {
    %1 = arith.constant 0 : i32
    %2 = arith.constant 1 : i32
    %3 = allo.struct_construct(%1, %2) : i32, i32 -> !allo.struct<i32, i32>
    %4 = allo.struct_construct(%3, %2) : !allo.struct<i32, i32>, i32 -> !allo.struct<!allo.struct<i32, i32>, i32>
    %5 = allo.struct_get %4[0] : !allo.struct<!allo.struct<i32, i32>, i32> -> !allo.struct<i32, i32>
    %6 = allo.struct_get %5[0] : !allo.struct<i32, i32> -> i32
    %7 = arith.addi %6, %2 : i32
    return
  }

  func.func @struct_memref() -> () {
    %1 = arith.constant 0 : i32
    %2 = arith.constant 1 : i32
    %3 = allo.struct_construct(%1, %2) : i32, i32 -> !allo.struct<i32, i32>
    %4 = memref.alloc() : memref<2x2x!allo.struct<i32, i32>>
    return
  }


  func.func @top(%arg0: memref<100xi8>, %arg1: memref<100x!allo.Fixed<13, 11>>, %arg2: memref<100xf32>) attributes {itypes = "s__", otypes = ""} {
    %0 = memref.alloc() {name = "compute_3"} : memref<100x!allo.struct<i8, !allo.Fixed<13, 11>, f32>>
    affine.for %arg3 = 0 to 100 {
      %2 = affine.load %arg0[%arg3] {from = "compute_0"} : memref<100xi8>
      %3 = affine.load %arg1[%arg3] {from = "compute_1"} : memref<100x!allo.Fixed<13, 11>>
      %4 = affine.load %arg2[%arg3] {from = "compute_2"} : memref<100xf32>
      %5 = allo.struct_construct(%2, %3, %4) : i8, !allo.Fixed<13, 11>, f32 -> <i8, !allo.Fixed<13, 11>, f32>
      affine.store %5, %0[%arg3] {to = "compute_3"} : memref<100x!allo.struct<i8, !allo.Fixed<13, 11>, f32>>
    } {loop_name = "x", op_name = "compute_3"}
    %1 = memref.alloc() {name = "compute_4"} : memref<100xi8>
    affine.for %arg3 = 0 to 100 {
      %2 = affine.load %0[%arg3] {from = "compute_3"} : memref<100x!allo.struct<i8, !allo.Fixed<13, 11>, f32>>
      %3 = allo.struct_get %2[0] : <i8, !allo.Fixed<13, 11>, f32> -> i8
      affine.store %3, %1[%arg3] {to = "compute_4"} : memref<100xi8>
    } {loop_name = "x", op_name = "compute_4"}
    return
  }
}