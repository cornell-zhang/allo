// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt %s --fixed-to-integer | FileCheck %s
module {

  func.func @issue_56(%arg0: memref<1000x!allo.Fixed<8, 6>>) -> memref<1000x!allo.Fixed<8, 6>> attributes {itypes = "_", otypes = "_", llvm.emit_c_interface, top} {
    %0 = memref.alloc() {name = "compute_1"} : memref<1000x!allo.Fixed<8, 6>>
    affine.for %arg1 = 0 to 1000 {
      %2 = affine.load %arg0[%arg1] {from = "compute_0"} : memref<1000x!allo.Fixed<8, 6>>
      affine.store %2, %0[%arg1] {to = "compute_1"} : memref<1000x!allo.Fixed<8, 6>>
    } {loop_name = "x", op_name = "compute_1"}
    %1 = memref.alloc() {name = "compute_2"} : memref<1000x!allo.Fixed<8, 6>>
    affine.for %arg1 = 0 to 1000 {
      %2 = affine.load %0[%arg1] {from = "compute_1"} : memref<1000x!allo.Fixed<8, 6>>
      affine.store %2, %1[%arg1] {to = "compute_2"} : memref<1000x!allo.Fixed<8, 6>>
    } {loop_name = "x", op_name = "compute_2"}
    return %1 : memref<1000x!allo.Fixed<8, 6>>
  }

  func.func @func_call(%arg0: memref<10xi32>, %arg1: memref<10xi32>) attributes {itypes = "ss", otypes = ""} {
    affine.for %arg2 = 0 to 10 {
      affine.for %arg3 = 0 to 10 {
        func.call @Stage_update_B(%arg0, %arg1, %arg3) {inputs = "compute_0,compute_1,"} : (memref<10xi32>, memref<10xi32>, index) -> ()
      } {loop_name = "loop_1"}
    } {loop_name = "loop_0"}
    return
  }
  func.func @Stage_update_B(%arg0: memref<10xi32>, %arg1: memref<10xi32>, %arg2: index) attributes {itypes = "sss"} {
    %0 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<10xi32>
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.addi %0, %c1_i32 : i32
    affine.store %1, %arg1[%arg2] {to = "compute_1"} : memref<10xi32>
    return
  }


  func.func @no_return(%arg0: memref<10x!allo.Fixed<32, 2>>, %arg1: memref<10x!allo.Fixed<32, 2>>, %arg3: memref<10x!allo.Fixed<32, 2>>) -> () {
    affine.for %arg2 = 0 to 10 {
      %1 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<10x!allo.Fixed<32, 2>>
      %2 = affine.load %arg1[%arg2] {from = "compute_1"} : memref<10x!allo.Fixed<32, 2>>
      %3 = "allo.add_fixed"(%1, %2) : (!allo.Fixed<32, 2>, !allo.Fixed<32, 2>) -> !allo.Fixed<32, 2>
      affine.store %3, %arg3[%arg2] {to = "compute_2"} : memref<10x!allo.Fixed<32, 2>>
    } {loop_name = "x", op_name = "compute_2"}
    return
  }

  func.func @top_vadd(%arg0: memref<10x!allo.Fixed<32, 2>>, %arg1: memref<10x!allo.Fixed<32, 2>>) -> memref<10x!allo.Fixed<32, 2>> {
    %0 = memref.alloc() {name = "compute_2"} : memref<10x!allo.Fixed<32, 2>>
    affine.for %arg2 = 0 to 10 {
      %1 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<10x!allo.Fixed<32, 2>>
      %2 = affine.load %arg1[%arg2] {from = "compute_1"} : memref<10x!allo.Fixed<32, 2>>
      %3 = "allo.add_fixed"(%1, %2) : (!allo.Fixed<32, 2>, !allo.Fixed<32, 2>) -> !allo.Fixed<32, 2>
      affine.store %3, %0[%arg2] {to = "compute_2"} : memref<10x!allo.Fixed<32, 2>>
    } {loop_name = "x", op_name = "compute_2"}
    return %0 : memref<10x!allo.Fixed<32, 2>>
  }


  func.func @top_vmul(%arg0: memref<10x!allo.Fixed<32, 2>>, %arg1: memref<10x!allo.Fixed<32, 2>>) -> memref<10x!allo.Fixed<32, 2>> {
    %0 = memref.alloc() {name = "compute_2"} : memref<10x!allo.Fixed<32, 2>>
    affine.for %arg2 = 0 to 10 {
      %1 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<10x!allo.Fixed<32, 2>>
      %2 = affine.load %arg1[%arg2] {from = "compute_1"} : memref<10x!allo.Fixed<32, 2>>
      %3 = "allo.mul_fixed"(%1, %2) : (!allo.Fixed<32, 2>, !allo.Fixed<32, 2>) -> !allo.Fixed<32, 2>
      affine.store %3, %0[%arg2] {to = "compute_2"} : memref<10x!allo.Fixed<32, 2>>
    } {loop_name = "x", op_name = "compute_2"}
    return %0 : memref<10x!allo.Fixed<32, 2>>
  }

  func.func @no_change_int(%arg0: memref<10xi32>) -> memref<10xi32> attributes {itypes = "s", otypes = "s"} {
    %0 = memref.alloc() {name = "compute_1"} : memref<10xi32>
    affine.for %arg1 = 0 to 10 {
      %1 = affine.load %arg0[%arg1] {from = "compute_0"} : memref<10xi32>
      %c1_i32 = arith.constant 1 : i32
      %2 = arith.addi %1, %c1_i32 : i32
      affine.store %2, %0[%arg1] {to = "compute_1"} : memref<10xi32>
    } {loop_name = "x", op_name = "compute_1"}
    return %0 : memref<10xi32>
  }
  func.func @no_change_float(%arg0: memref<10xf32>) -> memref<10xf32> attributes {itypes = "_", otypes = "_"} {
    %0 = memref.alloc() {name = "compute_1"} : memref<10xf32>
    affine.for %arg1 = 0 to 10 {
      %1 = affine.load %arg0[%arg1] {from = "compute_0"} : memref<10xf32>
      %cst = arith.constant 5.000000e-01 : f32
      %2 = arith.cmpf ogt, %1, %cst : f32
      %3 = affine.load %arg0[%arg1] {from = "compute_0"} : memref<10xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %4 = arith.select %2, %3, %cst_0 : f32
      affine.store %4, %0[%arg1] {to = "compute_1"} : memref<10xf32>
    } {loop_name = "x", op_name = "compute_1"}
    return %0 : memref<10xf32>
  }

  func.func @fixed_div(%arg0: memref<100x!allo.Fixed<6, 2>>, %arg1: memref<100x!allo.Fixed<6, 2>>) -> memref<100x!allo.Fixed<6, 2>> attributes {itypes = "__", otypes = "_"} {
    %0 = memref.alloc() {name = "compute_2"} : memref<100x!allo.Fixed<6, 2>>
    affine.for %arg2 = 0 to 100 {
      %1 = affine.load %arg0[%arg2] {from = "compute_0"} : memref<100x!allo.Fixed<6, 2>>
      %2 = affine.load %arg1[%arg2] {from = "compute_1"} : memref<100x!allo.Fixed<6, 2>>
      %3 = "allo.div_fixed"(%1, %2) : (!allo.Fixed<6, 2>, !allo.Fixed<6, 2>) -> !allo.Fixed<6, 2>
      affine.store %3, %0[%arg2] {to = "compute_2"} : memref<100x!allo.Fixed<6, 2>>
    } {loop_name = "x", op_name = "compute_2"}
    return %0 : memref<100x!allo.Fixed<6, 2>>
  }

  func.func @select_op(%arg0: memref<10x!allo.Fixed<8, 4>>, %arg1: memref<10x!allo.Fixed<8, 4>>) attributes {itypes = "__", otypes = ""} {
    affine.for %arg2 = 0 to 10 {
      %0 = affine.load %arg0[%arg2] {from = "tensor_0"} : memref<10x!allo.Fixed<8, 4>>
      %c0_i32 = arith.constant 0 : i32
      %1 = allo.fixed_to_fixed(%0) : !allo.Fixed<8, 4> -> !allo.Fixed<36, 4>
      %2 = allo.int_to_fixed(%c0_i32) : i32 -> !allo.Fixed<36, 4>
      %3 = allo.cmp_fixed sgt, %1, %2 : !allo.Fixed<36, 4>
      %4 = affine.load %arg0[%arg2] {from = "tensor_0"} : memref<10x!allo.Fixed<8, 4>>
      %5 = allo.int_to_fixed(%c0_i32) : i32 -> !allo.Fixed<8, 4>
      %6 = arith.select %3, %4, %5 : !allo.Fixed<8, 4> // CHECK: i8
      affine.store %6, %arg1[%arg2] {to = "tensor_1"} : memref<10x!allo.Fixed<8, 4>>
    } {loop_name = "x", op_name = "tensor_1"}
    return
  }

  func.func @foo(%arg0: i32) -> i32 attributes {itypes = "s", otypes = "s"} {
    %0 = arith.extsi %arg0 : i32 to i33
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.extsi %c1_i32 : i32 to i33
    %2 = arith.addi %0, %1 : i33
    %3 = arith.trunci %2 : i33 to i32
    return %3 : i32
  }
  func.func @issue_193(%arg0: memref<10xi32>) -> memref<10xi32> attributes {itypes = "s", otypes = "s"} {
    %alloc = memref.alloc() {name = "B"} : memref<10xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<10xi32>)
    affine.for %arg1 = 0 to 10 {
      %0 = affine.load %arg0[%arg1] {from = "A"} : memref<10xi32>
      %1 = func.call @foo(%0) : (i32) -> i32
      affine.store %1, %alloc[%arg1] {to = "B"} : memref<10xi32>
    } {loop_name = "i", op_name = "S_i_0"}
    return %alloc : memref<10xi32>
  }

  func.func @issue_194(%arg0: !allo.Fixed<8, 3>) -> i32 attributes {itypes = "s", otypes = "s"} {
    %alloc = memref.alloc() {name = "B"} : memref<1x!allo.Fixed<8, 3>>
    %c0_i32 = arith.constant 0 : i32
    %0 = allo.int_to_fixed(%c0_i32) : i32 -> !allo.Fixed<8, 3>
    affine.store %0, %alloc[0] {to = "B"} : memref<1x!allo.Fixed<8, 3>>
    %1 = affine.load %alloc[0] {from = "B"} : memref<1x!allo.Fixed<8, 3>>
    %2 = allo.cmp_fixed ugt, %arg0, %1 : !allo.Fixed<8, 3>
    %3 = allo.fixed_to_fixed(%arg0) : !allo.Fixed<8, 3> -> !allo.Fixed<35, 3>
    %c0_i32_0 = arith.constant 0 : i32
    %4 = allo.int_to_fixed(%c0_i32_0) : i32 -> !allo.Fixed<35, 3>
    %5 = allo.cmp_fixed ugt, %3, %4 : !allo.Fixed<35, 3>
    %6 = arith.andi %2, %5 : i1
    scf.if %6 {
      affine.store %arg0, %alloc[0] {to = "B"} : memref<1x!allo.Fixed<8, 3>>
    }
    %7 = affine.load %alloc[0] {from = "B"} : memref<1x!allo.Fixed<8, 3>>
    %8 = allo.fixed_to_int(%7) : !allo.Fixed<8, 3> -> i32
    return %8 : i32
  }

  func.func @callee(%arg0: f32, %arg1: f32) -> f32 attributes {itypes = "__", otypes = "_"} {
    %0 = arith.addf %arg0, %arg1 : f32
    %alloc = memref.alloc() {name = "c"} : memref<1xf32>
    affine.store %0, %alloc[0] {to = "c"} : memref<1xf32>
    %1 = affine.load %alloc[0] {from = "c"} : memref<1xf32>
    %2 = affine.load %alloc[0] {from = "c"} : memref<1xf32>
    return %2 : f32
  }

  func.func @test_scalar_result_func_calls(%arg0: memref<10xf32>, %arg1: memref<10xf32>) -> memref<10xf32> attributes {itypes = "__", otypes = "_"} {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32 : i32 to f32
    %alloc = memref.alloc() {name = "C"} : memref<10xf32>
    linalg.fill ins(%0 : f32) outs(%alloc : memref<10xf32>)
    affine.for %arg2 = 0 to 10 {
      %1 = affine.load %arg0[%arg2] {from = "A"} : memref<10xf32>
      %2 = affine.load %arg1[%arg2] {from = "B"} : memref<10xf32>
      %3 = func.call @callee(%1, %2) : (f32, f32) -> f32
      affine.store %3, %alloc[%arg2] {to = "C"} : memref<10xf32>
    } {loop_name = "i", op_name = "S_i_0"}
    return %alloc : memref<10xf32>
  }
}