// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt -opt %s | FileCheck %s

// Explanation: this test checks if the induction variable is correctly updated
// when the non-reduction loop bound is updated in a strided convolution.
module {
  func.func @top(%arg0: memref<1x1x16x16xi1>, %arg1: memref<16x1x3x3xi1>) -> memref<1x16x8x8xi8> attributes {itypes = "uu", otypes = "s"} {
    %0 = memref.alloc() {name = "conv1_pad"} : memref<1x1x18x18xi1>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        affine.for %arg4 = 0 to 18 {
          affine.for %arg5 = 0 to 18 {
            %true = arith.constant {unsigned} true
            %c1_i32 = arith.constant 1 : i32
            %8 = arith.index_cast %arg5 : index to i33
            %9 = arith.extsi %c1_i32 : i32 to i33
            %10 = arith.cmpi sge, %8, %9 : i33
            %11 = arith.andi %true, %10 {unsigned} : i1
            %c17_i32 = arith.constant 17 : i32
            %12 = arith.index_cast %arg5 : index to i33
            %13 = arith.extsi %c17_i32 : i32 to i33
            %14 = arith.cmpi slt, %12, %13 : i33
            %15 = arith.andi %11, %14 {unsigned} : i1
            %c1_i32_0 = arith.constant 1 : i32
            %16 = arith.index_cast %arg4 : index to i33
            %17 = arith.extsi %c1_i32_0 : i32 to i33
            %18 = arith.cmpi sge, %16, %17 : i33
            %19 = arith.andi %15, %18 {unsigned} : i1
            %c17_i32_1 = arith.constant 17 : i32
            %20 = arith.index_cast %arg4 : index to i33
            %21 = arith.extsi %c17_i32_1 : i32 to i33
            %22 = arith.cmpi slt, %20, %21 : i33
            %23 = arith.andi %19, %22 {unsigned} : i1
            %24 = affine.load %arg0[%arg2, %arg3, %arg4 - 1, %arg5 - 1] {from = "input", unsigned} : memref<1x1x16x16xi1>
            %c0_i32 = arith.constant 0 : i32
            %25 = arith.trunci %c0_i32 {unsigned} : i32 to i1
            %26 = arith.select %23, %24, %25 : i1
            affine.store %26, %0[%arg2, %arg3, %arg4, %arg5] {to = "conv1_pad", unsigned} : memref<1x1x18x18xi1>
          } {loop_name = "ww"}
        } {loop_name = "hh"}
      } {loop_name = "cc"}
    } {loop_name = "ii", op_name = "conv1_pad"}
    %1 = memref.alloc() {name = "conv1"} : memref<1x16x8x8xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %8 = memref.alloc() {name = "conv1_sum"} : memref<1xi8>
            %c0_i32 = arith.constant 0 : i32
            %c0 = arith.constant {unsigned} 0 : index
            %9 = arith.trunci %c0_i32 : i32 to i8
            affine.store %9, %8[0] {to = "conv1_sum"} : memref<1xi8>
            affine.for %arg6 = 0 to 3 {
              affine.for %arg7 = 0 to 3 {
                %true = arith.constant true
                scf.if %true {
                  %true_0 = arith.constant {unsigned} true
                  %c2_i32 = arith.constant 2 : i32
                  %11 = arith.index_cast %arg5 : index to i64
                  %12 = arith.extsi %c2_i32 : i32 to i64
                  %13 = arith.muli %11, %12 : i64
                  %c1_i32 = arith.constant 1 : i32
                  %14 = arith.index_cast %arg7 : index to i64
                  %15 = arith.extsi %c1_i32 : i32 to i64
                  %16 = arith.muli %14, %15 : i64
                  %17 = arith.extsi %13 : i64 to i65
                  %18 = arith.extsi %16 : i64 to i65
                  %19 = arith.addi %17, %18 : i65
                  %c1_i32_1 = arith.constant 1 : i32
                  %20 = arith.extsi %c1_i32_1 : i32 to i65
                  %21 = arith.cmpi sge, %19, %20 : i65
                  %22 = arith.andi %true_0, %21 {unsigned} : i1
                  %c2_i32_2 = arith.constant 2 : i32
                  %23 = arith.index_cast %arg5 : index to i64
                  %24 = arith.extsi %c2_i32_2 : i32 to i64
                  // CHECK: {{.*}} = arith.constant 2 : i64
                  // CHECK: {{.*}} = arith.subi {{.*}}, {{.*}} : i64
                  %25 = arith.muli %23, %24 : i64
                  %c1_i32_3 = arith.constant 1 : i32
                  %26 = arith.index_cast %arg7 : index to i64
                  %27 = arith.extsi %c1_i32_3 : i32 to i64
                  %28 = arith.muli %26, %27 : i64
                  %29 = arith.extsi %25 : i64 to i65
                  %30 = arith.extsi %28 : i64 to i65
                  %31 = arith.addi %29, %30 : i65
                  %c17_i32 = arith.constant 17 : i32
                  %32 = arith.extsi %c17_i32 : i32 to i65
                  %33 = arith.cmpi slt, %31, %32 : i65
                  %34 = arith.andi %22, %33 {unsigned} : i1
                  %c2_i32_4 = arith.constant 2 : i32
                  %35 = arith.index_cast %arg4 : index to i64
                  %36 = arith.extsi %c2_i32_4 : i32 to i64
                  // CHECK: {{.*}} = arith.constant 2 : i64
                  // CHECK: {{.*}} = arith.subi {{.*}}, {{.*}} : i64
                  %37 = arith.muli %35, %36 : i64
                  %c1_i32_5 = arith.constant 1 : i32
                  %38 = arith.index_cast %arg6 : index to i64
                  %39 = arith.extsi %c1_i32_5 : i32 to i64
                  %40 = arith.muli %38, %39 : i64
                  %41 = arith.extsi %37 : i64 to i65
                  %42 = arith.extsi %40 : i64 to i65
                  %43 = arith.addi %41, %42 : i65
                  %c1_i32_6 = arith.constant 1 : i32
                  %44 = arith.extsi %c1_i32_6 : i32 to i65
                  %45 = arith.cmpi sge, %43, %44 : i65
                  %46 = arith.andi %34, %45 {unsigned} : i1
                  %c2_i32_7 = arith.constant 2 : i32
                  %47 = arith.index_cast %arg4 : index to i64
                  %48 = arith.extsi %c2_i32_7 : i32 to i64
                  %49 = arith.muli %47, %48 : i64
                  %c1_i32_8 = arith.constant 1 : i32
                  %50 = arith.index_cast %arg6 : index to i64
                  %51 = arith.extsi %c1_i32_8 : i32 to i64
                  %52 = arith.muli %50, %51 : i64
                  %53 = arith.extsi %49 : i64 to i65
                  %54 = arith.extsi %52 : i64 to i65
                  %55 = arith.addi %53, %54 : i65
                  %c17_i32_9 = arith.constant 17 : i32
                  %56 = arith.extsi %c17_i32_9 : i32 to i65
                  %57 = arith.cmpi slt, %55, %56 : i65
                  %58 = arith.andi %46, %57 {unsigned} : i1
                  %c1_i32_10 = arith.constant 1 : i32
                  %59 = affine.load %0[%arg2, 0, %arg4 * 2 + %arg6, %arg5 * 2 + %arg7] {from = "conv1_pad", unsigned} : memref<1x1x18x18xi1>
                  %60 = arith.extsi %c1_i32_10 : i32 to i33
                  %61 = arith.extui %59 : i1 to i33
                  %62 = arith.subi %60, %61 : i33
                  %63 = affine.load %arg1[%arg3, 0, %arg6, %arg7] {from = "w_conv1", unsigned} : memref<16x1x3x3xi1>
                  %64 = arith.extui %63 : i1 to i33
                  %65 = arith.xori %62, %64 : i33
                  %c1_i32_11 = arith.constant 1 : i32
                  %66 = arith.extsi %c1_i32_11 : i32 to i33
                  %67 = arith.shli %65, %66 : i33
                  %c33_i33 = arith.constant 33 : i33
                  %68 = arith.cmpi sge, %66, %c33_i33 : i33
                  %c0_i33 = arith.constant 0 : i33
                  %69 = arith.select %68, %c0_i33, %67 : i33
                  %c1_i32_12 = arith.constant 1 : i32
                  %70 = arith.extui %69 : i33 to i34
                  %71 = arith.extsi %c1_i32_12 : i32 to i34
                  %72 = arith.subi %70, %71 : i34
                  %73 = arith.trunci %72 : i34 to i8
                  %c0_i32_13 = arith.constant 0 : i32
                  %74 = arith.trunci %c0_i32_13 : i32 to i8
                  %75 = arith.select %58, %73, %74 : i8
                  %76 = affine.load %8[0] {from = "conv1_sum"} : memref<1xi8>
                  %77 = arith.extsi %75 : i8 to i9
                  %78 = arith.extsi %76 : i8 to i9
                  %79 = arith.addi %77, %78 : i9
                  %80 = arith.trunci %79 : i9 to i8
                  affine.store %80, %8[0] {to = "conv1_sum"} : memref<1xi8>
                }
              } {loop_name = "rx", reduction}
            } {loop_name = "ry", reduction}
            %10 = affine.load %8[0] {from = "conv1_sum"} : memref<1xi8>
            affine.store %10, %1[%arg2, %arg3, %arg4, %arg5] {to = "conv1"} : memref<1x16x8x8xi8>
          } {loop_name = "xx"}
        } {loop_name = "yy"}
      } {loop_name = "ff"}
    } {loop_name = "nn", op_name = "conv1"}
    %2 = hcl.create_op_handle "conv1"
    %3 = hcl.create_loop_handle %2, "yy"
    %4 = hcl.reuse_at(%0 : memref<1x1x18x18xi1>, %3) -> memref<1xf32>
    %5 = hcl.create_op_handle "conv1"
    %6 = hcl.create_loop_handle %5, "xx"
    %7 = hcl.reuse_at(%4 : memref<1xf32>, %6) -> memref<1xf32>
    return %1 : memref<1x16x8x8xi8>
  }
}
