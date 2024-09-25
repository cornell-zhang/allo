// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func.func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                     %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                     %arg4: memref<10x10xf32>) {
        %s1 = hcl.create_op_handle "s1"
        %s2 = hcl.create_op_handle "s2"
        %l1 = hcl.create_loop_handle %s1, "i"
        %l2 = hcl.create_loop_handle %s1, "k"
        %l3 = hcl.create_loop_handle %s2, "i1"
        %l4 = hcl.create_loop_handle %s2, "k1"
        // CHECK: affine.for %arg5 = 0 to 3 {
        // CHECK:   affine.for %arg6 = 0 to 3 {
        affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
            %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
            %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
            %2 = arith.mulf %0, %1 : f32
            affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
            } {loop_name = "k"}
        } {loop_name = "i", op_name = "s1"}
        affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
            %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
            %1 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
            %2 = arith.addf %0, %1 : f32
            affine.store %2, %arg4[%arg5, %arg6] : memref<10x10xf32>
            } {loop_name = "k1"}
        } {loop_name = "i1", op_name = "s2"}
        hcl.compute_at (%s1, %s2, %l4)
        return
    }
    func.func @matrix_multiply( %A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>, %D: memref<1024x1024xf32>)
    {
        %s1 = hcl.create_op_handle "s1"
        %s2 = hcl.create_op_handle "s2"
        %l1 = hcl.create_loop_handle %s1, "i"
        %l2 = hcl.create_loop_handle %s1, "j"
        %l3 = hcl.create_loop_handle %s1, "k"
        %l4 = hcl.create_loop_handle %s2, "i1"
        %l5 = hcl.create_loop_handle %s2, "j1"
        %l6 = hcl.create_loop_handle %s2, "k1"
        // C=A*B
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", op_name = "s1" }
        // D=C*A
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 1024 {
                    %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
                    %c = affine.load %C[%k, %j] : memref<1024x1024xf32>
                    %d = affine.load %D[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %c : f32
                    %sum = arith.addf %prod, %d: f32
                    affine.store %sum, %D[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k1" }
            } { loop_name = "j1" }
        } { loop_name = "i1", op_name = "s2" }
        hcl.compute_at (%s1, %s2, %l6)
        return
    }
    func.func @no_load() -> (memref<10x10xi32>, memref<10x10xi32>) {
        %3 = hcl.create_op_handle "A"
        %0 = hcl.create_loop_handle %3, "y"
        %1 = hcl.create_loop_handle %3, "x"
        %2 = memref.alloc() {name = "A"} : memref<10x10xi32>
        affine.for %arg0 = 0 to 10 {
            affine.for %arg1 = 0 to 10 {
                %8 = arith.addi %arg0, %arg1 : index
                %9 = arith.index_cast %8 : index to i32
                affine.store %9, %2[%arg0, %arg1] {to = "A"} : memref<10x10xi32>
            } {loop_name = "x"}
        } {loop_name = "y", op_name = "A"}
        %7 = hcl.create_op_handle "B"
        %4 = hcl.create_loop_handle %7, "y"
        %5 = hcl.create_loop_handle %7, "x"
        %6 = memref.alloc() {name = "B"} : memref<10x10xi32>
        affine.for %arg0 = 0 to 10 {
            affine.for %arg1 = 0 to 10 {
                %8 = arith.subi %arg0, %arg1 : index
                %9 = arith.index_cast %8 : index to i32
                affine.store %9, %6[%arg0, %arg1] {to = "B"} : memref<10x10xi32>
            } {loop_name = "x"}
        } {loop_name = "y", op_name = "B"}
        hcl.compute_at(%3, %7, %5)
        return %2, %6 : memref<10x10xi32>, memref<10x10xi32>
    }
}