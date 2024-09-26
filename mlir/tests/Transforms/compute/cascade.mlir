// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt -opt %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0 * 16)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d1 + d0)>
// CHECK: #map2 = affine_map<(d0) -> (d0 * 2)>
// CHECK: #map3 = affine_map<(d0) -> (d0 * 4)>
// CHECK: #map4 = affine_map<(d0) -> (d0 * 8)>
module {
    func.func @gemm(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>) -> memref<1024x1024xf32>
    {
        %C = memref.alloc() : memref<1024x1024xf32>
        // CHECK: affine.for %[[ARG:.*]] = 0 to 1024 {
        affine.for %i = 0 to 1024 {
            // CHECK: affine.for %[[ARG1:.*]] = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                // CHECK: affine.for %[[ARG2:.*]] = 0 to 512 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k", reduction = 1 : i32}
            } { loop_name = "j" }
        } { loop_name = "i", op_name = "s" }
        return %C : memref<1024x1024xf32>
    }
    func.func @matrix_multiply(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %s = allo.create_op_handle "s"
        %l1 = allo.create_loop_handle %s, "i"
        %l2 = allo.create_loop_handle %s, "j"
        %l3 = allo.create_loop_handle %s, "k"
        // CHECK: affine.for %[[ARG:.*]] = 0 to 128 {
        // CHECK:   affine.for %[[ARG1:.*]] = 0 to 8 {
        affine.for %i = 0 to 1024 {
            // CHECK: affine.for %[[ARG2:.*]] = 0 to 32 {
            // CHECK: affine.for %[[ARG3:.*]] = 0 to 16 {
            // CHECK: affine.for %[[ARG4:.*]] = 0 to 128 {
            affine.for %j = 0 to 1024 {
                // CHECK: affine.for %[[ARG5:.*]] = 0 to 2 {
                // CHECK: affine.for %[[ARG6:.*]] = 0 to 4 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", op_name = "s" }
        %l4, %l5 = allo.split (%l1, 8)
        %l6, %l7, %l8, %l9 = allo.tile (%l2, %l3, 2, 4) // split & tile
        %l10, %l11 = allo.split (%l6, 16)
        return
    }
}