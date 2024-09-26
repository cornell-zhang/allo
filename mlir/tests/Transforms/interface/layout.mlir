// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt -opt %s | FileCheck %s

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
                    // CHECK: affine.load %[[ARG3:.*]][%[[ARG1:.*]], %[[ARG2:.*]]] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k", reduction = 1 : i32}
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        allo.reform(%B : memref<512x1024xf32>) {layout=affine_map<(d0,d1)->(d1,d0)>} -> memref<1024x512xf32>
        return %C : memref<1024x1024xf32>
    }
}