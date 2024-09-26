// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt -opt %s | FileCheck %s

// CHECK: #map = affine_map<(d0, d1) -> (d0 mod 4, d1 mod 4, d0 floordiv 4, d1 floordiv 4)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0 floordiv 256, d1 floordiv 512, d0 mod 256, d1 mod 512)>
// CHECK: #map2 = affine_map<(d0, d1) -> (d0, 0, 0, d1)>
module {
    func.func @matrix_multiply(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32>
    {
        %s = allo.create_op_handle "s"
        %l1 = allo.create_loop_handle %s, "i"
        %l2 = allo.create_loop_handle %s, "j"
        %l3 = allo.create_loop_handle %s, "k"
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
        } { loop_name = "i", op_name = "s" }
        allo.partition(%A: memref<1024x1024xf32>, "CyclicPartition", 0, 4)
        allo.partition(%B: memref<1024x1024xf32>, "BlockPartition", 2, 2)
        allo.partition(%B: memref<1024x1024xf32>, "BlockPartition", 1, 4)
        allo.partition(%C: memref<1024x1024xf32>, "CompletePartition", 1)
        // CHECK: return %arg2 : memref<1024x1024xf32, #map2>
        return %C : memref<1024x1024xf32>
    }
}