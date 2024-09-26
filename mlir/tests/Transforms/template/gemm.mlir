// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt -opt %s | FileCheck %s
module {
    // define a customization template
    allo.customization @gemm_opt(
        %AA: memref<?x?x!allo.Type>,
        %BB: memref<?x?x!allo.Type>,
        %CC: memref<?x?x!allo.Type>,
        %i: !allo.LoopHandle,
        %j: !allo.LoopHandle,
        %k: !allo.LoopHandle
    ) {
        allo.pipeline(%j, 1)
        allo.partition(%AA: memref<?x?x!allo.Type>, "CompletePartition", 2)
        allo.partition(%BB: memref<?x?x!allo.Type>, "CompletePartition", 2)
        allo.partition(%CC: memref<?x?x!allo.Type>, "CompletePartition", 2)
        allo.end
    }

    // CHECK: #map = affine_map<(d0, d1) -> (0, d1, d0, 0)>
    func.func @top(%A: memref<64x32xi32>, %B: memref<32x64xi32>, %C: memref<64x64xi32>) -> memref<64x64xi32>
    {   
        %s1 = allo.create_op_handle "s1"
        %i1 = allo.create_loop_handle %s1, "i1"
        %j1 = allo.create_loop_handle %s1, "j1"
        %k1 = allo.create_loop_handle %s1,  "k1"
        // D = A * B
        %D = memref.alloc() : memref<64x64xi32>
        affine.for %i = 0 to 64 {
            affine.for %j = 0 to 64 {
                affine.for %k = 0 to 32 {
                    %a = affine.load %A[%i, %k] : memref<64x32xi32>
                    %b = affine.load %B[%k, %j] : memref<32x64xi32>
                    %c = affine.load %D[%i, %j] : memref<64x64xi32>
                    %prod = arith.muli %a, %b : i32
                    %sum = arith.addi %prod, %c: i32
                    affine.store %sum, %D[%i, %j] : memref<64x64xi32>
                } { loop_name = "k1" }
            // CHECK: pipeline_ii = 1 : i32
            } { loop_name = "j1" }
        } { loop_name = "i1", op_name = "s1" }
        %s2 = allo.create_op_handle "s2"
        %i2 = allo.create_loop_handle %s2, "i2"
        %j2 = allo.create_loop_handle %s2, "j2"
        %k2 = allo.create_loop_handle %s2, "k2"
        // E = C * D
        %E = memref.alloc() : memref<64x64xi32>
        affine.for %i = 0 to 64 {
            affine.for %j = 0 to 64 {
                affine.for %k = 0 to 64 {
                    %c = affine.load %C[%i, %k] : memref<64x64xi32>
                    %d = affine.load %D[%k, %j] : memref<64x64xi32>
                    %e = affine.load %E[%i, %j] : memref<64x64xi32>
                    %prod = arith.muli %c, %d : i32
                    %sum = arith.addi %prod, %e: i32
                    affine.store %sum, %E[%i, %j] : memref<64x64xi32>
                } { loop_name = "k2" }
                // CHECK: pipeline_ii = 1 : i32
            } { loop_name = "j2" }
        } { loop_name = "i2", op_name = "s2" }

        // apply the customization template
        allo.apply @gemm_opt(%A, %B, %D, %i1, %j1, %k1) : (memref<64x32xi32>, memref<32x64xi32>, memref<64x64xi32>, !allo.LoopHandle, !allo.LoopHandle, !allo.LoopHandle) -> ()
        allo.apply @gemm_opt(%C, %D, %E, %i2, %j2, %k2) : (memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32>, !allo.LoopHandle, !allo.LoopHandle, !allo.LoopHandle) -> ()
        return %E : memref<64x64xi32>
    }
}