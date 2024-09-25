// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt -opt %s | FileCheck %s

// CHECK: #map = affine_map<(d0, d1) -> (0, d1, d0, 0)>
module {
    hcl.customization @gemm_opt(
        %A: memref<?x?x!hcl.Type>,
        %B: memref<?x?x!hcl.Type>,
        %C: memref<?x?x!hcl.Type>,
        %s: !hcl.OpHandle,
        %i: !hcl.LoopHandle,
        %j: !hcl.LoopHandle,
        %k: !hcl.LoopHandle
    ) {
        hcl.pipeline(%j, 1)
        hcl.partition(%A: memref<?x?x!hcl.Type>, "CompletePartition", 2)
        hcl.partition(%B: memref<?x?x!hcl.Type>, "CompletePartition", 2)
        hcl.partition(%C: memref<?x?x!hcl.Type>, "CompletePartition", 2)
        hcl.end
    }
    func.func @top(%A: memref<64x32xi32>, %B: memref<32x64xi32>, %C: memref<64x64xi32>) -> memref<64x64xi32>
    {
        %s1 = hcl.create_op_handle "s1"
        %i1 = hcl.create_loop_handle %s1, "i1"
        %j1 = hcl.create_loop_handle %s1, "j1"
        %k1 = hcl.create_loop_handle %s1, "k1"
        // D = A * B
        %D = memref.alloc() : memref<64x64xi32>
        affine.for %i = 0 to 64 {
            affine.for %j = 0 to 64 {
                affine.for %k = 0 to 32 {
                    // CHECK: affine.load %[[ARG:.*]][%[[I:.*]], %[[K:.*]]] : memref<64x32xi32, #map>
                    %a = affine.load %A[%i, %k] : memref<64x32xi32>
                    %b = affine.load %B[%k, %j] : memref<32x64xi32>
                    %c = affine.load %D[%i, %j] : memref<64x64xi32>
                    %prod = arith.muli %a, %b : i32
                    %sum = arith.addi %prod, %c: i32
                    affine.store %sum, %D[%i, %j] : memref<64x64xi32>
                } { loop_name = "k1" }
            // CHECK:  } {loop_name = "j1", pipeline_ii = 1 : i32}
            } { loop_name = "j1" }
        } { loop_name = "i1", op_name = "s1" }
        %s2 = hcl.create_op_handle "s2"
        %i2 = hcl.create_loop_handle %s2, "i2"
        %j2 = hcl.create_loop_handle %s2, "j2"
        %k2 = hcl.create_loop_handle %s2, "k2"
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
            // CHECK:  } {loop_name = "j2", pipeline_ii = 1 : i32}
            } { loop_name = "j2" }
        } { loop_name = "i2", op_name = "s2" }
        hcl.apply @gemm_opt(%A, %B, %D, %s1, %i1, %j1, %k1) : (memref<64x32xi32>, memref<32x64xi32>, memref<64x64xi32>, !hcl.OpHandle, !hcl.LoopHandle, !hcl.LoopHandle, !hcl.LoopHandle) -> ()
        hcl.apply @gemm_opt(%C, %D, %E, %s2, %i2, %j2, %k2) : (memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32>, !hcl.OpHandle, !hcl.LoopHandle, !hcl.LoopHandle, !hcl.LoopHandle) -> ()
        return %E : memref<64x64xi32>
    }
}