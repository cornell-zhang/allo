// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func.func @gemm_fuse_two(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %s = hcl.create_op_handle "s"
        %li = hcl.create_loop_handle %s, "i"
        %lj = hcl.create_loop_handle %s, "j"
        %lk = hcl.create_loop_handle %s, "k"
        // CHECK: affine.for %[[ARG:.*]] = 0 to 1048576 {
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
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
        %l_fused = hcl.fuse (%li, %lj)
        // (i,j)->(ij/1024,ij%1024)
        return
    }
    func.func @gemm_fuse_three(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %s = hcl.create_op_handle "s"
        %li = hcl.create_loop_handle %s, "i"
        %lj = hcl.create_loop_handle %s, "j"
        %lk = hcl.create_loop_handle %s, "k"
        // CHECK: affine.for %[[ARG:.*]] = 0 to 536870912 {
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
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
        %l_fused = hcl.fuse (%li, %lj, %lk)
        // (i,j,k)->(ijk/(1024*1024),ijk/1024%1024,ijk%1024)
        return
    }
    func.func @gemm_fuse_two_among_four(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %s = hcl.create_op_handle "s"
        %li = hcl.create_loop_handle %s, "i"
        %lj = hcl.create_loop_handle %s, "j"
        %lk = hcl.create_loop_handle %s, "k"
        %ll = hcl.create_loop_handle %s, "l"
        // CHECK: affine.for %[[ARG:.*]] = 0 to 1024 {
        affine.for %i = 0 to 1024 {
            // CHECK: affine.for %[[ARG1:.*]] = 0 to 1048576 {
            affine.for %j = 0 to 1024 {
                affine.for %l = 0 to 1024 {
                    affine.for %k = 0 to 512 {
                        %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                        %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                        %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                        %prod = arith.mulf %a, %b : f32
                        %sum = arith.addf %prod, %c: f32
                        affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                    } { loop_name = "k" }
                } { loop_name = "l" }
            } { loop_name = "j" }
        } { loop_name = "i", op_name = "s" }
        %l_fused = hcl.fuse (%lj, %ll)
        // (i,j,k)->(ijk/(1024*1024),ijk/1024%1024,ijk%1024)
        return
    }
}