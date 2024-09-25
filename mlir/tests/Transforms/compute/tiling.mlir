// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func.func @matrix_multiply(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %s = hcl.create_op_handle "s"
        %li = hcl.create_loop_handle %s, "i"
        %lj = hcl.create_loop_handle %s, "j"
        %lk = hcl.create_loop_handle %s, "k"
        // CHECK: affine.for %arg3 = 0 to 8 {
        // CHECK:   affine.for %arg4 = 0 to 8 {
        // CHECK:     affine.for %arg5 = 0 to 2 {
        // CHECK:       affine.for %arg6 = 0 to 2 {
        // CHECK:         affine.for %arg7 = 0 to 2 {
        // CHECK:           affine.for %arg8 = 0 to 2 {
        // CHECK:             affine.for %arg9 = 0 to 64 {
        // CHECK:               affine.for %arg10 = 0 to 64 {
        // CHECK:                 affine.for %arg11 = 0 to 16 {
        // CHECK:                   affine.for %arg12 = 0 to 8 {
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = arith.mulf %a, %b : f32
                    %sum = arith.addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                // CHECK:     } {loop_name = "k.inner", unroll = 16 : i32}
                // CHECK:   } {loop_name = "k.outer", pipeline_ii = 1 : i32}
                // CHECK: } {loop_name = "j.inner", parallel = 1 : i32}
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", op_name = "s" }
        %li_outer, %li_inner = hcl.split (%li, 16)
        %li_in_out, %li_in_in = hcl.split (%li_inner, 4) // nest with split
        %li_out_out, %li_out_in = hcl.split (%li_outer, 8) // multiple split
        %lj_out, %lj_in, %lk_out, %lk_in = hcl.tile (%lj, %lk, 16, 8) // split & tile
        %l14, %l15, %l16, %l17 = hcl.tile (%li_in_out, %li_in_in, 2, 2) // nest with split (failed)
        hcl.unroll (%lk_in, 16) // unroll
        hcl.pipeline (%lk_out, 1) // pipeline
        hcl.parallel (%lj_in) // parallel
        return
    }
}