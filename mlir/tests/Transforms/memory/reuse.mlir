// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt -opt %s | FileCheck %s

// CHECK: #set = affine_set<(d0) : (d0 - 2 >= 0)>
module {
    func.func @blur(%A: memref<10x10xf32>, %B: memref<10x8xf32>)
    {
        %s = hcl.create_op_handle "s"
        %li = hcl.create_loop_handle %s, "i"
        %lj = hcl.create_loop_handle %s, "j"
        affine.for %i = 0 to 10 {
            affine.for %j = 0 to 8 {
                // CHECK: %[[VAR1:.*]] = affine.load %[[VAR:.*]][1] : memref<3xf32>
                // CHECK: affine.store %[[VAR1]], %[[VAR]][0] : memref<3xf32>
                // CHECK: %[[VAR2:.*]] = affine.load %[[VAR]][2] : memref<3xf32>
                // CHECK: affine.store %[[VAR2]], %[[VAR]][1] : memref<3xf32>
                // CHECK: %[[VAR3:.*]] = affine.load {{.*}}[{{.*}}, {{.*}}] : memref<10x10xf32>
                // CHECK: affine.store %[[VAR3]], %[[VAR]][2] : memref<3xf32>
                // CHECK: affine.if #set(%arg3) {
                // CHECK:   {{.*}} = affine.load %[[VAR]][0] : memref<3xf32>
                // CHECK:   {{.*}} = affine.load %[[VAR]][1] : memref<3xf32>
                // CHECK:   {{.*}} = affine.load %[[VAR]][2] : memref<3xf32>
                // CHECK:   affine.store {{.*}}, {{.*}}[{{.*}}, {{.*}} - 2] : memref<10x8xf32>
                // CHECK: }
                %tmp = affine.load %A[%i, %j] : memref<10x10xf32>
                %tmp1 = affine.load %A[%i, %j+1] : memref<10x10xf32>
                %tmp2 = affine.load %A[%i, %j+2] : memref<10x10xf32>
                %sum = arith.addf %tmp, %tmp1: f32
                %sum1 = arith.addf %sum, %tmp2: f32
                affine.store %sum1, %B[%i, %j] : memref<10x8xf32>
            } { loop_name = "j" }
        } { loop_name = "i", op_name = "s" }
        %buf = hcl.reuse_at(%A: memref<10x10xf32>, %lj) -> memref<3xf32>
        return
    }
    func.func @blur5(%A: memref<10x10xf32>, %B: memref<10x5xf32>)
    {
        %s = hcl.create_op_handle "s"
        %li = hcl.create_loop_handle %s, "i"
        %lj = hcl.create_loop_handle %s, "j"
        affine.for %i = 0 to 10 {
            affine.for %j = 0 to 5 {
                %tmp = affine.load %A[%i, %j] : memref<10x10xf32>
                %tmp1 = affine.load %A[%i, %j+1] : memref<10x10xf32>
                %tmp2 = affine.load %A[%i, %j+2] : memref<10x10xf32>
                %tmp3 = affine.load %A[%i, %j+3] : memref<10x10xf32>
                %tmp4 = affine.load %A[%i, %j+4] : memref<10x10xf32>
                %sum = arith.addf %tmp, %tmp1: f32
                %sum1 = arith.addf %tmp2, %tmp3: f32
                %sum2 = arith.addf %sum1, %tmp4: f32
                %sum3 = arith.addf %sum, %sum2: f32
                affine.store %sum3, %B[%i, %j] : memref<10x5xf32>
            } { loop_name = "j" }
        } { loop_name = "i", op_name = "s" }
        %buf = hcl.reuse_at(%A: memref<10x10xf32>, %lj) -> memref<5xf32>
        return
    }
    func.func @blur_x(%A: memref<10x10xf32>, %B: memref<8x10xf32>)
    {
        %s = hcl.create_op_handle "s"
        %li = hcl.create_loop_handle %s, "i"
        %lj = hcl.create_loop_handle %s, "j"
        affine.for %i = 0 to 8 {
            affine.for %j = 0 to 10 {
                %tmp = affine.load %A[%i, %j] : memref<10x10xf32>
                %tmp1 = affine.load %A[%i+1, %j] : memref<10x10xf32>
                %tmp2 = affine.load %A[%i+2, %j] : memref<10x10xf32>
                %sum = arith.addf %tmp, %tmp1: f32
                %sum1 = arith.addf %sum, %tmp2: f32
                affine.store %sum1, %B[%i, %j] : memref<8x10xf32>
            } { loop_name = "j" }
        } { loop_name = "i", op_name = "s" }
        %buf = hcl.reuse_at(%A: memref<10x10xf32>, %li) -> memref<3x10xf32>
        return
    }
    // func @blur_reduction(%A: memref<10x10xf32>, %B: memref<10x8xf32>) -> memref<10x8xf32>
    // {
    //     %s = hcl.create_op_handle "s"
    //     %li = hcl.create_loop_handle %s, "i"
    //     %lj = hcl.create_loop_handle %s, "j"
    //     affine.for %i = 0 to 10 {
    //         affine.for %j = 0 to 8 {
    //             %zero = constant 0.0 : f32
    //             %sum = affine.for %r = 0 to 3 iter_args(%sum_iter = %zero) -> (f32) {
    //                 %tmp = affine.load %A[%i, %j+%r] : memref<10x10xf32>
    //                 %sum_next = arith.addf %sum_iter, %tmp: f32
    //                 affine.yield %sum_next : f32
    //             } { loop_name = "r", reduction = 1}
    //             affine.store %sum, %B[%i, %j] : memref<10x8xf32>
    //         } { loop_name = "j" }
    //     } { loop_name = "i", op_name = "s" }
    //     %buf = hcl.reuse_at(%A: memref<10x10xf32>, %lj) -> memref<3xf32>
    //     return %B : memref<10x8xf32>
    // }
    func.func @conv2d(%A: memref<10x10xf32>, %B: memref<8x8xf32>)
    {
        %s = hcl.create_op_handle "s"
        %li = hcl.create_loop_handle %s, "i"
        %lj = hcl.create_loop_handle %s, "j"
        affine.for %i = 0 to 8 {
            affine.for %j = 0 to 8 {
                %tmp = affine.load %A[%i, %j] : memref<10x10xf32>
                %tmp1 = affine.load %A[%i, %j+1] : memref<10x10xf32>
                %tmp2 = affine.load %A[%i, %j+2] : memref<10x10xf32>
                %tmp3 = affine.load %A[%i+1, %j] : memref<10x10xf32>
                %tmp4 = affine.load %A[%i+1, %j+1] : memref<10x10xf32>
                %tmp5 = affine.load %A[%i+1, %j+2] : memref<10x10xf32>
                %tmp6 = affine.load %A[%i+2, %j] : memref<10x10xf32>
                %tmp7 = affine.load %A[%i+2, %j+1] : memref<10x10xf32>
                %tmp8 = affine.load %A[%i+2, %j+2] : memref<10x10xf32>
                %sum = arith.addf %tmp, %tmp1: f32
                %sum1 = arith.addf %tmp2, %tmp3: f32
                %sum2 = arith.addf %sum1, %tmp4: f32
                %sum3 = arith.addf %sum, %sum2: f32
                affine.store %sum3, %B[%i, %j] : memref<8x8xf32>
            } { loop_name = "j" }
        } { loop_name = "i", op_name = "s" }
        %buf = hcl.reuse_at(%A: memref<10x10xf32>, %li) -> memref<3x10xf32>
        hcl.partition(%buf: memref<3x10xf32>, "CompletePartition", 1)
        return
    }
}