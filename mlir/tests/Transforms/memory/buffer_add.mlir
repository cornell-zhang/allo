// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func.func @add_buffer_at_axis_0(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>)
    {
        %s = hcl.create_op_handle "s"
        %l1 = hcl.create_loop_handle %s, "i"
        %l2 = hcl.create_loop_handle %s, "j"
        affine.for %i = 0 to 1024 {
            // CHECK: %[[MEM:.*]] = memref.alloc() : memref<1024xf32>
            // CHECK: %cst = arith.constant 0.000000e+00 : f32
            // CHECK: affine.for %[[VAR:.*]] = 0 to 1024 {
            // CHECK:     affine.store %cst, %[[MEM]][%[[VAR]]] : memref<1024xf32>
            // CHECK: } {buffer, loop_name = "j_init", pipeline_ii = 1 : i32}
            // CHECK: affine.for %[[VAR]] = 0 to 1024
            affine.for %j = 0 to 1024 {
                // B[i, j] = A[i, j] + 1
                %a = affine.load %A[%i, %j] : memref<1024x1024xf32>
                %cst = arith.constant 1.0 : f32
                %sum = arith.addf %a, %cst: f32 //register
                // CHECK: affine.store {{.*}}, %[[MEM]][%[[VAR]]] : memref<1024xf32>
                affine.store %sum, %B[%i, %j] : memref<1024x1024xf32>
            } { loop_name = "j" }
            // CHECK: affine.for %[[VAR]] = 0 to 1024 {
            // CHECK:     %[[RES:.*]] = affine.load %[[MEM]][%[[VAR]]] : memref<1024xf32>
            // CHECK:     affine.store %[[RES]], {{.*}}[{{.*}}, %[[VAR]]] : memref<1024x1024xf32>
            // CHECK: } {buffer, loop_name = "j_back", pipeline_ii = 1 : i32}
        } { loop_name = "i", op_name = "s" }
        %buf = hcl.buffer_at(%B: memref<1024x1024xf32>, %l1) -> memref<1024xf32>
        return
    }
    // Notice: buffer_at cannot apply to the inner-most non-reduction loop
    // func @add_buffer_at_axis_1(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>)
    // {
    //     %s = hcl.create_op_handle "s"
    //     %l1 = hcl.create_loop_handle %s, "i"
    //     %l2 = hcl.create_loop_handle %s, "j"
    //     affine.for %i = 0 to 1024 {
    //         affine.for %j = 0 to 1024 {
    //             // B[i, j] = A[i, j] + 1
    //             %a = affine.load %A[%i, %j] : memref<1024x1024xf32>
    //             %cst = arith.constant 1.0 : f32
    //             %sum = arith.addf %a, %cst: f32 //register
    //             affine.store %sum, %B[%i, %j] : memref<1024x1024xf32>
    //         } { loop_name = "j" }
    //     } { loop_name = "i", op_name = "s" }
    //     // expected-error@+1 {{Cannot buffer at the inner-most loop: axis=1 inner-most axis=1}}
    //     %buf = hcl.buffer_at(%B: memref<1024x1024xf32>, %l2) -> memref<1xf32>
    //     return
    // }
}