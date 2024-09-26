// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt -opt %s | FileCheck %s

module {
    func.func @vector_add(%A: memref<256xf32>, %B: memref<256xf32>, %C: memref<256xf32>)
    {
        %s = allo.create_op_handle "s"
        %li = allo.create_loop_handle %s, "i"
        // CHECK: affine.for %arg3 = 0 to 4 {
        // CHECK:   affine.for %arg4 = 0 to 64 {
        affine.for %i = 0 to 256 {
            %a = affine.load %A[%i] : memref<256xf32>
            %b = affine.load %B[%i] : memref<256xf32>
            %sum = arith.addf %a, %b : f32
            affine.store %sum, %C[%i] : memref<256xf32>
            // CHECK:     } {loop_name = "i.inner", thread_axis = 3 : i32}
            // CHECK:   } {loop_name = "i.outer", op_name = "s", thread_axis = 0 : i32}
        } { loop_name = "i", op_name = "s" }

        %li_outer, %li_inner = allo.split (%li, 64)
        allo.bind (%li_outer, "BlockIdxX")
        allo.bind (%li_inner, "ThreadIdxX")
        return
    }
}
