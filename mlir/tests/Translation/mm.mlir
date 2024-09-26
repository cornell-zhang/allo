// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt --opt -jit %s | FileCheck %s

module {

  memref.global "private" @gv0 : memref<4x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]>
  memref.global "private" @gv1 : memref<4x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]>
  memref.global "private" @gv2 : memref<4x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]>

  func.func @matrix_multiply(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>)
  { 
    %s = allo.create_op_handle "s"
    %li = allo.create_loop_handle %s, "i"
    %lj = allo.create_loop_handle %s, "j"
    %lk = allo.create_loop_handle %s, "k"
    // CHECK: llvm.func @matrix_multiply(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) {
    affine.for %i = 0 to 4 {          
      affine.for %j = 0 to 4 {      
        affine.for %k = 0 to 4 {
          %a = affine.load %A[%i, %k] : memref<4x4xf32> 
          %b = affine.load %B[%k, %j] : memref<4x4xf32> 
          %c = affine.load %C[%i, %j] : memref<4x4xf32> 
          %prod = arith.mulf %a, %b : f32
          %sum  = arith.addf %prod, %c: f32
          affine.store %sum, %C[%i, %j] : memref<4x4xf32> 
        } {loop_name = "k"}
      } {loop_name = "j"}
    } {loop_name = "i", op_name="s"}
      
    %li0, %li1 = allo.split (%li, 2)
    %lj0, %lj1 = allo.split (%lj, 2)
    allo.reorder(%li0, %lj0, %li1,%lj1)
    allo.unroll(%lj1)
    allo.pipeline(%lj1, 1)
    return
  }

  func.func @top() -> () {
    %0 = memref.get_global @gv0 : memref<4x4xf32>
    %1 = memref.get_global @gv0 : memref<4x4xf32>
    %2 = memref.get_global @gv0 : memref<4x4xf32>

    call @matrix_multiply(%0, %1, %2) : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> () 
    return
  }
}
