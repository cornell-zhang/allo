// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt -opt %s | FileCheck %s

module {
    func.func private @S0(%arg0: index, %arg1: index, %arg2: memref<61xf32>, %arg3: index, %arg4: memref<64xf32>, %arg5: memref<3xf32>) attributes {scop.stmt} {
        %0 = affine.load %arg5[symbol(%arg0)+symbol(%arg1)] : memref<3xf32>
        %1 = affine.load %arg4[symbol(%arg1)] : memref<64xf32>
        %2 = arith.mulf %0, %1 : f32
        %3 = affine.load %arg2[symbol(%arg3)] : memref<61xf32>
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %arg2[symbol(%arg3)] : memref<61xf32>
        return
    }

    func.func @conv1d(%A: memref<64xf32>, %W: memref<3xf32>, %C: memref<61xf32>)
    {
        %s = hcl.create_op_handle "s"
        %li = hcl.create_loop_handle %s, "i"
        %lj = hcl.create_loop_handle %s, "j"

        // Polymer (PoCC) post-procssed loop nest
        affine.for %i = 0 to 61 {
            affine.for %j = 0 to 3 {
                func.call @S0(%i, %j, %C, %j, %A, %W) : (index, index, memref<61xf32>, index, memref<64xf32>, memref<3xf32>) -> ()
            // CHECK:  } {dep_distance = 1 : i64, loop_name = "j", unroll = 3 : i32}
            } { loop_name = "j", dep_distance = 1 }
        } { loop_name = "i", op_name = "s", dep_distance = 0 }

        %pe_array = hcl.unfold( %lj, 3 ) 
        hcl.to(%W : memref<3xf32>, %pe_array) { pe_index = [0,1,2] } -> memref<1xf32>
        %pe0_w = hcl.to(%W: memref<3xf32>, %pe_array) { pe_index = [0] } -> memref<1xf32>
        %pe1_w = hcl.to(%pe0_w: memref<1xf32>, %pe_array) { pe_index = [1] } -> memref<1xf32>
        return
    }
}