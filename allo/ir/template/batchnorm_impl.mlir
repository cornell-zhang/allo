// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// weights = scale / Sqrt(var+eps)
// bias = - mean * scale / Sqrt(var+eps) + beta
// Y = a * X + b

func.func @batchnorm(%input: memref<1x3x224x224xf32>, %weight: memref<3xf32>, %bias: memref<3xf32>) -> memref<1x3x224x224xf32> {
    %alloc = memref.alloc() {name="output"} : memref<1x3x224x224xf32>
    
    linalg.generic {indexing_maps = [affine_map<(i, j, k, l) -> (i, j, k, l)>, affine_map<(i, j, k, l) -> (j)>, affine_map<(i, j, k, l) -> (j)>, affine_map<(i, j, k, l) -> (i, j, k, l)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%input, %weight, %bias: memref<1x3x224x224xf32>, memref<3xf32>, memref<3xf32>) outs(%alloc: memref<1x3x224x224xf32>) {
        ^bb0(%in: f32, %w: f32, %b: f32, %a: f32):
            %0 = arith.mulf %in, %w : f32
            %1 = arith.addf %0, %b : f32
            linalg.yield %1 : f32
    }

    return %alloc : memref<1x3x224x224xf32>
}