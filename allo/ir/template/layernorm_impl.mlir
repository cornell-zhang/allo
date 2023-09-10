// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

  func.func @layernorm(%arg0: memref<32x48xf32>, %arg1: memref<48xf32>, %arg2: memref<48xf32>) -> memref<32x48xf32> {
    %alloc = memref.alloc() {name = "output"} : memref<32x48xf32>
    %alloc_0 = memref.alloc() {name = "mean"} : memref<32xf32>
    %cst = arith.constant {name="zero"} 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<32xf32>)
    %alloc_1 = memref.alloc() {name = "mean2"} : memref<32xf32>
    %cst_2 = arith.constant {name="zero"} 0.000000e+00 : f32
    linalg.fill ins(%cst_2 : f32) outs(%alloc_1 : memref<32xf32>)
    %alloc_3 = memref.alloc() {name = "var"} : memref<32xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], name = "mean"} ins(%arg0, %alloc_0, %alloc_1 : memref<32x48xf32>, memref<32xf32>, memref<32xf32>) outs(%alloc_0, %alloc_1 : memref<32xf32>, memref<32xf32>) {
    ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32, %out_6: f32):
      %0 = arith.addf %in, %in_4 : f32
      %1 = arith.mulf %in, %in : f32
      %2 = arith.addf %1, %in_5 : f32
      linalg.yield %0, %2 : f32, f32
    }
    %c_i32 = arith.constant {name="dimension"} 48 : i32
    %const = arith.sitofp %c_i32 : i32 to f32
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"], name = "var"} ins(%alloc_0, %alloc_1 : memref<32xf32>, memref<32xf32>) outs(%alloc_0, %alloc_1, %alloc_3 : memref<32xf32>, memref<32xf32>, memref<32xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32, %out_5: f32, %out_6: f32):
      %1 = arith.divf %in, %const : f32
      %2 = arith.divf %in_4, %const : f32
      %3 = arith.mulf %1, %1 : f32
      %4 = arith.subf %2, %3 : f32
      linalg.yield %1, %2, %4 : f32, f32, f32
    }
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], name = "output"} ins(%alloc_0, %alloc_3, %arg0, %arg1, %arg2 : memref<32xf32>, memref<32xf32>, memref<32x48xf32>, memref<48xf32>, memref<48xf32>) outs(%alloc : memref<32x48xf32>) {
    ^bb0(%in: f32, %in_4: f32, %in_5: f32, %in_6: f32, %in_7: f32, %out: f32):
      %0 = arith.subf %in_5, %in : f32
      %1 = arith.mulf %in_6, %0 : f32
      %cst_8 = arith.constant 9.99999974E-6 : f32
      %2 = arith.addf %in_4, %cst_8 : f32
      %3 = math.sqrt %2 : f32
      %4 = arith.divf %1, %3 : f32
      %5 = arith.addf %in_7, %4 : f32
      linalg.yield %5 : f32
    }
    return %alloc : memref<32x48xf32>
  }