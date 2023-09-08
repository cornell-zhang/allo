// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// https://github.com/llvm/llvm-project/blob/main/mlir/test/Dialect/Linalg/transform-op-decompose.mlir

func.func @softmax(%A: memref<2x16x32xf32>, %B: memref<2x16x32xf32>) -> memref<2x16x32xf32> {
        %0 = memref.alloc() : memref<2x16xf32>
        %C0_f32 = arith.constant 0xFF800000 : f32
        linalg.fill ins(%C0_f32 : f32) outs(%0 : memref<2x16xf32>)
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel",
      "parallel", "reduction"]} ins(%A : memref<2x16x32xf32>) outs(%0 : memref<2x16xf32>) {
          ^bb0(%IN: f32, %OUT: f32):
            %7 = arith.maxf %IN, %OUT : f32
            linalg.yield %7 : f32
          }
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types =
      ["parallel", "parallel", "parallel"]} ins(%A, %0 : memref<2x16x32xf32>, memref<2x16xf32>)
      outs(%B : memref<2x16x32xf32>) {
          ^bb0(%IN1: f32, %IN2: f32, %OUT: f32):
            %7 = arith.subf %IN1, %IN2 : f32
            %8 = math.exp %7 : f32
            linalg.yield %8 : f32
          }
          %1 = memref.alloc() : memref<2x16xf32>
          %C1 = arith.constant 0.000000e+00 : f32
          linalg.fill ins(%C1 : f32) outs(%1 : memref<2x16xf32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel",
      "parallel", "reduction"]} ins(%B : memref<2x16x32xf32>) outs(%1 : memref<2x16xf32>) {
          ^bb0(%IN: f32, %OUT: f32):
            %7 = arith.addf %IN, %OUT : f32
            linalg.yield %7 : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types =
      ["parallel", "parallel", "parallel"]} ins(%B, %1 : memref<2x16x32xf32>, memref<2x16xf32>)
      outs(%B : memref<2x16x32xf32>) {
          ^bb0(%IN1: f32, %IN2: f32, %OUT: f32):
            %7 = arith.divf %IN1, %IN2 : f32
            linalg.yield %7 : f32
          }
          return %B : memref<2x16x32xf32>
}