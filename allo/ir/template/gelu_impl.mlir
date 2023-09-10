// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

  func.func @Gelu_layer(%arg0: memref<16x72xf32>) -> memref<16x72xf32> attributes {itypes = "_", otypes = "_"} {
    %alloc = memref.alloc() {name = "outp"} : memref<16x72xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<16x72xf32>) outs(%alloc : memref<16x72xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst = arith.constant 5.000000e-01 : f32
      %cst_0 = arith.constant 3.000000e+00 : f32
      %cst_1 = arith.constant 4.471500e-02 : f32
      %cst_2 = arith.constant 7.978850e-01 : f32
      %cst_3 = arith.constant 1.000000e+00 : f32
      %0 = arith.mulf %cst, %in : f32
      %1 = math.powf %in, %cst_0 : f32
      %2 = arith.mulf %cst_1, %1 : f32
      %3 = arith.addf %in, %2 : f32
      %4 = arith.mulf %cst_2, %3 : f32
      %5 = math.tanh %4 : f32
      %6 = arith.addf %cst_3, %5 : f32
      %7 = arith.mulf %0, %6 : f32
      linalg.yield %7 : f32
    }
    return %alloc : memref<16x72xf32>
  }