// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

  func.func @tri(%arg0: memref<5x5xf32>) -> memref<5x5xf32> attributes {itypes = "_", otypes = "_"} {
    %alloc = memref.alloc() {name = "outp"} : memref<5x5xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<5x5xf32>) outs(%alloc : memref<5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst = arith.constant 0.000000e+00 : f32
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = arith.cmpi ult, %0, %1 : index
      %3 = arith.select %2, %cst, %in : f32
      linalg.yield %3 : f32
    }
    return %alloc : memref<5x5xf32>
  }