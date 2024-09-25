// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --lower-anywidth-integer
module {
  func.func @kernel(%arg0: memref<4x4xf32>, %arg1: f32, %arg2: f32) -> memref<4x4xf32> attributes {"top"} {
    return %arg0 : memref<4x4xf32>
  }
}