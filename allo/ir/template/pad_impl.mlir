// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// https://mlir.llvm.org/docs/Dialects/Affine/#affineif-affineaffineifop

#interior = affine_set<(o, c, i, j) : 
    (i - {PADDING} >= 0, j - {PADDING} >= 0,  {ORI_WIDTH_PAD} - i >= 0, {ORI_HEIGHT_PAD} - j >= 0)>

func.func @pad_tensor(%I : memref<{BATCH}x{CHANNEL}x{ORI_WIDTH}x{ORI_HEIGHT}xf32>) -> memref<{BATCH}x{CHANNEL}x{NEW_WIDTH}x{NEW_HEIGHT}xf32> {
  %O = memref.alloc() {name = "outp"} : memref<{BATCH}x{CHANNEL}x{NEW_WIDTH}x{NEW_HEIGHT}xf32>
  affine.parallel (%b, %c, %i, %j) = (0, 0, 0, 0) to ({BATCH}, {CHANNEL}, {NEW_WIDTH}, {NEW_HEIGHT}) {
    affine.if #interior (%b, %c, %i, %j) {
      %2 = affine.load %I[%b, %c, %i - {PADDING}, %j - {PADDING}] : memref<{BATCH}x{CHANNEL}x{ORI_WIDTH}x{ORI_HEIGHT}xf32>
      affine.store %2, %O[%b, %c, %i, %j] : memref<{BATCH}x{CHANNEL}x{NEW_WIDTH}x{NEW_HEIGHT}xf32>
    } else {
      %2 = arith.constant 0.0 : f32
      affine.store %2, %O[%b, %c, %i, %j] : memref<{BATCH}x{CHANNEL}x{NEW_WIDTH}x{NEW_HEIGHT}xf32>
    }
  }
  return %O: memref<{BATCH}x{CHANNEL}x{NEW_WIDTH}x{NEW_HEIGHT}xf32>
}






