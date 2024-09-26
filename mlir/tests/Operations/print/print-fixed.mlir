// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt %s --fixed-to-integer

module {
  func.func @top(%0: memref<4x4x!allo.Fixed<4, 2>>) -> () {
    allo.print(%0) : memref<4x4x!allo.Fixed<4, 2>>
    return
  }
}
