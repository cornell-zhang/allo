// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s --fixed-to-integer | FileCheck %s

module {
  func.func private @conv1(memref<1x1x28x28xf32>, memref<6x1x5x5xf32>) -> memref<1x6x28x28xi8>
  func.func private @relu(memref<1x6x28x28xi8>) -> memref<1x6x28x28xi8>
  func.func private @pool(memref<1x6x28x28xi8>) -> memref<1x6x14x14xi8>
  func.func private @conv2(memref<1x6x14x14xi8>, memref<16x6x5x5xf32>) -> memref<1x16x10x10xi8>
  func.func private @relu_1(memref<1x16x10x10xi8>) -> memref<1x16x10x10xi8>
  func.func private @pool_1(memref<1x16x10x10xi8>) -> memref<1x16x5x5xi8>
  func.func private @flatten(memref<1x16x5x5xi8>) -> memref<1x400xi8>
  func.func private @fc1(memref<1x400xi8>, memref<120x400xf32>, memref<1x120xf32>) -> memref<1x120x!hcl.Fixed<8, 4>>
  func.func private @relu_2(memref<1x120x!hcl.Fixed<8, 4>>) -> memref<1x120x!hcl.Fixed<8, 4>>
  func.func private @fc2(memref<1x120x!hcl.Fixed<8, 4>>, memref<84x120xf32>, memref<1x84xf32>) -> memref<1x84x!hcl.Fixed<8, 4>>
  func.func private @relu_3(memref<1x84x!hcl.Fixed<8, 4>>) -> memref<1x84x!hcl.Fixed<8, 4>>
  func.func private @fc3(memref<1x84x!hcl.Fixed<8, 4>>, memref<10x84xf32>, memref<1x10xf32>) -> memref<1x10xf32>
  func.func @main() {
    %0 = memref.alloc() : memref<1x1x28x28xf32>
    %1 = memref.alloc() : memref<6x1x5x5xf32>
    %2 = memref.alloc() : memref<16x6x5x5xf32>
    %3 = memref.alloc() : memref<120x400xf32>
    %4 = memref.alloc() : memref<1x120xf32>
    %5 = memref.alloc() : memref<84x120xf32>
    %6 = memref.alloc() : memref<1x84xf32>
    %7 = memref.alloc() : memref<10x84xf32>
    %8 = memref.alloc() : memref<1x10xf32>
    %9 = call @conv1(%0, %1) : (memref<1x1x28x28xf32>, memref<6x1x5x5xf32>) -> memref<1x6x28x28xi8>
    %10 = call @relu(%9) : (memref<1x6x28x28xi8>) -> memref<1x6x28x28xi8>
    %11 = call @pool(%10) : (memref<1x6x28x28xi8>) -> memref<1x6x14x14xi8>
    %12 = call @conv2(%11, %2) : (memref<1x6x14x14xi8>, memref<16x6x5x5xf32>) -> memref<1x16x10x10xi8>
    %13 = call @relu_1(%12) : (memref<1x16x10x10xi8>) -> memref<1x16x10x10xi8>
    %14 = call @pool_1(%13) : (memref<1x16x10x10xi8>) -> memref<1x16x5x5xi8>
    %15 = call @flatten(%14) : (memref<1x16x5x5xi8>) -> memref<1x400xi8>
    %16 = call @fc1(%15, %3, %4) : (memref<1x400xi8>, memref<120x400xf32>, memref<1x120xf32>) -> memref<1x120x!hcl.Fixed<8, 4>>
    // CHECK: call @fc1(%6, %alloc_2, %alloc_3) : (memref<1x400xi8>, memref<120x400xf32>, memref<1x120xf32>) -> memref<1x120xi8>
    %17 = call @relu_2(%16) : (memref<1x120x!hcl.Fixed<8, 4>>) -> memref<1x120x!hcl.Fixed<8, 4>>
    // CHECK: call @relu_2(%7) : (memref<1x120xi8>) -> memref<1x120xi8>
    %18 = call @fc2(%17, %5, %6) : (memref<1x120x!hcl.Fixed<8, 4>>, memref<84x120xf32>, memref<1x84xf32>) -> memref<1x84x!hcl.Fixed<8, 4>>
    // CHECK: call @fc2(%8, %alloc_4, %alloc_5) : (memref<1x120xi8>, memref<84x120xf32>, memref<1x84xf32>) -> memref<1x84xi8>
    %19 = call @relu_3(%18) : (memref<1x84x!hcl.Fixed<8, 4>>) -> memref<1x84x!hcl.Fixed<8, 4>>
    // CHECK: call @relu_3(%9) : (memref<1x84xi8>) -> memref<1x84xi8>
    %20 = call @fc3(%19, %7, %8) : (memref<1x84x!hcl.Fixed<8, 4>>, memref<10x84xf32>, memref<1x10xf32>) -> memref<1x10xf32>
    // CHECK: call @fc3(%10, %alloc_6, %alloc_7) : (memref<1x84xi8>, memref<10x84xf32>, memref<1x10xf32>) -> memref<1x10xf32>
    return
  }
}