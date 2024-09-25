// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt -opt %s | FileCheck %s

module {
    // CHECK: func @test(%arg0: memref<1024x512x!hcl.Fixed<12, 6>>, %arg1: memref<512x1024x!hcl.UFixed<12, 2>>) {
    func.func @test(%A: memref<1024x512x!hcl.Fixed<12,6>>, %B: memref<512x1024x!hcl.UFixed<12,2>>)
    {
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                %a = affine.load %A[%i, %j] : memref<1024x512x!hcl.Fixed<12,6>>
                %b = affine.load %B[%i, %j] : memref<512x1024x!hcl.UFixed<12,2>>
            }
        }
        return
    }
}