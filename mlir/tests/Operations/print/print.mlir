// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt --fixed-to-integer --lower-print-ops --jit %s

// This file tests hcl.print operations.
// - Different types: int, float, fixed
// - Different print formats
// - Different number of values to print

module {
    memref.global "private" @gv_cst : memref<1xi64> = dense<[8]>
    memref.global "private" @gv_f64 : memref<1xf64> = dense<[8.0]>
    func.func @top () -> () {
        %c1 = arith.constant 0 : index
        %fixed_memref = hcl.get_global_fixed @gv_cst : memref<1x!hcl.Fixed<32,2>>
        %loaded_fixed = affine.load %fixed_memref[%c1] : memref<1x!hcl.Fixed<32,2>>
        hcl.print(%loaded_fixed) {format="test fixed point print: %.2f \n"} : !hcl.Fixed<32,2>

        %c1_i32 = arith.constant 144 : i32
        hcl.print(%c1_i32) {format="test integer print: %d \n"} : i32

        %c2_i32 = arith.constant -128 : i22
        hcl.print(%c1_i32, %c2_i32) {format="test two integers print: %d %d \n"} : i32, i22

        %0 = memref.get_global @gv_f64 : memref<1xf64>
        %1 = memref.load %0[%c1] : memref<1xf64>
        %c1_f64 = arith.constant 1351.5 : f64
        hcl.print(%1)  {format="loaded from memref: %.2f \n"} : f64
        hcl.print(%c1_f64) {format="constant result: %.2f \n"} : f64
        return
    }
}