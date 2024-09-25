// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s
module {
    func.func @top(%arg0 : memref<1xi1>) {
        %c0 = arith.constant 0 : index
        %2 = hcl.or {
            %true = arith.constant 1 : i1
            hcl.yield %true : i1
        }, {
            %0 = memref.load %arg0[%c0] : memref<1xi1>
            hcl.yield %0 : i1
        } : i1
        func.return
    }
}