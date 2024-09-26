// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt  %s 

// check https://github.com/zzzDavid/allo-debug/tree/main/read_write_file for full runnable examples.
// this test is only to make sure the generated IR is correct.
// the IR here is not runnable, because read/write requires absolute path.

module {
    llvm.mlir.global internal constant @str_global("input.txt\00")
    llvm.mlir.global internal constant @str_global2("output.txt\00")
    func.func private @readMemrefI64(memref<*xi64>, !llvm.ptr<i8>)
    func.func private @writeMemrefI64(memref<*xi64>, !llvm.ptr<i8>)
    func.func private @printMemrefI64(%ptr : memref<*xi64>)
    func.func @top () -> () {
        %0 = memref.alloc() : memref<2x2xi64>
        %1 = memref.cast %0 : memref<2x2xi64> to memref<*xi64>
        %2 = llvm.mlir.addressof @str_global : !llvm.ptr<array<10 x i8>>
        %3 = llvm.mlir.constant(0 : index) : i64
        %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<10 x i8>>, i64, i64) -> !llvm.ptr<i8>
        call @readMemrefI64(%1, %4) : (memref<*xi64>, !llvm.ptr<i8>) -> ()
        call @printMemrefI64(%1) : (memref<*xi64>) -> ()

        %5 = llvm.mlir.addressof @str_global2 : !llvm.ptr<array<11 x i8>>
        %6 = llvm.mlir.constant(0 : index) : i64
        %7 = llvm.getelementptr %5[%6, %6] : (!llvm.ptr<array<11 x i8>>, i64, i64) -> !llvm.ptr<i8>
        call @writeMemrefI64(%1, %7) : (memref<*xi64>, !llvm.ptr<i8>) -> ()
        return
    }
    func.func @main() -> () {
        call @top() : () -> ()
        return
    }
}