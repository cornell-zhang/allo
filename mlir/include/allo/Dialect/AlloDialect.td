/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HETEROCL_DIALECT
#define HETEROCL_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// allo dialect definition.
//===----------------------------------------------------------------------===//

def Allo_Dialect : Dialect {
    let name = "allo";
    let summary = "A allo out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let useDefaultTypePrinterParser = 1;
    let cppNamespace = "::mlir::allo";
}

//===----------------------------------------------------------------------===//
// Base allo operation definition.
//===----------------------------------------------------------------------===//

class Allo_Op<string mnemonic, list<Trait> traits = []> :
        Op<Allo_Dialect, mnemonic, traits>;

class Allo_Type<string mnemonic, list<Trait> traits = []> : 
        TypeDef<Allo_Dialect, mnemonic, traits>;

class Allo_Attr<string mnemonic, list<Trait> traits = []> : 
        AttrDef<Allo_Dialect, mnemonic, traits>;

#endif // HETEROCL_DIALECT
