/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===- HeteroCLDialect.cpp - hcl dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"

#include "llvm/ADT/TypeSwitch.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLTypes.h"

#include "hcl/Dialect/HeteroCLAttrs.h"
#include "hcl/Dialect/HeteroCLOps.h"

using namespace mlir;
using namespace mlir::hcl;

#include "hcl/Dialect/HeteroCLDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "hcl/Dialect/HeteroCLTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hcl/Dialect/HeteroCLAttrs.cpp.inc"

#include "hcl/Dialect/HeteroCLEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
void HeteroCLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hcl/Dialect/HeteroCLOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "hcl/Dialect/HeteroCLTypes.cpp.inc"
      >();
  addAttributes< // test/lib/Dialect/Test/TestAttributes.cpp
#define GET_ATTRDEF_LIST
#include "hcl/Dialect/HeteroCLAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Extra methods
//===----------------------------------------------------------------------===//

void StructType::print(mlir::AsmPrinter &p) const {
  p << "<";
  llvm::interleaveComma(getElementTypes(), p);
  p << '>';
}

Type StructType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    elementTypes.push_back(elementType);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater())
    return Type();
  return get(parser.getContext(), elementTypes);
}

LogicalResult ApplyOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  CustomizationOp fn =
      symbolTable.lookupNearestSymbolFrom<CustomizationOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid customization";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  // for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
  //   if (getOperand(i).getType() != fnType.getInput(i))
  //     return emitOpError("operand type mismatch: expected operand type ")
  //            << fnType.getInput(i) << ", but provided "
  //            << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  // for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
  //   if (getResult(i).getType() != fnType.getResult(i)) {
  //     auto diag = emitOpError("result type mismatch at index ") << i;
  //     diag.attachNote() << "      op result types: " << getResultTypes();
  //     diag.attachNote() << "function result types: " << fnType.getResults();
  //     return diag;
  //   }

  return success();
}