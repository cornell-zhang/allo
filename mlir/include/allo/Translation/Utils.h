/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_UTILS_H
#define ALLO_TRANSLATION_UTILS_H

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"

using namespace mlir;
using namespace allo;

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

/// This class maintains the mutable state that cross-cuts and is shared by the
/// various emitters.
class AlloEmitterState {
public:
  explicit AlloEmitterState(raw_ostream &os) : os(os) {}

  // The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIndent = 0;

  // This table contains all declared values.
  DenseMap<Value, SmallString<8>> nameTable;
  std::map<std::string, int> nameConflictCnt;

private:
  AlloEmitterState(const AlloEmitterState &) = delete;
  void operator=(const AlloEmitterState &) = delete;
};

/// This is the base class for all of the HLSCpp Emitter components.
class AlloEmitterBase {
public:
  explicit AlloEmitterBase(AlloEmitterState &state)
      : state(state), os(state.os) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  raw_ostream &indent() { return os.indent(state.currentIndent); }

  void addIndent() { state.currentIndent += 2; }
  void reduceIndent() { state.currentIndent -= 2; }

  // All of the mutable state we are maintaining.
  AlloEmitterState &state;

  // The stream to emit to.
  raw_ostream &os;

  /// Value name management methods.
  SmallString<8> addName(Value val, bool isPtr = false, std::string name = "");

  SmallString<8> getName(Value val);

  bool isDeclared(Value val) {
    if (getName(val).empty()) {
      return false;
    } else
      return true;
  }

private:
  AlloEmitterBase(const AlloEmitterBase &) = delete;
  void operator=(const AlloEmitterBase &) = delete;
};

void fixUnsignedType(Value &result, bool isUnsigned);
void fixUnsignedType(memref::GlobalOp &op, bool isUnsigned);

#endif // ALLO_TRANSLATION_UTILS_H