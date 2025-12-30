/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Translation/EmitBaseHLS.h"

using namespace mlir;
using namespace allo;
using namespace hls;

void ModuleEmitterBase::emitBinary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  Value result = op->getResult(0);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, rank);
  os << " = ";
  emitValue(op->getOperand(0), rank);
  os << " " << syntax << " ";
  emitValue(op->getOperand(1), rank);
  os << ";";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void ModuleEmitterBase::emitUnary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  Value result = op->getResult(0);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, rank);
  os << " = " << syntax << "(";
  emitValue(op->getOperand(0), rank);
  os << ");";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void ModuleEmitterBase::emitPower(Operation *op) {
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  emitValue(op->getResult(0), rank);
  os << " = pow(";
  emitValue(op->getOperand(0), rank);
  os << ", ";
  emitValue(op->getOperand(1), rank);
  os << ");";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void ModuleEmitterBase::emitMaxMin(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  Value result = op->getResult(0);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, rank);
  os << " = " << syntax << "(";
  emitValue(op->getOperand(0), rank);
  os << ", ";
  emitValue(op->getOperand(1), rank);
  os << ");";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void ModuleEmitterBase::emitNestedLoopTail(unsigned rank) {
  for (unsigned i = 0; i < rank; ++i) {
    reduceIndent();
    indent();
    os << "}\n";
  }
}

void ModuleEmitterBase::emitInfoAndNewLine(Operation *op) {
  os << "\t//";
  // Print line number.
  if (auto loc = llvm::dyn_cast<FileLineColLoc>(op->getLoc()))
    os << " L" << loc.getLine();
  os << "\n";
}
