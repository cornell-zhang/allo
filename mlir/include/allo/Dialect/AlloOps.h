/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HETEROCL_OPS_H
#define HETEROCL_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "allo/Dialect/AlloAttrs.h"
#include "allo/Dialect/AlloTypes.h"

namespace mlir {
namespace allo {
    void buildTerminatedBody(OpBuilder &builder, Location loc);
} // namespace allo
} // namespace mlir

#define GET_OP_CLASSES
#include "allo/Dialect/AlloOps.h.inc"

#endif // HETEROCL_OPS_H
