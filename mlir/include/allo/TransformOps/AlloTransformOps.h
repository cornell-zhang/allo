/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSFORMOPS_H
#define ALLO_TRANSFORMOPS_H

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "allo/TransformOps/AlloTransformOps.h.inc"

namespace mlir::allo {
void registerTransformDialectExtension(DialectRegistry &registry);

constexpr StringLiteral OpIdentifier = "sym_name";
}

#endif // ALLO_TRANSFORMOPS_H



