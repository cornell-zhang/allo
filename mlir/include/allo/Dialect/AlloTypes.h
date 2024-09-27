/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TYPES_H
#define ALLO_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "allo/Dialect/AlloTypes.h.inc"

#endif // ALLO_TYPES_H