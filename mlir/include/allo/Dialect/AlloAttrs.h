/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_ATTRS_H
#define ALLO_ATTRS_H

#include "mlir/IR/BuiltinAttributes.h"

#include "allo/Dialect/AlloEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "allo/Dialect/AlloAttrs.h.inc"

#endif // ALLO_ATTRS_H