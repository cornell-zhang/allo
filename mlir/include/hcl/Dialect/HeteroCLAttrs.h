/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCLATTRS_H
#define HCLATTRS_H

#include "mlir/IR/BuiltinAttributes.h"

#include "hcl/Dialect/HeteroCLEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hcl/Dialect/HeteroCLAttrs.h.inc"

#endif // HCLATTRS_H