/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hcl-c/Dialect/HCLAttributes.h"
#include "hcl/Dialect/HeteroCLAttrs.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace hcl;

// bool mlirAttributeIsAIntegerSet(MlirAttribute attr) {
//   return unwrap(attr).isa<IntegerSetAttr>();
// }

MlirAttribute mlirIntegerSetAttrGet(MlirIntegerSet set) {
  return wrap(IntegerSetAttr::get(unwrap(set)));
}

bool mlirAttributeIsAPartitionKind(MlirAttribute attr) {
  return unwrap(attr).isa<PartitionKindEnumAttr>();
}

MlirAttribute mlirPartitionKindGet(MlirContext ctx, MlirAttribute kind) {
  IntegerAttr attr = unwrap(kind).cast<IntegerAttr>();
  PartitionKindEnum kindEnum = static_cast<PartitionKindEnum>(attr.getInt());
  return wrap(PartitionKindEnumAttr::get(unwrap(ctx), kindEnum));
}

bool mlirAttributeIsANDRangeDimKind(MlirAttribute attr) {
  return unwrap(attr).isa<NDRangeDimKindEnumAttr>();
}

MlirAttribute mlirNDRangeDimKindGet(MlirContext ctx, MlirAttribute kind) {
  IntegerAttr attr = unwrap(kind).cast<IntegerAttr>();
  NDRangeDimKindEnum kindEnum = static_cast<NDRangeDimKindEnum>(attr.getInt());
  return wrap(NDRangeDimKindEnumAttr::get(unwrap(ctx), kindEnum));
}