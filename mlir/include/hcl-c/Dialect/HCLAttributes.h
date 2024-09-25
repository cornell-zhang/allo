/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_MLIR_C_HCL_ATTRIBUTES__H
#define HCL_MLIR_C_HCL_ATTRIBUTES__H

#include "mlir-c/IR.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IntegerSet.h"

#ifdef __cplusplus
extern "C" {
#endif

// MLIR_CAPI_EXPORTED bool mlirAttributeIsAIntegerSet(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirIntegerSetAttrGet(MlirIntegerSet set);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAPartitionKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPartitionKindGet(MlirContext ctx,
                                                      MlirAttribute kind);

MLIR_CAPI_EXPORTED bool mlirAttributeIsANDRangeDimKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirNDRangeDimKindGet(MlirContext ctx,
                                                       MlirAttribute kind);

#ifdef __cplusplus
}
#endif

#endif // HCL_MLIR_C_HCL_ATTRIBUTES__H
