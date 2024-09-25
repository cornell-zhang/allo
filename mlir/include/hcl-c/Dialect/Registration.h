/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_MLIR_C_REGISTRATION_H
#define HCL_MLIR_C_REGISTRATION_H

#include "mlir/CAPI/IR.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Registers all dialects with a context.
 * This is needed before creating IR for these Dialects.
 */
MLIR_CAPI_EXPORTED void hclMlirRegisterAllDialects(MlirContext context);

/** Registers all passes for symbolic access with the global registry. */
MLIR_CAPI_EXPORTED void hclMlirRegisterAllPasses();

#ifdef __cplusplus
}
#endif

#endif // HCL_MLIR_C_REGISTRATION_H