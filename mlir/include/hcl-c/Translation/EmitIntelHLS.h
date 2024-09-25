/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_C_TRANSLATION_EMITINTELHLS_H
#define HCL_C_TRANSLATION_EMITINTELHLS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirLogicalResult mlirEmitIntelHls(
    MlirModule module, MlirStringCallback callback, void *userData);

#ifdef __cplusplus
}
#endif

#endif // HCL_C_TRANSLATION_EMITINTELHLS_H
