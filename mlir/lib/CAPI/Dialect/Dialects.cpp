/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hcl-c/Dialect/Dialects.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HCL, hcl, mlir::hcl::HeteroCLDialect)