/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_CONVERSION_PASSDETAIL_H
#define ALLO_CONVERSION_PASSDETAIL_H

#include "allo/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace allo {

#define GEN_PASS_DECL
#include "allo/Conversion/Passes.h.inc"

} // namespace allo
} // end namespace mlir

#endif // ALLO_CONVERSION_PASSDETAIL_H
