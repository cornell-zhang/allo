/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_TRANSLATION_EMITVIVADOHLS_H
#define HCL_TRANSLATION_EMITVIVADOHLS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace hcl {

LogicalResult emitVivadoHLS(ModuleOp module, llvm::raw_ostream &os);
void registerEmitVivadoHLSTranslation();

} // namespace hcl
} // namespace mlir

#endif // HCL_TRANSLATION_EMITVIVADOHLS_H