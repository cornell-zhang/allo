/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_TRANSLATION_EMITINTELHLS_H
#define HCL_TRANSLATION_EMITINTELHLS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace hcl {

LogicalResult emitIntelHLS(ModuleOp module, llvm::raw_ostream &os);
void registerEmitIntelHLSTranslation();

} // namespace hcl
} // namespace mlir

#endif // HCL_TRANSLATION_EMITINTELHLS_H