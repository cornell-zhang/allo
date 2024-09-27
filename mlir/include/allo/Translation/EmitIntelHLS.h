/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITINTELHLS_H
#define ALLO_TRANSLATION_EMITINTELHLS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace allo {

LogicalResult emitIntelHLS(ModuleOp module, llvm::raw_ostream &os);
void registerEmitIntelHLSTranslation();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITINTELHLS_H