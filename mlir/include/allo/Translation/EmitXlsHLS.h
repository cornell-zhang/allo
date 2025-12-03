/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITXLSHLS_H
#define ALLO_TRANSLATION_EMITXLSHLS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace allo {

/// Emit XLS HLS code from MLIR module.
/// @param module The MLIR module to emit
/// @param os Output stream to write to
/// @param useMemory If true, emit arrays as __xls_memory<T, size> (for SRAM/BRAM).
///                  If false (default), emit as plain C arrays (for registers).
LogicalResult emitXlsHLS(ModuleOp module, llvm::raw_ostream &os, bool useMemory = false);
void registerEmitXlsHLSTranslation();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITXLSHLS_H 