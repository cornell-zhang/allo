/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MLIR_PASSDETAIL_H
#define ALLO_MLIR_PASSDETAIL_H

#include "allo/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace allo {

#define GEN_PASS_DEF_LOOPTRANSFORMATION
#define GEN_PASS_DEF_DATAPLACEMENT
#define GEN_PASS_DEF_ANYWIDTHINTEGER
#define GEN_PASS_DEF_MOVERETURNTOINPUT
#define GEN_PASS_DEF_LEGALIZECAST
#define GEN_PASS_DEF_REMOVESTRIDEMAP
#define GEN_PASS_DEF_MEMREFDCE
#define GEN_PASS_DEF_COPYONWRITE
#include "allo/Transforms/Passes.h.inc"

} // namespace allo
} // end namespace mlir

#endif // ALLO_MLIR_PASSDETAIL_H
