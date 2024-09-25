/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MLIR_DIALECT_TRANSFORMOPS_HCLTRANSFORMOPS_H
#define MLIR_DIALECT_TRANSFORMOPS_HCLTRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
namespace affine {
class AffineForOp;
} // namespace affine
namespace hcl {
class ForOp;
} // namespace hcl
} // namespace mlir

namespace mlir {
namespace transform {

enum class FailurePropagationMode : uint32_t;
class FailurePropagationModeAttr;

/// A builder function that populates the body of a SequenceOp.
using SequenceBodyBuilderFn = ::llvm::function_ref<void(
    ::mlir::OpBuilder &, ::mlir::Location, ::mlir::BlockArgument)>;
using SequenceBodyBuilderArgsFn =
    ::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location,
                              ::mlir::BlockArgument, ::mlir::ValueRange)>;

} // namespace transform
} // namespace mlir

#define GET_OP_CLASSES
#include "hcl/Dialect/TransformOps/HCLTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace hcl {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace hcl
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORMOPS_HCLTRANSFORMOPS_H
