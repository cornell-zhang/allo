/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_INIT_ALL_EXTENSIONS_H
#define ALLO_INIT_ALL_EXTENSIONS_H

#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtension.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"

#include "allo/TransformOps/AlloTransformOps.h"

namespace mlir::allo {
inline void registerAllExtensions(DialectRegistry &registry) {
  // Register all transform dialect extensions.
  affine::registerTransformDialectExtension(registry);
  bufferization::registerTransformDialectExtension(registry);
  func::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  tensor::registerTransformDialectExtension(registry);
  transform::registerLoopExtension(registry);
  vector::registerTransformDialectExtension(registry);

  allo::registerTransformDialectExtension(registry);
}
}
#endif // ALLO_INIT_ALL_EXTENSIONS_H
