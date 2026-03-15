/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_INIT_ALL_PASSES_H
#define ALLO_INIT_ALL_PASSES_H

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "allo/Transforms/Passes.h"

namespace mlir::allo {
inline void registerAllPasses() {
  registerTransformsPasses();

  // Conversion passes
  registerConversionPasses();

  // Dialect passes
  affine::registerAffinePasses();
  arith::registerArithPasses();
  bufferization::registerBufferizationPasses();
  func::registerFuncPasses();
  registerLinalgPasses();
  LLVM::registerLLVMPasses();
  LLVM::registerTargetLLVMIRTransformsPasses();
  math::registerMathPasses();
  memref::registerMemRefPasses();
  omp::registerOpenMPPasses();
  registerSCFPasses();
  registerShapePasses();
  tensor::registerTensorPasses();
  tosa::registerTosaOptPasses();
  transform::registerTransformPasses();
  vector::registerVectorPasses();
  allo::registerAlloPasses();

  // Dialect pipelines
  bufferization::registerBufferizationPipelines();
  tosa::registerTosaToLinalgPipelines();
}
}



#endif // ALLO_INIT_ALL_PASSES_H