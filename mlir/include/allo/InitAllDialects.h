/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_INIT_ALL_DIALECTS_H
#define ALLO_INIT_ALL_DIALECTS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "allo/Dialect/AlloDialect.h"

namespace mlir::allo {
inline void registerAllDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    affine::AffineDialect,
    arith::ArithDialect,
    bufferization::BufferizationDialect,
    cf::ControlFlowDialect,
    func::FuncDialect,
    index::IndexDialect,
    linalg::LinalgDialect,
    LLVM::LLVMDialect,
    math::MathDialect,
    memref::MemRefDialect,
    omp::OpenMPDialect,
    scf::SCFDialect,
    shape::ShapeDialect,
    tensor::TensorDialect,
    tosa::TosaDialect,
    vector::VectorDialect,
    transform::TransformDialect,
    allo::AlloDialect
  >();
  // clang-format on
}
} // namespace mlir::allo

#endif // ALLO_INIT_ALL_DIALECTS_H
