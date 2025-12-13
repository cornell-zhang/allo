/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Dialect/Registration.h"
#include "allo/Conversion/Passes.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "allo/Dialect/AlloDialect.h"
#include "mlir/InitAllDialects.h"

void alloMlirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::allo::AlloDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::pdl::PDLDialect,
                  mlir::transform::TransformDialect>();
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void alloMlirRegisterAllPasses() {
  // General passes
  mlir::registerTransformsPasses();

  // Conversion passes
  mlir::registerConversionPasses();

  // Dialect passes
  mlir::affine::registerAffinePasses();
  mlir::arith::registerArithPasses();
  mlir::memref::registerMemRefPasses();
  mlir::registerLinalgPasses();

  mlir::allo::registerAlloPasses();
  mlir::allo::registerAlloConversionPasses();
}