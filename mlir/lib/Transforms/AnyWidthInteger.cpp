/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// AnyWidthInteger Pass
// This pass is to support any-width integer input from numpy.
// The input program has any-width integer input/output arguments
// The output program has 64-bit integer input/output and casts
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Support/Utils.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

void updateTopFunctionSignature(func::FuncOp &funcOp) {
  FunctionType functionType = funcOp.getFunctionType();
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 8> arg_types;
  for (const auto &argEn : llvm::enumerate(funcOp.getArguments()))
    arg_types.push_back(argEn.value().getType());

  SmallVector<Type, 4> new_result_types;
  SmallVector<Type, 8> new_arg_types;

  for (Type t : result_types) {
    if (MemRefType memrefType = llvm::dyn_cast<MemRefType>(t)) {
      Type et = memrefType.getElementType();
      // If result memref element type is integer
      // change it to i64 to be compatible with numpy
      if (llvm::isa<IntegerType>(et)) {
        size_t width = 64;
        Type newElementType = IntegerType::get(funcOp.getContext(), width);
        new_result_types.push_back(memrefType.clone(newElementType));
      } else {
        new_result_types.push_back(memrefType);
      }
    } else {
      new_result_types.push_back(t);
    }
  }

  for (Type t : arg_types) {
    if (MemRefType memrefType = llvm::dyn_cast<MemRefType>(t)) {
      Type et = memrefType.getElementType();
      // If argument memref element type is integer
      // change it to i64 to be compatible with numpy
      if (llvm::isa<IntegerType>(et)) {
        size_t width = 64;
        Type newElementType = IntegerType::get(funcOp.getContext(), width);
        new_arg_types.push_back(memrefType.clone(newElementType));
      } else {
        new_arg_types.push_back(memrefType);
      }
    } else {
      new_arg_types.push_back(t);
    }
  }

  // Get signedness hint information
  std::string itypes = "";
  if (funcOp->hasAttr("itypes")) {
    itypes =
        llvm::dyn_cast<StringAttr>(funcOp->getAttr("itypes")).getValue().str();
  }
  std::string otypes = "";
  if (funcOp->hasAttr("otypes")) {
    otypes =
        llvm::dyn_cast<StringAttr>(funcOp->getAttr("otypes")).getValue().str();
  }

  // Update func::FuncOp's block argument types
  // Also build loop nest to cast the input args
  SmallVector<Value, 4> newMemRefs;
  SmallVector<Value, 4> blockArgs;
  OpBuilder builder(funcOp->getRegion(0));
  for (Block &block : funcOp.getBlocks()) {
    for (unsigned i = 0; i < block.getNumArguments(); i++) {
      for (unsigned i = 0; i < block.getNumArguments(); ++i) {
        MemRefType memrefType =
            llvm::dyn_cast<MemRefType>(block.getArgument(i).getType());
        if (!memrefType) {
          continue;
        }

        Type et = memrefType.getElementType();
        if (!llvm::isa<IntegerType>(et)) {
          continue;
        }

        size_t width = 64;
        Type newType = IntegerType::get(funcOp.getContext(), width);
        Type newMemRefType = memrefType.clone(newType);
        block.getArgument(i).setType(newMemRefType);

        bool is_unsigned = false;
        if (i < itypes.length()) {
          is_unsigned = itypes[i] == 'u';
        }

        auto blockArgI = block.getArgument(i);
        Value newMemRef = castIntMemRef(
            builder, funcOp->getLoc(), blockArgI,
            llvm::dyn_cast<IntegerType>(et).getWidth(), is_unsigned);
        newMemRefs.push_back(newMemRef);
        blockArgs.push_back(block.getArgument(i));
      }
    }
  }

  // Update func::FuncOp's return types
  SmallVector<Operation *, 4> returnOps;
  funcOp.walk([&](Operation *op) {
    if (auto add_op = dyn_cast<func::ReturnOp>(op)) {
      returnOps.push_back(op);
    }
  });
  for (auto op : returnOps) {
    OpBuilder returnRewriter(op);
    // Cast the return values
    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      Value arg = op->getOperand(i);
      if (MemRefType type = llvm::dyn_cast<MemRefType>(arg.getType())) {
        Type etype = type.getElementType();
        if (llvm::isa<IntegerType>(etype)) {
          if (auto allocOp = dyn_cast<memref::AllocOp>(arg.getDefiningOp())) {
            bool is_unsigned = false;
            if (i < otypes.length()) {
              is_unsigned = otypes[i] == 'u';
            }
            auto result = allocOp.getResult();
            Value newMemRef = castIntMemRef(returnRewriter, op->getLoc(),
                                            result, 64, is_unsigned, false);
            // Only replace the single use of oldMemRef: returnOp
            op->setOperand(i, newMemRef);
          }
        }
      }
    }
    // Cast the input arguments
    for (auto v : llvm::enumerate(newMemRefs)) {
      Value newMemRef = v.value();
      Value &blockArg = blockArgs[v.index()];
      bool is_unsigned = false;
      if (v.index() < itypes.length()) {
        is_unsigned = itypes[v.index()] == 'u';
      }
      castIntMemRef(returnRewriter, op->getLoc(), newMemRef, 64, is_unsigned,
                    false, blockArg);
    }
  }

  // Update function signature
  FunctionType newFuncType =
      FunctionType::get(funcOp.getContext(), new_arg_types, new_result_types);
  funcOp.setType(newFuncType);
}

/// entry point
bool applyAnyWidthInteger(ModuleOp &mod) {
  // Find top-level function
  bool isFoundTopFunc = false;
  func::FuncOp *topFunc;
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    if (func->hasAttr("top")) {
      isFoundTopFunc = true;
      topFunc = &func;
      break;
    }
  }

  if (isFoundTopFunc && topFunc) {
    updateTopFunctionSignature(*topFunc);
  }

  return true;
}

} // namespace allo
} // namespace mlir

namespace {

struct AlloAnyWidthIntegerTransformation
    : public mlir::allo::impl::AnyWidthIntegerBase<
          AlloAnyWidthIntegerTransformation> {

  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyAnyWidthInteger(mod))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createAnyWidthIntegerPass() {
  return std::make_unique<AlloAnyWidthIntegerTransformation>();
}

} // namespace allo
} // namespace mlir