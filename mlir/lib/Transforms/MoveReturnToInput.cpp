/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// MoveReturnToInput Pass
// This pass is to support multiple return values for LLVM backend.
// The input program may have multiple return values
// The output program has no return values, all the return values
// is moved to the input argument list
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

void moveReturnToInput(func::FuncOp &funcOp) {
  FunctionType functionType = funcOp.getFunctionType();
  SmallVector<Type, 4> resTypes = llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 4> argTypes;
  for (auto &arg : funcOp.getArguments()) {
    argTypes.push_back(arg.getType());
  }

  // Create corresponding block args for return values
  SmallVector<BlockArgument, 4> blockArgs;
  for (Block &block : funcOp.getBlocks()) {
    for (Type t : resTypes) {
      auto bArg = block.addArgument(t, funcOp.getLoc());
      blockArgs.push_back(bArg);
    }
  }

  // Find the allocation op for return values
  // and replace their uses with newly created block args
  SmallVector<Operation *, 4> returnOps;
  funcOp.walk([&](Operation *op) {
    if (auto add_op = dyn_cast<func::ReturnOp>(op)) {
      returnOps.push_back(op);
    }
  });

  // Build loops to copy return values to block args
  OpBuilder builder(funcOp.getContext());
  // build right before the terminator
  builder.setInsertionPointToEnd(&funcOp.getBlocks().back());
  for (auto op : returnOps) {
    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      Value arg = op->getOperand(i);
      if (MemRefType type = llvm::dyn_cast<MemRefType>(arg.getType())) {
        BlockArgument bArg = blockArgs[i];
        // build an memref.copy op to copy the return value to block arg
        builder.create<memref::CopyOp>(op->getLoc(), arg, bArg);
      } else {
        // issue an error
        op->emitError("MoveReturnToInput Pass does not support non-memref "
                      "return values now.");
      }
    }
    // erase return op
    op->erase();
  }
  // build a new empty return op
  builder.create<func::ReturnOp>(funcOp.getLoc());

  // Append resTypes to argTypes and clear resTypes
  argTypes.insert(std::end(argTypes), std::begin(resTypes), std::end(resTypes));
  resTypes.clear();
  // Update function signature
  FunctionType newFuncType =
      FunctionType::get(funcOp.getContext(), argTypes, resTypes);
  funcOp.setType(newFuncType);
}

/// entry point
bool applyMoveReturnToInput(ModuleOp &mod) {
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
    moveReturnToInput(*topFunc);
  }
  return true;
}

} // namespace allo
} // namespace mlir

namespace {

struct AlloMoveReturnToInputTransformation
    : public mlir::allo::impl::MoveReturnToInputBase<
          AlloMoveReturnToInputTransformation> {

  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyMoveReturnToInput(mod))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createMoveReturnToInputPass() {
  return std::make_unique<AlloMoveReturnToInputTransformation>();
}

} // namespace allo
} // namespace mlir