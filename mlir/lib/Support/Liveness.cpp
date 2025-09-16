/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/Liveness.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {
// ! These functions may fail if multiple first/last use are in branches
Operation *getFirstUse(Value value, Operation &func) {
  func::FuncOp funcOp = llvm::cast<mlir::func::FuncOp>(func);
  DominanceInfo domInfo(&func);
  OpOperand *firstUse = nullptr;
  for (auto &use : value.getUses()) {
    auto *op = use.getOwner();
    if (!firstUse || domInfo.properlyDominates(op, firstUse->getOwner())) {
      firstUse = &use;
      //   llvm::errs() << firstUse->get() << "\n";
      //   llvm::errs() << *firstUse->getOwner() << "\n";
    }
  }
  if (!firstUse) {
    return nullptr;
  } else {
    return firstUse->getOwner();
  }
}

Operation *getLastUse(Value value, Operation &func) {
  func::FuncOp funcOp = llvm::cast<mlir::func::FuncOp>(func);
  PostDominanceInfo postDom(&func);
  OpOperand *lastUse = nullptr;
  for (auto &use : value.getUses()) {
    auto *op = use.getOwner();
    if (!lastUse || postDom.properlyPostDominates(op, lastUse->getOwner())) {
      lastUse = &use;
      //   llvm::errs() << lastUse->get() << "\n";
      //   llvm::errs() << *lastUse->getOwner() << "\n";
    }
  }
  if (!lastUse) {
    return nullptr;
  } else {
    return lastUse->getOwner();
  }
}
} // namespace allo
} // namespace mlir