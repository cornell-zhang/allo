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
    }
  }
  if (!lastUse) {
    return nullptr;
  } else {
    return lastUse->getOwner();
  }
}

Operation *getNextUse(Value value, Operation *curUse, Operation &func) {
  func::FuncOp funcOp = llvm::cast<mlir::func::FuncOp>(func);
  DominanceInfo domInfo(&func);
  OpOperand *nextUse = nullptr;
  for (auto &use : value.getUses()) {
    Operation *useOp = use.getOwner();
    if (useOp != curUse && domInfo.properlyDominates(curUse, useOp)) {
      if (!nextUse || domInfo.properlyDominates(useOp, nextUse->getOwner())) {
        nextUse = &use;
      }
    }
  }
  if (!nextUse) {
    return nullptr;
  } else {
    return nextUse->getOwner();
  }
}

} // namespace allo
} // namespace mlir