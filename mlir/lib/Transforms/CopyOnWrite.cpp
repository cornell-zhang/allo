/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// CopyOnWrtite Pass
// This pass avoids copying data until the source have to be modifed
// TODO: better solution for 'last use' and 'all use after'
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

void removeRedundentCopy(func::FuncOp &func) {
  SmallVector<Operation *, 8> copyOps;
  func.walk([&](Operation *op) {
    if (auto memRefCopyOp = dyn_cast<memref::CopyOp>(op)) {
      copyOps.push_back(memRefCopyOp);
    } else if (auto linalgCopyOp = dyn_cast<linalg::CopyOp>(op)) {
      copyOps.push_back(linalgCopyOp);
    }
  });
  for (auto op : copyOps) {
    // llvm::errs() << *op << "\n";
    auto src = op->getOperand(0);
    auto dst = op->getOperand(1);

    bool resolvable = true;
    Operation *last_user = nullptr;
    for (auto &use : src.getUses()) {
      Operation *user = use.getOwner();
      if (last_user == nullptr) {
        last_user = user;
        continue;
      }
      if (user->getBlock() != op->getBlock()) {
        // Use in a different block, don't trust isBeforeInBlock
        resolvable = false;
        break;
      }
      if (last_user->isBeforeInBlock(user)) {
        last_user = user;
      }
    }
    // if copy is the last use of src
    if (resolvable && last_user == op) {
      // if dst is local
      if (dst.getDefiningOp<memref::AllocOp>()) {
        for (auto &use : dst.getUses()) {
          Operation *user = use.getOwner();
          if (user->getBlock() != op->getBlock()) {
            resolvable = false;
            break;
          }
        }
        if (resolvable) {
          for (auto &use : llvm::make_early_inc_range(dst.getUses())) {
            Operation *user = use.getOwner();
            if (user->getBlock() == op->getBlock() &&
                op->isBeforeInBlock(user)) {
              use.set(src);
            }
          }
          op->erase();
        }
      }
      // if dst is not local, but src is local
      else if (src.getDefiningOp<memref::AllocOp>()) {
        for (auto &use : dst.getUses()) {
          Operation *user = use.getOwner();
          if (user->getBlock() != op->getBlock() || user->isBeforeInBlock(op)) {
            resolvable = false;
            break;
          }
        }
        if (resolvable) {
          for (auto &use : llvm::make_early_inc_range(src.getUses())) {
            Operation *user = use.getOwner();
            use.set(dst);
          }
          op->erase();
        }
      }
    }
  }
}

/// Pass entry point
bool applyCopyOnWrite(ModuleOp &mod) {
  for (auto func : mod.getOps<func::FuncOp>()) {
    removeRedundentCopy(func);
  }
  return true;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloCopyOnWriteTransformation
    : public CopyOnWriteBase<AlloCopyOnWriteTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyCopyOnWrite(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createCopyOnWritePass() {
  return std::make_unique<AlloCopyOnWriteTransformation>();
}
} // namespace allo
} // namespace mlir