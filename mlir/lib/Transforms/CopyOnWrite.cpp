/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// CopyOnWrite Pass
// This pass avoids copying data until the source have to be modifed
// TODO: better solution for 'last use' and 'all use after'
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Support/Liveness.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

void removeRedundantCopy(func::FuncOp &func) {
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

    auto srcType = src.getType().dyn_cast<MemRefType>();
    auto dstType = dst.getType().dyn_cast<MemRefType>();
    if (!srcType || !dstType || srcType != dstType) {
      continue;
    }

    bool resolvable = true;
    Operation *last_user = getLastUse(src, *func);

    // if copy is the last use of src
    if (last_user && last_user == op) {
      Operation *defOp = dst.getDefiningOp();
      // if dst is local, replace the use of dst with src
      // TODO: A more precise check should detect actual local def
      if (defOp && !llvm::isa<memref::SubViewOp>(defOp)) {
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
    removeRedundantCopy(func);
  }
  return true;
}

void applyCopyOnWriteOnFunction(Operation &func) {
  func::FuncOp funcOp = llvm::cast<mlir::func::FuncOp>(func);
  removeRedundantCopy(funcOp);
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