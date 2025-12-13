/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

void removeStrideMap(func::FuncOp &func) {
  SmallVector<Operation *, 8> allocOps;
  func.walk([&](Operation *op) {
    if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
      allocOps.push_back(alloc);
    }
  });

  for (auto op : allocOps) {
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    MemRefType memRefType = llvm::cast<MemRefType>(allocOp.getType());
    auto memRefMaps = memRefType.getLayout();
    if (memRefMaps.getAffineMap().isIdentity() ||
        memRefMaps.getAffineMap().isEmpty()) {
      continue;
    }
    auto newMemRefType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType());
    op->getResult(0).setType(newMemRefType);
  }

  FunctionType functionType = func.getFunctionType();
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 8> arg_types;
  for (const auto &argEn : llvm::enumerate(func.getArguments()))
    arg_types.push_back(argEn.value().getType());
  SmallVector<Type, 4> new_result_types;
  SmallVector<Type, 8> new_arg_types;
  for (auto result_type : result_types) {
    if (llvm::isa<MemRefType>(result_type)) {
      MemRefType new_result_type;
      if (auto layout = dyn_cast<AffineMapAttr>(
              llvm::cast<MemRefType>(result_type).getLayout())) {
        // array partition
        new_result_type = MemRefType::get(
            llvm::cast<MemRefType>(result_type).getShape(),
            llvm::cast<MemRefType>(result_type).getElementType());
      } else {
        new_result_type = MemRefType::get(
            llvm::cast<MemRefType>(result_type).getShape(),
            llvm::cast<MemRefType>(result_type).getElementType(),
            llvm::cast<MemRefType>(result_type).getLayout());
      }
      new_result_types.push_back(new_result_type);
    } else {
      new_result_types.push_back(result_type);
    }
  }
  for (auto arg_type : arg_types) {
    if (llvm::isa<MemRefType>(arg_type)) {
      MemRefType new_arg_type;
      if (auto layout = dyn_cast<AffineMapAttr>(
              llvm::cast<MemRefType>(arg_type).getLayout())) {
        // array partition
        new_arg_type =
            MemRefType::get(llvm::cast<MemRefType>(arg_type).getShape(),
                            llvm::cast<MemRefType>(arg_type).getElementType());
      } else {
        new_arg_type =
            MemRefType::get(llvm::cast<MemRefType>(arg_type).getShape(),
                            llvm::cast<MemRefType>(arg_type).getElementType(),
                            llvm::cast<MemRefType>(arg_type).getLayout());
      }
      new_arg_types.push_back(new_arg_type);
    } else {
      new_arg_types.push_back(arg_type);
    }
  }

  for (Block &block : func.getBlocks()) {
    for (unsigned i = 0; i < block.getNumArguments(); i++) {
      block.getArgument(i).setType(new_arg_types[i]);
    }
  }

  FunctionType new_function_type =
      FunctionType::get(func.getContext(), new_arg_types, new_result_types);
  func.setType(new_function_type);
}

/// Pass entry point
bool applyRemoveStrideMap(ModuleOp &module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func.getBlocks().empty()) {
      continue;
    }
    removeStrideMap(func);
  }
  return true;
}

} // namespace allo
} // namespace mlir

namespace {
struct AlloRemoveStrideMapTransformation
    : public mlir::allo::impl::RemoveStrideMapBase<
          AlloRemoveStrideMapTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyRemoveStrideMap(mod)) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {
std::unique_ptr<OperationPass<ModuleOp>> createRemoveStrideMapPass() {
  return std::make_unique<AlloRemoveStrideMapTransformation>();
}
} // namespace allo
} // namespace mlir