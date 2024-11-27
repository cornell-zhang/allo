/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// LowerCompositeType Pass
// This file defines the lowering of composite types such as structs.
// This pass is separated from AlloToLLVM because it could be used
// in other backends as well, such as HLS backend.
//===----------------------------------------------------------------------===//

#include "allo/Conversion/Passes.h"
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

void deadStructConstructElimination(func::FuncOp &func) {
  SmallVector<Operation *, 8> structConstructOps;
  func.walk([&](Operation *op) {
    if (auto structConstructOp = dyn_cast<StructConstructOp>(op)) {
      structConstructOps.push_back(structConstructOp);
    }
  });
  std::reverse(structConstructOps.begin(), structConstructOps.end());
  for (auto op : structConstructOps) {
    auto structValue = op->getResult(0);
    if (structValue.use_empty()) {
      op->erase();
    }
  }
}

void deadMemRefAllocElimination(func::FuncOp &func) {
  SmallVector<Operation *, 8> memRefAllocOps;
  func.walk([&](Operation *op) {
    if (auto memRefAllocOp = dyn_cast<memref::AllocOp>(op)) {
      memRefAllocOps.push_back(memRefAllocOp);
    }
  });
  std::reverse(memRefAllocOps.begin(), memRefAllocOps.end());
  for (auto op : memRefAllocOps) {
    auto v = op->getResult(0);
    if (v.use_empty()) {
      op->erase();
    }
  }
}

void deadAffineLoadElimination(func::FuncOp &func) {
  SmallVector<Operation *, 8> affineLoadOps;
  func.walk([&](Operation *op) {
    if (auto affineLoadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      affineLoadOps.push_back(affineLoadOp);
    }
  });
  std::reverse(affineLoadOps.begin(), affineLoadOps.end());
  for (auto op : affineLoadOps) {
    auto v = op->getResult(0);
    if (v.use_empty()) {
      op->erase();
    }
  }
}

void lowerStructType(func::FuncOp &func) {

  SmallVector<Operation *, 10> structGetOps;
  func.walk([&](Operation *op) {
    if (auto structGetOp = dyn_cast<StructGetOp>(op)) {
      structGetOps.push_back(structGetOp);
    }
  });

  std::map<mlir::detail::ValueImpl *, SmallVector<Value, 8>>
      structMemRef2fieldMemRefs;

  for (auto op : structGetOps) {
    // Collect info from structGetOp
    auto structGetOp = dyn_cast<StructGetOp>(op);
    Value struct_value = structGetOp->getOperand(0);
    Value struct_field = structGetOp->getResult(0);
    Location loc = op->getLoc();
    auto index = structGetOp.getIndex();

    // This flag indicates whether we can erase
    // struct construct and relevant ops.
    bool erase_struct_construct = false;

    // The defOp can be either a StructConstructOp or
    // a load from a memref.
    // Load: we are operating on a memref of struct
    // Construct: we are operating on a struct value
    Operation *defOp = struct_value.getDefiningOp();
    if (auto affine_load = dyn_cast<affine::AffineLoadOp>(defOp)) {
      // Case 1: defOp is loadOp from memref
      // Note: the idea to lower struct from memref is to
      // first create a memref for each struct field, and then
      // add store operation to those memrefs right after the
      // struct construction. With that, we can replace
      // struct get with loading from field memrefs.

      // Step1: create memref for each field
      Value struct_memref = affine_load.getMemref();
      // Try to find field_memrefs associated with this struct_memref
      SmallVector<Value, 4> field_memrefs;
      auto it = structMemRef2fieldMemRefs.find(struct_memref.getImpl());
      if (it == structMemRef2fieldMemRefs.end()) {
        // Create a memref for each field
        OpBuilder builder(struct_memref.getDefiningOp());
        StructType struct_type = struct_value.getType().cast<StructType>();

        for (Type field_type : struct_type.getElementTypes()) {
          MemRefType newMemRefType = struct_memref.getType()
                                         .cast<MemRefType>()
                                         .clone(field_type)
                                         .cast<MemRefType>();
          Value field_memref =
              builder.create<memref::AllocOp>(loc, newMemRefType);
          field_memrefs.push_back(field_memref);
        }
        structMemRef2fieldMemRefs.insert(
            std::make_pair(struct_memref.getImpl(), field_memrefs));
        erase_struct_construct = true;
      } else {
        field_memrefs.append(it->second);
        erase_struct_construct = false;
      }

      // Step2: add store to each field memref
      for (auto &use : struct_memref.getUses()) {
        if (auto storeOp = dyn_cast<affine::AffineStoreOp>(use.getOwner())) {
          // Find a storeOp to the struct memref, we add
          // store to each field memref here.
          OpBuilder builder(storeOp);
          for (const auto &field_memref_en : llvm::enumerate(field_memrefs)) {
            auto field_memref = field_memref_en.value();
            auto field_index = field_memref_en.index();
            // Find the struct_construct op
            auto struct_construct_op = dyn_cast<StructConstructOp>(
                storeOp.getOperand(0).getDefiningOp());
            builder.create<affine::AffineStoreOp>(
                loc, struct_construct_op.getOperand(field_index), field_memref,
                storeOp.getAffineMap(), storeOp.getIndices());
          }
          // erase the storeOp that stores to the struct memref
          if (erase_struct_construct) {
            storeOp.erase();
          }
          break;
        }
      }

      // Step3: replace structGetOp with load from field memrefs
      OpBuilder load_builder(op);
      Value loaded_field = load_builder.create<affine::AffineLoadOp>(
          loc, field_memrefs[index], affine_load.getAffineMap(),
          affine_load.getIndices());
      struct_field.replaceAllUsesWith(loaded_field);
      op->erase(); // erase structGetOp
    } else if (auto structConstructOp = dyn_cast<StructConstructOp>(defOp)) {
      // Case 2: defOp is a struct construction op
      Value replacement = defOp->getOperand(index);
      struct_field.replaceAllUsesWith(replacement);
      op->erase(); // erase structGetOp
    } else {
      llvm_unreachable("unexpected defOp for structGetOp");
    }
  }

  // Run DCE after all struct get is folded
  deadAffineLoadElimination(func);
  deadStructConstructElimination(func);
  deadMemRefAllocElimination(func);
}

Value buildStructFromInt(OpBuilder &builder, Location loc, Value int_value,
                         StructType struct_type, int lo) {
  SmallVector<Value, 4> struct_elements;
  for (Type field_type : struct_type.cast<StructType>().getElementTypes()) {
    if (field_type.isa<IntegerType>()) {
      int field_bitwidth = field_type.getIntOrFloatBitWidth();
      int hi = lo + (field_bitwidth - 1);
      Value hi_idx = builder.create<mlir::arith::ConstantIndexOp>(loc, hi);
      Value lo_idx = builder.create<mlir::arith::ConstantIndexOp>(loc, lo);
      lo += field_bitwidth;
      Value field_value = builder.create<mlir::allo::GetIntSliceOp>(
          loc, field_type, int_value, hi_idx, lo_idx);
      struct_elements.push_back(field_value);
    } else if (field_type.isa<StructType>()) {
      struct_elements.push_back(buildStructFromInt(
          builder, loc, int_value, field_type.cast<StructType>(), lo));
    } else {
      llvm_unreachable("unexpected type");
    }
  }
  return builder.create<StructConstructOp>(loc, struct_type, struct_elements);
}

void lowerIntToStructOp(func::FuncOp &func) {
  SmallVector<Operation *, 10> intToStructOps;
  func.walk([&](Operation *op) {
    if (auto intToStructOp = dyn_cast<IntToStructOp>(op)) {
      intToStructOps.push_back(intToStructOp);
    }
  });

  for (auto op : intToStructOps) {
    auto intToStructOp = dyn_cast<IntToStructOp>(op);
    Value struct_value = intToStructOp->getResult(0);
    Value int_value = intToStructOp->getOperand(0);
    Location loc = op->getLoc();
    // Step1: create get_bit op for each field
    StructType struct_type = struct_value.getType().cast<StructType>();
    OpBuilder builder(op);
    int lo = 0;
    // Step2: create struct construct op
    auto struct_construct =
        buildStructFromInt(builder, loc, int_value, struct_type, lo);
    // Step3: replace intToStructOp with struct construct
    struct_value.replaceAllUsesWith(struct_construct);
  }

  // Erase intToStructOps
  for (auto op : intToStructOps) {
    op->erase();
  }
}

bool isLegal(func::FuncOp &func) {
  bool legal = true;
  func.walk([&](Operation *op) {
    if (auto structGetOp = dyn_cast<StructGetOp>(op)) {
      legal = false;
      llvm::errs()
          << "Error: [Pass][LowerCompositeType] structGetOp is not legal: "
          << *op << "\n";
      WalkResult::interrupt();
    } else if (auto structConstructOp = dyn_cast<StructConstructOp>(op)) {
      legal = false;
      llvm::errs() << "Error: [Pass][LowerCompositeType] structConstructOp is "
                      "not legal: "
                   << *op << "\n";
      WalkResult::interrupt();
    }
  });
  return legal;
}

/// Pass entry point
bool applyLowerCompositeType(ModuleOp &mod) {
  // First check if there are any struct operations to lower
  bool hasStructOps = false;
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<StructGetOp, StructConstructOp, IntToStructOp>(op)) {
        hasStructOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasStructOps)
      break;
  }

  // If no struct operations, return success without doing anything
  if (!hasStructOps) {
    return true;
  }

  // Only apply transformations if we found struct operations
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    lowerIntToStructOp(func);
  }
  // Only run DCE if we actually did some transformations
  applyMemRefDCE(mod);

  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    lowerStructType(func);
  }
  // Run final DCE pass
  applyMemRefDCE(mod);

  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    if (!isLegal(func)) {
      return false;
    }
  }
  return true;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerCompositeTypeTransformation
    : public LowerCompositeTypeBase<AlloLowerCompositeTypeTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerCompositeType(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createLowerCompositeTypePass() {
  return std::make_unique<AlloLowerCompositeTypeTransformation>();
}
} // namespace allo
} // namespace mlir