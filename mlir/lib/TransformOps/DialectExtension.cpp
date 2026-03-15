/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"

#include "allo/TransformOps/AlloTransformOps.h"
#include "allo/TransformOps/Utils.h"

using namespace mlir;
using namespace mlir::allo;

#define GET_OP_CLASSES
#include "allo/TransformOps/AlloTransformOps.cpp.inc"

namespace {
class AlloTransformDialectExtension
    : public transform::TransformDialectExtension<
          AlloTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlloTransformDialectExtension)

  using Base::Base;

  void init() {
    declareDependentDialect<allo::AlloDialect>();
    declareDependentDialect<linalg::LinalgDialect>();
    declareDependentDialect<affine::AffineDialect>();
    declareDependentDialect<arith::ArithDialect>();
    declareDependentDialect<scf::SCFDialect>();
    declareDependentDialect<tensor::TensorDialect>();
    declareDependentDialect<vector::VectorDialect>();
    declareDependentDialect<memref::MemRefDialect>();
    declareDependentDialect<bufferization::BufferizationDialect>();
    declareDependentDialect<math::MathDialect>();
    declareDependentDialect<LLVM::LLVMDialect>();

    // clang-format off
    registerTransformOps<
#define GET_OP_LIST
#include "allo/TransformOps/AlloTransformOps.cpp.inc"
        >();
    // clang-format on
  }
};
} // namespace

void allo::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<AlloTransformDialectExtension>();
}

///===----------------------------------------------------------------------===//
/// RenameOp implementation
///===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::RenameOp::applyToOne(transform::TransformRewriter &rewriter,
                                Operation *target,
                                transform::ApplyToEachResultList &results,
                                transform::TransformState &state) {
  if (isa<SymbolOpInterface>(target)) {
    Operation *symTableOp = SymbolTable::getNearestSymbolTable(target);
    if (!symTableOp)
      return emitSilenceableError() << "cannot find symbol table for target";

    SymbolTable symTable(symTableOp);
    if (failed(symTable.rename(target, getName()))) {
      return emitSilenceableError() << "failed to rename symbol";
    }
  } else {
    target->setAttr(OpIdentifier, getNameAttr());
  }
  return DiagnosedSilenceableFailure::success();
}

///===----------------------------------------------------------------------===//
/// MatchValueOp implementation
///===----------------------------------------------------------------------===//
namespace {
enum class MatchValueSourceKind : int64_t {
  Auto = 0,
  BlockArgument = 1,
  Result = 2,
};

SmallVector<BlockArgument> getBlockArguments(Operation *op) {
  SmallVector<BlockArgument> blockArgs;
  for (auto &region : op->getRegions()) {
    for (auto &block : region.getBlocks()) {
      llvm::append_range(blockArgs, block.getArguments());
    }
  }
  return blockArgs;
}
} // namespace

/// The result values must be of type `OpResult` or `BlockArgument`
DiagnosedSilenceableFailure
transform::MatchValueOp::applyToOne(transform::TransformRewriter &rewriter,
                                    Operation *target,
                                    transform::ApplyToEachResultList &results,
                                    transform::TransformState &state) {
  int64_t number = getNumberAttr().getInt();
  int64_t sourceKindValue = getSourceKindAttr().getInt();
  auto sourceKind =
      static_cast<MatchValueSourceKind>(static_cast<int64_t>(sourceKindValue));

  auto blockArgs = getBlockArguments(target);
  auto tryPushResult = [&]() -> DiagnosedSilenceableFailure {
    if (number < 0 ||
        static_cast<uint64_t>(number) >= target->getNumResults()) {
      return emitSilenceableError()
             << "result number out of bounds for the target operation";
    }
    results.push_back(target->getResult(number));
    return DiagnosedSilenceableFailure::success();
  };

  auto tryPushBlockArg = [&]() -> DiagnosedSilenceableFailure {
    if (number < 0 || static_cast<uint64_t>(number) >= blockArgs.size()) {
      return emitSilenceableError()
             << "block argument number out of bounds for the target operation";
    }
    results.push_back(blockArgs[number]);
    return DiagnosedSilenceableFailure::success();
  };

  switch (sourceKind) {
  case MatchValueSourceKind::Auto:
    if (blockArgs.empty())
      return tryPushResult();
    return tryPushBlockArg();
  case MatchValueSourceKind::BlockArgument:
    return tryPushBlockArg();
  case MatchValueSourceKind::Result:
    return tryPushResult();
  default:
    return emitSilenceableError()
           << "invalid source_kind " << sourceKindValue
           << " (expected 0:auto, 1:block_arg, 2:result)";
  }
}

///===----------------------------------------------------------------------===//
/// PartitionOp implementation
///===----------------------------------------------------------------------===//
namespace {
constexpr StringLiteral kPartitionAttrName = "allo.part";

// merge two partition attributes. `b` will override `a` if there is a conflict.
PartitionAttr mergePartitionAttrs(PartitionAttr a, PartitionAttr b) {
  if (!a)
    return b;
  if (!b)
    return a;

  auto partA = a.getPartitions();
  auto partB = b.getPartitions();
  std::map<int64_t, std::pair<PartitionKindEnum, int64_t>> partMap;
  // init with A
  for (auto part : partA) {
    partMap[part.getDim()] = {part.getKind(), part.getFactor()};
  }
  // merge B to A
  for (auto part : partB) {
    partMap[part.getDim()] = {part.getKind(), part.getFactor()};
  }
  // part map is an ordered map
  // no need to sort the dims after merging

  // construct the merged partition attribute
  SmallVector<PartitionAxisAttr, 4> mergedParts;
  MLIRContext *ctx = a.getContext();
  for (auto &[dim, info] : partMap) {
    auto attr = PartitionAxisAttr::get(ctx, info.first, info.second, dim);
    mergedParts.push_back(attr);
  }

  return PartitionAttr::get(ctx, mergedParts);
}
} // namespace

DiagnosedSilenceableFailure
transform::PartitionOp::apply(transform::TransformRewriter &rewriter,
                              transform::TransformResults &results,
                              transform::TransformState &state) {
  auto newPart = getPartition();

  for (Value value : state.getPayloadValues(getTarget())) {
    auto memrefType = dyn_cast<MemRefType>(value.getType());
    if (!memrefType)
      return emitSilenceableError()
             << "partition target value must be of memref type";

    Value root = resolveMemRefValueRoot(value);
    auto rootMemrefType = dyn_cast<MemRefType>(root.getType());
    if (!rootMemrefType)
      return emitSilenceableError()
             << "cannot resolve allocation point for the memref value";

    Attribute oldAttr;
    Operation *attrOwner = nullptr;
    std::optional<unsigned> argNumber;

    if (auto arg = dyn_cast<BlockArgument>(root)) {
      // Case 1: memref introduced as a block argument.
      auto func = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp());
      if (!func) {
        return emitSilenceableError() << "partition target root block argument "
                                         "must belong to a kernel op";
      }
      attrOwner = func;
      argNumber = arg.getArgNumber();
      oldAttr = func.getArgAttr(*argNumber, kPartitionAttrName);
    } else {
      // Case 2: memref introduced by an alloc-like op.
      Operation *defOp = root.getDefiningOp();
      if (!defOp)
        return emitSilenceableError()
               << "cannot resolve partition root for value";

      if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(defOp)) {
        auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
            getGlobal, getGlobal.getNameAttr());
        if (!global) {
          return emitSilenceableError()
                 << "failed to find memref.global @" << getGlobal.getName();
        }
        attrOwner = global;
      } else {
        if (!isa<memref::AllocOp, memref::AllocaOp>(defOp)) {
          return emitSilenceableError()
                 << "partition target root must resolve to memref.alloc, "
                    "memref.alloca, memref.get_global, or function argument";
        }
        attrOwner = defOp;
      }
      oldAttr = attrOwner->getAttr(kPartitionAttrName);
    }

    if (oldAttr && !isa<PartitionAttr>(oldAttr)) {
      return emitSilenceableError()
             << "existing " << kPartitionAttrName
             << " attribute is not a partition attribute";
    }

    auto oldPart = dyn_cast_if_present<PartitionAttr>(oldAttr);
    auto mergedPart = mergePartitionAttrs(oldPart, newPart);

    if (argNumber.has_value()) {
      auto kernel = cast<func::FuncOp>(attrOwner);
      rewriter.modifyOpInPlace(kernel, [&]() {
        kernel.setArgAttr(*argNumber, kPartitionAttrName, mergedPart);
      });
    } else {
      rewriter.modifyOpInPlace(attrOwner, [&]() {
        attrOwner->setAttr(kPartitionAttrName, mergedPart);
      });
    }
  }

  return DiagnosedSilenceableFailure::success();
}
