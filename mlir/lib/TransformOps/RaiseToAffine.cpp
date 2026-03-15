/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/TransformOps/AlloTransformOps.h"
#include "allo/TransformOps/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::allo;

/// Try to interpret select(cmp, a, b) as min/max(a,b).
/// If successful, returns {isMax, x, y} where result == (isMax? max(x,y) :
/// min(x,y)). Supports lhs/rhs swapping and true/false swapping for
/// sge/sgt/sle/slt (also u* variants).
/// Case 1:
/// %cmp = cmpi sge/uge/sgt/ugt %a, %b
/// $res = select %cmp, %a, %b => max(%a, %b)
/// Case 2:
/// %cmp = cmpi sge/uge/sgt/ugt %a, %b
/// $res = select %cmp, %b, %a => min(%a, %b)
/// Case 3:
/// %cmp = cmpi sle/ule/slt/ult %a, %b
/// $res = select %cmp, %a, %b => min(%a, %b)
/// Case 4:
/// %cmp = cmpi sle/ule/slt/ult %a, %b
/// $res = select %cmp, %b, %a => max(%a, %b)
static FailureOr<std::tuple<bool, Value, Value>>
matchSelectAsMinMax(arith::SelectOp sel) {
  auto cmp = sel.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return failure();

  auto lhs = cmp.getLhs();
  auto rhs = cmp.getRhs();
  auto t = sel.getTrueValue();
  auto f = sel.getFalseValue();
  auto pred = cmp.getPredicate();

  auto isGE = [](arith::CmpIPredicate p) {
    using P = arith::CmpIPredicate;
    return p == P::sge || p == P::sgt || p == P::uge || p == P::ugt;
  };

  auto isLE = [](arith::CmpIPredicate p) {
    using P = arith::CmpIPredicate;
    return p == P::sle || p == P::slt || p == P::ule || p == P::ult;
  };

  auto checkSwapped = [&](Value a, Value b) -> std::optional<bool> {
    if (t == a && f == b)
      return false; // not swapped
    if (t == b && f == a)
      return true; // swapped
    return std::nullopt;
  };

  auto swappedOr = checkSwapped(lhs, rhs);
  if (!swappedOr) {
    // Not a min/max pattern if true/false doesn't match (lhs,rhs) or (rhs,lhs).
    return failure();
  }
  bool swapped = *swappedOr;

  bool predIsGE = isGE(pred);
  bool predIsLE = isLE(pred);
  if (!predIsGE && !predIsLE) {
    // eq/ne predicates don't match min/max patterns.
    return failure();
  }

  if (!swapped) {
    if (predIsGE)
      return std::make_tuple(true, lhs, rhs); // max
    return std::make_tuple(false, lhs, rhs);  // min
  }
  if (predIsGE)
    return std::make_tuple(false, lhs, rhs); // min
  return std::make_tuple(true, lhs, rhs);    // max
}

namespace {
struct AffineParallelBoundSet {
  SmallVector<AffineMap, 4> maps;
  SmallVector<Value, 8> operands;
  AffineParallelBoundSet(SmallVectorImpl<AffineMap> &&maps,
                         SmallVectorImpl<Value> &&operands)
      : maps(std::move(maps)), operands(std::move(operands)) {}
};
} // namespace

static LogicalResult collectBoundExprs(AffineValueMapBuilder &builder,
                                       Value value, bool isLowerBound);

static LogicalResult collectBinaryBoundExprs(AffineValueMapBuilder &builder,
                                             Value lhs, Value rhs,
                                             bool isLowerBound) {
  if (failed(collectBoundExprs(builder, lhs, isLowerBound)))
    return failure();
  if (failed(collectBoundExprs(builder, rhs, isLowerBound)))
    return failure();
  return success();
}

static LogicalResult collectBoundExprs(AffineValueMapBuilder &builder,
                                       Value value, bool isLowerBound) {
  Value v = stripCast(value);
  Operation *defOp = v.getDefiningOp();

  // Expand arithmetic max for lower bounds and min for upper bounds.
  if (isLowerBound) {
    if (auto maxOp = dyn_cast_or_null<arith::MaxSIOp>(defOp)) {
      return collectBinaryBoundExprs(builder, maxOp.getLhs(), maxOp.getRhs(),
                                     isLowerBound);
    }
    if (auto maxOp = dyn_cast_or_null<arith::MaxUIOp>(defOp)) {
      return collectBinaryBoundExprs(builder, maxOp.getLhs(), maxOp.getRhs(),
                                     isLowerBound);
    }
  } else {
    if (auto minOp = dyn_cast_or_null<arith::MinSIOp>(defOp)) {
      return collectBinaryBoundExprs(builder, minOp.getLhs(), minOp.getRhs(),
                                     isLowerBound);
    }
    if (auto minOp = dyn_cast_or_null<arith::MinUIOp>(defOp)) {
      return collectBinaryBoundExprs(builder, minOp.getLhs(), minOp.getRhs(),
                                     isLowerBound);
    }
  }

  // Expand select-as-min/max if it matches the bound direction.
  if (auto sel = dyn_cast_or_null<arith::SelectOp>(defOp)) {
    auto match = matchSelectAsMinMax(sel);
    if (succeeded(match)) {
      auto [isMax, x, y] = *match;
      if ((isLowerBound && isMax) || (!isLowerBound && !isMax)) {
        return collectBinaryBoundExprs(builder, x, y, isLowerBound);
      }
    }
  }

  // Expand affine.max/affine.min by importing all map results.
  if (auto maxOp = dyn_cast_or_null<affine::AffineMaxOp>(defOp)) {
    if (!isLowerBound)
      return failure();
    return builder.importMapAndOperands(
        maxOp.getAffineMap(), maxOp.getDimOperands(), maxOp.getSymbolOperands(),
        /*allowMultiResults=*/true);
  }
  if (auto minOp = dyn_cast_or_null<affine::AffineMinOp>(defOp)) {
    if (isLowerBound)
      return failure();
    return builder.importMapAndOperands(
        minOp.getAffineMap(), minOp.getDimOperands(), minOp.getSymbolOperands(),
        /*allowMultiResults=*/true);
  }

  // Leaf case: import as a single affine expression.
  // This covers plain dim/symbol/constant expressions and affine.apply chains.
  if (failed(builder.importValue(value)))
    return failure();
  return success();
}

static FailureOr<affine::AffineValueMap>
matchAffineBound(AffineValueMapBuilder &builder, Value root,
                 bool isLowerBound) {
  builder.reset();
  if (failed(collectBoundExprs(builder, root, isLowerBound)))
    return failure();
  return builder.compose();
}

static FailureOr<AffineParallelBoundSet>
normalizeParallelBounds(ArrayRef<affine::AffineValueMap> bounds) {
  if (bounds.empty())
    return failure();

  MLIRContext *ctx = bounds.front().getAffineMap().getContext();
  llvm::SmallMapVector<Value, unsigned, 8> globalDims;
  llvm::SmallMapVector<Value, unsigned, 8> globalSyms;

  auto registerOperand = [&](Value v) -> LogicalResult {
    v = stripCast(v);
    auto addDim = [&]() {
      if (!globalDims.contains(v)) {
        globalDims[v] = globalDims.size();
      }
      return success();
    };
    auto addSym = [&]() {
      if (!globalSyms.count(v)) {
        globalSyms[v] = globalSyms.size();
      }
      return success();
    };
    // Prefer dim if both are legal.
    if (affine::isValidDim(v))
      return addDim();
    if (affine::isValidSymbol(v))
      return addSym();
    return failure();
  };

  for (const auto &bound : bounds) {
    for (Value operand : bound.getOperands()) {
      if (failed(registerOperand(operand)))
        return failure();
    }
  }

  SmallVector<AffineMap, 4> normalizedMaps;
  normalizedMaps.reserve(bounds.size());

  auto getGlobalExpr = [&](Value v) {
    v = stripCast(v);
    if (globalDims.contains(v)) {
      return getAffineDimExpr(globalDims[v], ctx);
    }
    if (globalSyms.contains(v)) {
      return getAffineSymbolExpr(globalSyms[v], ctx);
    }
    llvm_unreachable("operand was not registered as either dim or symbol");
  };

  for (const auto &bound : bounds) {
    SmallVector<AffineExpr, 4> dimExprs;
    SmallVector<AffineExpr, 4> symExprs;

    unsigned nDims = bound.getNumDims();
    unsigned nSyms = bound.getNumSymbols();
    auto operands = bound.getOperands();
    assert(nDims + nSyms == operands.size());

    for (unsigned i = 0; i < nDims; ++i) {
      dimExprs.push_back(getGlobalExpr(operands[i]));
    }
    for (unsigned i = 0; i < nSyms; ++i) {
      symExprs.push_back(getGlobalExpr(operands[nDims + i]));
    }

    SmallVector<AffineExpr, 4> results;
    for (AffineExpr expr : bound.getAffineMap().getResults())
      results.push_back(expr.replaceDimsAndSymbols(dimExprs, symExprs));

    // Rebuild each bound map in a shared input space required by
    // affine.parallel.
    normalizedMaps.push_back(
        AffineMap::get(globalDims.size(), globalSyms.size(), results, ctx));
  }
  auto operands = llvm::to_vector<8>(globalDims.keys());
  llvm::append_range(operands, globalSyms.keys());
  return AffineParallelBoundSet(std::move(normalizedMaps), std::move(operands));
}

static FailureOr<affine::AffineValueMap>
computeAffineMapAndArgs(AffineValueMapBuilder &builder, ValueRange indices) {
  for (Value idx : indices) {
    // Build one affine map result per index expression.
    auto expr = builder.importValue(idx);
    if (failed(expr))
      return failure();
  }
  return builder.compose();
}

// used for static assert
template <class> inline constexpr bool dependent_false_v = false;

template <typename OpTy>
static void updateIndicesToAffineMapOrApply(RewriterBase &b,
                                            AffineValueMapBuilder &builder,
                                            OpTy op) {
  OpBuilder::InsertionGuard g(b);
  builder.reset();
  auto indices = op.getIndices();
  auto mapOr = computeAffineMapAndArgs(builder, indices);
  if (failed(mapOr)) {
    return;
  }
  auto map = mapOr->getAffineMap();
  auto operands = mapOr->getOperands();
  b.setInsertionPoint(op);
  if constexpr (std::is_same_v<OpTy, memref::LoadOp>) {
    b.replaceOpWithNewOp<affine::AffineLoadOp>(op, op.getMemRef(), map,
                                               operands);
  } else if constexpr (std::is_same_v<OpTy, memref::StoreOp>) {
    b.replaceOpWithNewOp<affine::AffineStoreOp>(op, op.getValueToStore(),
                                                op.getMemRef(), map, operands);
  }
  // used to support stream access ops with SSA indices
  // else if constexpr (std::is_base_of_v<ChannelAccessOpInterface, OpTy>) {
  //   // replace indices with the result of affine.apply
  //   SmallVector<Value, 4> newIndices;
  //   for (unsigned i = 0; i < indices.size(); ++i) {
  //     auto subMap = map.getSubMap(i);
  //     auto apply = affine::AffineApplyOp::create(
  //         b, op.getLoc(), indices[i].getType(), subMap, operands);
  //     newIndices.push_back(apply);
  //   }
  //   b.modifyOpInPlace(op, [&]() { op.getIndicesMutable().assign(newIndices);
  //   });
  // }
  else {
    static_assert(dependent_false_v<OpTy>,
                  "unsupported op type for updateIndicesToAffineApply");
  }
}

static void raiseAccesses(AffineValueMapBuilder &builder,
                          transform::TransformRewriter &rewriter,
                          Operation *root) {
  root->walk([&](Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      updateIndicesToAffineMapOrApply(rewriter, builder, load);
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      updateIndicesToAffineMapOrApply(rewriter, builder, store);
    }
    // else if (auto chanOp = dyn_cast<ChannelAccessOpInterface>(op)) {
    //   updateIndicesToAffineApply(rewriter, builder, chanOp);
    // }
  });
}

static std::optional<int64_t> getConstPositiveStep(Value step) {
  IntegerAttr::ValueType cst;
  if (!matchPattern(step, m_ConstantInt(&cst)))
    return std::nullopt;
  int64_t stepVal = cst.getSExtValue();
  if (stepVal <= 0)
    return std::nullopt;
  return stepVal;
}

/// raise `scf.for` to `affine.for` if the bounds and step are affine.
static DiagnosedSilenceableFailure
raiseForOp(transform::TransformRewriter &rewriter, scf::ForOp forOp,
           transform::ApplyToEachResultList &results,
           transform::TransformState &state) {
  AffineValueMapBuilder builder(forOp.getContext());
  rewriter.setInsertionPoint(forOp);
  auto lbOr =
      matchAffineBound(builder, forOp.getLowerBound(), /*isLowerBound=*/true);
  if (failed(lbOr)) {
    return emitSilenceableFailure(forOp)
           << "lower bound does not match affine bound pattern";
  }
  auto ubOr =
      matchAffineBound(builder, forOp.getUpperBound(), /*isLowerBound=*/false);
  if (failed(ubOr)) {
    return emitSilenceableFailure(forOp)
           << "upper bound does not match affine bound pattern";
  }

  affine::AffineValueMap &lb = *lbOr;
  affine::AffineValueMap &ub = *ubOr;
  Value step = stripCast(forOp.getStep());

  // Fast path: constant positive step
  if (auto stepInt = getConstPositiveStep(step)) {
    auto affineLoop = affine::AffineForOp::create(
        rewriter, forOp->getLoc(), lb.getOperands(), lb.getAffineMap(),
        ub.getOperands(), ub.getAffineMap(), *stepInt, forOp.getInitArgs());
    // merge loop body into the new loop and update induction variable uses.
    Block *affineBody = affineLoop.getBody();
    // delete the implicit yield
    if (affineLoop->getNumResults() == 0) {
      affineBody->getTerminator()->erase();
    }
    // manually create affine.yield
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    rewriter.setInsertionPoint(yield);
    affine::AffineYieldOp::create(rewriter, yield->getLoc(),
                                  yield->getOperands());
    yield->erase();
    // Merge the rest of the body into the new loop. The source block has one
    // IV argument plus all loop-carried region arguments, so provide a full
    // replacement list in that order.
    SmallVector<Value, 4> argRepls;
    argRepls.push_back(affineLoop.getInductionVar());
    for (Value iterArg : affineLoop.getRegionIterArgs())
      argRepls.push_back(iterArg);
    rewriter.mergeBlocks(forOp.getBody(), affineLoop.getBody(), argRepls);
    // After raising loops, try to raise in-body memref accesses to affine ops.
    raiseAccesses(builder, rewriter, affineLoop);
    if (forOp->hasAttr(OpIdentifier))
      affineLoop->setAttr(OpIdentifier, forOp->getAttr(OpIdentifier));
    rewriter.replaceOp(forOp, affineLoop);
    results.push_back(affineLoop);
    return DiagnosedSilenceableFailure::success();
  }
  return emitSilenceableFailure(forOp)
         << "step is not a constant positive integer";
}

static DiagnosedSilenceableFailure
raiseParallelOp(transform::TransformRewriter &rewriter, scf::ParallelOp parOp,
                transform::ApplyToEachResultList &results,
                transform::TransformState &state) {
  AffineValueMapBuilder builder(parOp.getContext());
  if (parOp.getNumReductions() != 0) {
    // Keep this path conservative for now; only non-reduction parallel loops
    // are raised in this transform.
    return emitSilenceableFailure(parOp)
           << "parallel reduction is not supported yet";
  }

  unsigned numLoops = parOp.getNumLoops();
  auto lowerBounds = parOp.getLowerBound();
  auto upperBounds = parOp.getUpperBound();
  auto stepValues = parOp.getStep();
  SmallVector<affine::AffineValueMap, 4> lbs;
  SmallVector<affine::AffineValueMap, 4> ubs;
  SmallVector<int64_t, 4> steps;
  lbs.reserve(numLoops);
  ubs.reserve(numLoops);
  steps.reserve(numLoops);

  for (unsigned i = 0; i < numLoops; ++i) {
    auto lbOr =
        matchAffineBound(builder, lowerBounds[i], /*isLowerBound=*/true);
    if (failed(lbOr)) {
      return emitSilenceableFailure(parOp)
             << "lower bound of loop dim " << i
             << " does not match affine bound pattern";
    }
    lbs.push_back(*lbOr);

    auto ubOr =
        matchAffineBound(builder, upperBounds[i], /*isLowerBound=*/false);
    if (failed(ubOr)) {
      return emitSilenceableFailure(parOp)
             << "upper bound of loop dim " << i
             << " does not match affine bound pattern";
    }
    ubs.push_back(*ubOr);

    Value step = stripCast(stepValues[i]);
    auto stepInt = getConstPositiveStep(step);
    if (!stepInt) {
      return emitSilenceableFailure(parOp)
             << "step of loop dim " << i
             << " is not a constant positive integer";
    }
    steps.push_back(*stepInt);
  }

  auto normLBs = normalizeParallelBounds(lbs);
  if (failed(normLBs)) {
    return emitSilenceableFailure(parOp)
           << "failed to normalize lower bounds to affine.parallel input space";
  }
  auto normUBs = normalizeParallelBounds(ubs);
  if (failed(normUBs)) {
    return emitSilenceableFailure(parOp)
           << "failed to normalize upper bounds to affine.parallel input space";
  }

  rewriter.setInsertionPoint(parOp);
  SmallVector<arith::AtomicRMWKind> reductions;
  // Construct affine.parallel with normalized per-dimension min/max bounds.
  auto affineParallel = affine::AffineParallelOp::create(
      rewriter, parOp.getLoc(), TypeRange{}, reductions, normLBs->maps,
      normLBs->operands, normUBs->maps, normUBs->operands, steps);

  Block *affineBody = affineParallel.getBody();
  if (affineParallel->getNumResults() == 0)
    affineBody->getTerminator()->erase();

  auto reduce = cast<scf::ReduceOp>(parOp.getBody()->getTerminator());
  rewriter.setInsertionPoint(reduce);
  affine::AffineYieldOp::create(rewriter, reduce.getLoc(),
                                reduce.getOperands());
  reduce->erase();

  ValueRange affineIvs(affineParallel.getIVs());
  rewriter.mergeBlocks(parOp.getBody(), affineParallel.getBody(), affineIvs);
  raiseAccesses(builder, rewriter, affineParallel);
  if (parOp->hasAttr(OpIdentifier))
    affineParallel->setAttr(OpIdentifier, parOp->getAttr(OpIdentifier));
  rewriter.replaceOp(parOp, affineParallel->getResults());
  results.push_back(affineParallel);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::RaiseToAffineOp::applyToOne(
    TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (auto forOp = dyn_cast<scf::ForOp>(target)) {
    return raiseForOp(rewriter, forOp, results, state);
  }
  if (auto parOp = dyn_cast<scf::ParallelOp>(target)) {
    return raiseParallelOp(rewriter, parOp, results, state);
  }
  if (isa<affine::AffineForOp, affine::AffineParallelOp>(target)) {
    // Already affine loops; just try to raise in-body memref accesses to affine
    AffineValueMapBuilder builder(target->getContext());
    raiseAccesses(builder, rewriter, target);
    results.push_back(target);
    return DiagnosedSilenceableFailure::success();
  }
  return emitSilenceableFailure(target)
         << "expected scf.for or scf.parallel, but got " << target->getName();
}
