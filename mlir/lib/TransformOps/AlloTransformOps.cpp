/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "allo/Dialect/AlloAttrs.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/RegionUtils.h"

#include "allo/Dialect/AlloTransformOps.h"

using namespace mlir;

/// --------------------------------------------------------------
/// Allo Transform Dialect Extension
/// --------------------------------------------------------------

class AlloTransformDialectExtension
    : public transform::TransformDialectExtension<
          AlloTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlloTransformDialectExtension)

  AlloTransformDialectExtension() {
    registerTransformOps<
#define GET_OP_LIST
#include "allo/Dialect/AlloTransformOps.cpp.inc"

        >();
  }
};

#define GET_OP_CLASSES
#include "allo/Dialect/AlloTransformOps.cpp.inc"

void allo::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<AlloTransformDialectExtension>();
}

/// --------------------------------------------------------------
/// Utilities
/// --------------------------------------------------------------

static bool checkSplitFactor(affine::AffineForOp forOp, unsigned factor) {
  int64_t lb = forOp.getConstantLowerBound();
  int64_t ub = forOp.getConstantUpperBound();
  // in affine.for, ub > lb is always true. step < 0 is illegal.
  return ub - lb >= static_cast<int64_t>(factor);
}

// check if all loops are in the same perfectly nested loop band
// the loops don't need to be in the order of depth nor adjacent
// return the top-level loop of the band if true, otherwise return nullptr
static affine::AffineForOp
inSamePerfectlyNestedLoopBand(const ArrayRef<affine::AffineForOp> &loops) {
  if (loops.empty())
    return {};
  if (loops.size() == 1)
    return {};
  // create a temp copy and sort by depth
  auto tmp = llvm::to_vector(loops);
  DenseMap<affine::AffineForOp, unsigned> depthMap;
  std::for_each(tmp.begin(), tmp.end(), [&depthMap](auto op) {
    unsigned depth = 0;
    Operation *curr = op;
    while ((curr = curr->getParentOp()))
      depth++;
    depthMap[op] = depth;
  });
  std::sort(tmp.begin(), tmp.end(),
            [&depthMap](auto a, auto b) { return depthMap[a] < depthMap[b]; });

  // no need to be contiguous
  // check perfectly nested
  for (unsigned i = 0; i < tmp.size() - 1; ++i) {
    affine::AffineForOp currLoop = tmp[i];
    affine::AffineForOp nextLoop = tmp[i + 1];
    // check if they are in the same loop nest
    if (!currLoop->isProperAncestor(nextLoop)) {
      return {};
    }
    // check if perfectly nested between currLoop and nextLoop
    Operation *ptr = currLoop.getOperation();
    while (ptr != nextLoop.getOperation()) {
      auto loop = dyn_cast<affine::AffineForOp>(ptr);
      if (!loop) {
        return {};
      }
      Block *body = loop.getBody();
      // the first one is affine.for
      // the second one is the terminator
      if (body->getOperations().size() != 2) {
        return {};
      }
      ptr = &body->getOperations().front();
    }
  }
  // return the top-level loop
  return tmp.front();
}

/// --------------------------------------------------------------
/// LoopSplit Op
/// --------------------------------------------------------------

DiagnosedSilenceableFailure
transform::LoopSplitOp::applyToOne(transform::TransformRewriter &rewriter,
                                   Operation *target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {

  auto factor = getFactor();
  if (auto forOp = dyn_cast<affine::AffineForOp>(target)) {
    // check lb and ub
    if (!forOp.hasConstantBounds()) {
      return emitSilenceableError()
             << "only loops with constant bounds are supported";
    }
    if (!checkSplitFactor(forOp, factor)) {
      return emitSilenceableError()
             << "loop range is smaller than the split factor";
    }
    // perform split
    SmallVector<affine::AffineForOp> splitOps;
    if (failed(affine::tilePerfectlyNested(forOp, factor, &splitOps))) {
      return emitSilenceableError() << "failed to split the loop";
    }

    // normalize loop
    auto outerLoop = splitOps.front();
    auto innerLoop = splitOps.back();
    if (failed(affine::normalizeAffineFor(outerLoop)) ||
        failed(affine::normalizeAffineFor(innerLoop))) {
      return emitSilenceableError() << "failed to normalize the loop";
    }
    AffineMap innerUb = innerLoop.getUpperBoundMap();
    if (innerUb.isConstant() && innerUb.getNumInputs() != 0) {
      // a special case: affine_map<(d0) -> (32)>.
      // the result is constant but the map has a redundant dimension.
      // reconstruct the map to () -> (32)
      auto cstUb =
          dyn_cast<AffineConstantExpr>(innerUb.getResult(0)).getValue();
      rewriter.setInsertionPoint(innerLoop);
      innerLoop.setUpperBound({}, rewriter.getConstantAffineMap(cstUb));
    }

    // sink apply ops
    for (auto &outerOp :
         llvm::make_early_inc_range(outerLoop.getBody()->getOperations())) {
      if (auto applyOp = dyn_cast<affine::AffineApplyOp>(outerOp)) {
        // no need to check dominance
        applyOp->moveBefore(&innerLoop.getBody()->front());
        break;
      }
    }
    results.push_back(outerLoop);
    results.push_back(innerLoop);
  } else {
    return emitSilenceableError() << "expected an affine.for operation";
  }
  // record results
  return DiagnosedSilenceableFailure::success();
}

/// --------------------------------------------------------------
/// LoopTile Op
/// --------------------------------------------------------------

DiagnosedSilenceableFailure
transform::LoopTileOp::apply(transform::TransformRewriter &rewriter,
                             transform::TransformResults &results,
                             transform::TransformState &state) {
  SmallVector<affine::AffineForOp> band;
  // validate input operation handles
  for (auto payload : getLoops()) {
    auto ops = llvm::to_vector(state.getPayloadOps(payload));
    for (auto op : ops) {
      auto forOp = dyn_cast<affine::AffineForOp>(op);
      if (!forOp) {
        return emitSilenceableError() << "expected an affine.for operation";
      }
      band.push_back(forOp);
    }
  }
  if (!affine::isPerfectlyNested(band)) {
    return emitSilenceableError() << "loops must be perfectly nested";
  }

  // validate tile sizes
  auto tileSizes = getTileSizes();
  if (tileSizes.size() != band.size()) {
    return emitSilenceableError()
           << "number of tile sizes must match the number of loops";
  }
  SmallVector<unsigned> factors;
  for (auto [loop, factor] : llvm::zip_equal(band, tileSizes)) {
    if (!loop.hasConstantBounds())
      return emitSilenceableError()
             << "only loops with constant bounds are supported";
    if (!checkSplitFactor(loop, static_cast<unsigned>(factor))) {
      return emitSilenceableError()
             << "loop range is smaller than the tile size";
    }
    factors.push_back(static_cast<unsigned>(factor));
  }

  // perform tiling
  SmallVector<affine::AffineForOp> tiledLoops;
  if (failed(affine::tilePerfectlyNested(band, factors, &tiledLoops))) {
    return emitSilenceableError() << "failed to tile the loop nest";
  }

  // normalize tiled loops
  for (auto loop : tiledLoops) {
    if (failed(affine::normalizeAffineFor(loop))) {
      return emitSilenceableError() << "failed to normalize the loop";
    }
  }

  // simplify point loops' upper bounds if necessary
  // after the simplication, the upper bound map should be constant or only
  // consisted of the tile loops' ivs.
  std::size_t nLoops = band.size();
  for (auto loop : llvm::drop_begin(tiledLoops, nLoops)) {
    AffineMap ubMap = loop.getUpperBoundMap();
    if (ubMap.isConstant() && ubMap.getNumInputs() != 0) {
      auto cstUb = dyn_cast<AffineConstantExpr>(ubMap.getResult(0)).getValue();
      rewriter.setInsertionPoint(loop);
      loop.setUpperBound({}, rewriter.getConstantAffineMap(cstUb));
    } else if (!ubMap.isConstant()) {
      // try to simplify a min/max affine_map.
      // after the tiling, if this situation happens, then the map must be in
      // the form of affine_map<(d0) -> min(cst, expr)>.
      // where cst @ result(0), expr @ result(1)
      assert(ubMap.getNumResults() == 2);
      assert(ubMap.getNumInputs() == 1);
      auto addMap =
          AffineMap::get(/*dimCount*/ 1, /*symbolCount*/ 0, ubMap.getResult(1));
      // we cannot cast without previous loop normalization
      auto applyOp = cast<affine::AffineApplyOp>(
          loop.getUpperBoundOperands()[0].getDefiningOp());
      auto outerIV = applyOp.getOperand(0);
      AffineMap mulMap = applyOp.getAffineMap();
      AffineMap composed = addMap.compose(mulMap);
      SmallVector<AffineExpr, 2> exprs{ubMap.getResult(0),
                                       composed.getResult(0)};
      AffineMap final = AffineMap::get(/*dimCount*/ 1, /*symbolCount*/ 0, exprs,
                                       rewriter.getContext());
      loop.setUpperBound(outerIV, final);
    }
  }

  // sink apply ops from outer tile loops to inner point loops
  for (unsigned i = 0; i < nLoops; ++i) {
    affine::AffineForOp tile = tiledLoops[i];
    affine::AffineForOp point = tiledLoops[i + nLoops];
    for (auto &outerOp :
         llvm::make_early_inc_range(tile.getBody()->getOperations())) {
      if (auto applyOp = dyn_cast<affine::AffineApplyOp>(outerOp)) {
        // no need to check dominance
        applyOp->moveBefore(&point.getBody()->front());
        break;
      }
    }
  }

  // record results
  // we pack corresponding tile and point loops together
  for (unsigned i = 0; i < nLoops; ++i) {
    SmallVector<affine::AffineForOp, 2> pack{tiledLoops[i],
                                             tiledLoops[i + nLoops]};
    results.set(cast<OpResult>(getTiledLoops()[i]), pack);
  }
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::LoopTileOp::verify() {
  unsigned nInputs = getLoops().size();
  unsigned nResults = getTiledLoops().size();
  if (nInputs != nResults) {
    return emitOpError(
               "number of tiled loops must match the number of input loops")
           << ", got " << nResults << " results for " << nInputs << " loops";
  }
  return success();
}

void transform::LoopTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getLoopsMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

/// --------------------------------------------------------------
/// LoopReorder Op
/// --------------------------------------------------------------

DiagnosedSilenceableFailure
transform::LoopReorderOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {
  SmallVector<affine::AffineForOp> loops;
  // validate input operation handles
  for (auto payload : getLoops()) {
    auto ops = llvm::to_vector(state.getPayloadOps(payload));
    for (auto op : ops) {
      auto forOp = dyn_cast<affine::AffineForOp>(op);
      if (!forOp) {
        return emitSilenceableError() << "expected an affine.for operation";
      }
      loops.push_back(forOp);
    }
  }
  if (loops.size() < 2) {
    return emitSilenceableError()
           << "at least two loops are required for reordering";
  }
  affine::AffineForOp outermostLoop = inSamePerfectlyNestedLoopBand(loops);
  if (!outermostLoop) {
    return emitSilenceableError()
           << "loops must be in the same perfectly nested loop band";
  }
  // validate permutation
  unsigned nLoops = getLoops().size();
  auto permutation = getPermutation();
  unsigned nPerm = permutation.size();
  if (nLoops != nPerm) {
    return emitSilenceableError()
           << "the size of permutation must match the number of loops";
  }

  // get the entire loop nest
  SmallVector<affine::AffineForOp> band;
  affine::getPerfectlyNestedLoops(band, outermostLoop);

  // construct complete permutation map
  SmallVector<unsigned> selectedOrgIndices;
  for (auto l : loops) {
    auto it = llvm::find(band, l);
    unsigned idx = std::distance(band.begin(), it);
    selectedOrgIndices.push_back(idx);
  }
  SmallVector<unsigned> permMap(band.size());
  std::iota(permMap.begin(), permMap.end(), 0u);
  for (unsigned i = 0; i < nPerm; ++i) {
    unsigned targetPos = selectedOrgIndices[i];
    unsigned srcPos = selectedOrgIndices[permutation[i]];
    permMap[targetPos] = srcPos;
  }

  // perform reordering
  if (!affine::isValidLoopInterchangePermutation(band, permMap)) {
    return emitSilenceableError() << "permutation violates data dependencies";
  }
  affine::permuteLoops(band, permMap);

  // record results
  // actually we don't need to update the loop handles, they are still valid
  // after affine::permuteLoops
  for (unsigned i = 0; i < loops.size(); i++) {
    results.set(cast<OpResult>(getReorderedLoops()[i]), {loops[i]});
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::LoopReorderOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getLoopsMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

LogicalResult transform::LoopReorderOp::verify() {
  // we cannot know the number of loops at verification time
  // so we only check the validity of the permutation itself
  unsigned nPerm = getPermutation().size();
  auto permutation = getPermutation();
  for (unsigned i = 0; i < nPerm; ++i) {
    if (static_cast<uint32_t>(permutation[i]) >= nPerm) {
      return emitOpError("permutation index out of bounds: ") << permutation[i];
    }
    for (unsigned j = i + 1; j < nPerm; ++j) {
      if (permutation[i] == permutation[j]) {
        return emitOpError("permutation contains duplicate index: ")
               << permutation[i];
      }
    }
  }
  unsigned nLoops = getLoops().size();
  unsigned nResults = getReorderedLoops().size();
  if (nLoops != nResults) {
    return emitOpError(
               "number of reordered loops must match the number of input loops")
           << ", got " << nResults << " results for " << nLoops << " loops";
  }
  return success();
}

/// --------------------------------------------------------------
/// LoopUnroll Op
/// --------------------------------------------------------------

DiagnosedSilenceableFailure
transform::LoopUnrollOp::applyToOne(transform::TransformRewriter &rewriter,
                                    Operation *target,
                                    transform::ApplyToEachResultList &results,
                                    transform::TransformState &state) {
  if (!isa<affine::AffineForOp, scf::ForOp>(target)) {
    return emitSilenceableError()
           << "expected an affine.for or scf.for operation";
  }
  auto factor = getFactor().value_or(0u); // 0u = fully unroll
  target->setAttr("unroll", rewriter.getUI32IntegerAttr(factor));
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

/// --------------------------------------------------------------
/// LoopParallel Op
/// --------------------------------------------------------------

DiagnosedSilenceableFailure
transform::LoopParallelOp::applyToOne(transform::TransformRewriter &rewriter,
                                      Operation *target,
                                      transform::ApplyToEachResultList &results,
                                      transform::TransformState &state) {
  if (!isa<affine::AffineForOp, scf::ForOp>(target)) {
    return emitSilenceableError()
           << "expected an affine.for or scf.for operation";
  }
  target->setAttr("parallel", rewriter.getUnitAttr());
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

/// --------------------------------------------------------------
/// LoopPipeline Op
/// --------------------------------------------------------------

DiagnosedSilenceableFailure
transform::LoopPipelineOp::applyToOne(transform::TransformRewriter &rewriter,
                                      Operation *target,
                                      transform::ApplyToEachResultList &results,
                                      transform::TransformState &state) {
  if (!isa<affine::AffineForOp, scf::ForOp>(target)) {
    return emitSilenceableError()
           << "expected an affine.for or scf.for operation";
  }
  auto ii = getInterval().value_or(1); // default ii = 1
  target->setAttr("pipeline_ii", rewriter.getUI32IntegerAttr(ii));
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

/// --------------------------------------------------------------
/// LoopFuse Op
/// --------------------------------------------------------------

// modified from lib/Transforms/Utils/LoopUtils.cpp
static void coalesceLoops(MutableArrayRef<affine::AffineForOp> loops,
                          transform::TransformRewriter &rewriter) {
  // RAII helper to restore the insertion point.
  OpBuilder::InsertionGuard guard(rewriter);

  affine::AffineForOp innermost = loops.back();
  affine::AffineForOp outermost = loops.front();
  affine::AffineBound ub = outermost.getUpperBound();
  Location loc = outermost.getLoc();
  SmallVector<Value, 4> upperBoundSymbols;
  SmallVector<Value, 4> ubOperands(ub.getOperands().begin(),
                                   ub.getOperands().end());

  // 1. Store the upper bound of the outermost loop in a variable.
  SmallVector<int64_t, 4> ubs;
  for (auto loop : loops) {
    auto cstUb = loop.getConstantUpperBound();
    ubs.push_back(cstUb);
  }

  // 2. Emit code computing the upper bound of the coalesced loop as product of
  // the number of iterations of all loops.
  int64_t prod =
      std::accumulate(ubs.begin(), ubs.end(), 1ll, std::multiplies<>());
  outermost.setConstantUpperBound(prod);

  // 3. Remap induction variables. For each original loop, the value of the
  // induction variable can be obtained by dividing the induction variable of
  // the linearized loop by the total number of iterations of the loops nested
  // in it modulo the number of iterations in this loop (remove the values
  // related to the outer loops):
  //   iv_i = floordiv(iv_linear, product-of-loop-ranges-until-i) mod range_i.
  // Compute these iteratively from the innermost loop by creating a "running
  // quotient" of division by the range.
  rewriter.setInsertionPointToStart(outermost.getBody());
  Value previous = outermost.getInductionVar();
  SmallVector<Operation *> opToSink;
  for (unsigned idx = loops.size(); idx > 0; --idx) {
    int64_t currUb = ubs[idx - 1];
    if (idx != loops.size()) {
      auto quotientMap =
          AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                         rewriter.getAffineDimExpr(0).floorDiv(currUb));
      previous =
          affine::AffineApplyOp::create(rewriter, loc, quotientMap, previous);
      opToSink.push_back(previous.getDefiningOp());
    }
    // Modified value of the induction variables of the nested loops after
    // coalescing.
    Value inductionVariable;
    if (idx == 1) {
      inductionVariable = previous;
    } else {
      auto modMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                   rewriter.getAffineDimExpr(0) % currUb);
      inductionVariable =
          affine::AffineApplyOp::create(rewriter, loc, modMap, previous);
      opToSink.push_back(inductionVariable.getDefiningOp());
    }
    replaceAllUsesInRegionWith(loops[idx - 1].getInductionVar(),
                               inductionVariable, loops.back().getRegion());
  }

  // 4. Move the operations from the innermost just above the second-outermost
  // loop, delete the extra terminator and the second-outermost loop.
  affine::AffineForOp secondOutermostLoop = loops[1];
  innermost.getBody()->back().erase();
  rewriter.inlineBlockBefore(
      innermost.getBody(), outermost.getBody(),
      Block::iterator(secondOutermostLoop.getOperation()));
  for (auto [iter, init] :
       llvm::zip_equal(secondOutermostLoop.getRegionIterArgs(),
                       secondOutermostLoop.getInits())) {
    iter.replaceAllUsesWith(init);
    iter.dropAllUses();
  }
  secondOutermostLoop.erase();

  // 5. Sink affine.apply operations.
  std::reverse(opToSink.begin(), opToSink.end());
  outermost.walk([&](affine::AffineForOp nestedLoop) {
    if (nestedLoop == outermost)
      return;
    bool canSinkAll = true;
    for (auto op : opToSink) {
      for (auto user : op->getUsers()) {
        if (!nestedLoop->isAncestor(user)) {
          canSinkAll = false;
          break;
        }
      }
      if (!canSinkAll)
        break;
    }
    if (canSinkAll) {
      Block *body = nestedLoop.getBody();
      for (auto op : opToSink) {
        op->moveBefore(&body->front());
      }
    }
  });
}

DiagnosedSilenceableFailure
transform::LoopFuseOp::apply(transform::TransformRewriter &rewriter,
                             transform::TransformResults &results,
                             transform::TransformState &state) {
  SmallVector<affine::AffineForOp> loops;
  // validate input operation handles
  for (auto payload : getLoops()) {
    auto ops = llvm::to_vector(state.getPayloadOps(payload));
    for (auto op : ops) {
      auto forOp = dyn_cast<affine::AffineForOp>(op);
      if (!forOp) {
        return emitSilenceableError() << "expected an affine.for operation";
      }
      if (forOp.getStepAsInt() != 1 || !forOp.hasConstantBounds() ||
          forOp.getConstantLowerBound() != 1) {
        return emitSilenceableError() << "only normalized loops with step=1 "
                                         "and lower bound=0 are supported";
      }
      loops.push_back(forOp);
    }
  }
  if (loops.size() < 2) {
    return DiagnosedSilenceableFailure::success();
  }
  if (!affine::isPerfectlyNested(loops)) {
    return emitSilenceableError() << "loops must be perfectly nested";
  }
  affine::AffineForOp outermostLoop = loops.front();
  while (auto curr = outermostLoop->getParentOfType<affine::AffineForOp>()) {
    outermostLoop = curr;
  }

  // perform flattening
  coalesceLoops(loops, rewriter);

  // record results
  results.set(cast<OpResult>(getFlattenedLoop()), {outermostLoop});
  return DiagnosedSilenceableFailure::success();
}

void transform::LoopFuseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getLoopsMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

/// --------------------------------------------------------------
/// LoopUnfold Op
/// --------------------------------------------------------------

DiagnosedSilenceableFailure
transform::LoopUnfoldOp::applyToOne(transform::TransformRewriter &rewriter,
                                    Operation *target,
                                    transform::ApplyToEachResultList &results,
                                    transform::TransformState &state) {
  if (!isa<affine::AffineForOp, scf::ForOp>(target)) {
    return emitSilenceableError()
           << "expected an affine.for or scf.for operation";
  }
  auto factor = getFactor().value_or(1u); // 1u = skip unfolding
  target->setAttr("unfold", rewriter.getUI32IntegerAttr(factor));
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

/// --------------------------------------------------------------
/// ComputeAt Op
/// --------------------------------------------------------------

static std::optional<std::string>
tryAffineLoopFusion(affine::AffineForOp producer, affine::AffineForOp consumer,
                    unsigned targetDepth) {
  using affine::FusionResult;
  affine::ComputationSliceState sliceState;
  affine::FusionStrategy strategy(affine::FusionStrategy::ProducerConsumer);
  FusionResult test = affine::canFuseLoops(producer, consumer, targetDepth,
                                           &sliceState, strategy);
  if (test.value == FusionResult::Success) {
    affine::fuseLoops(producer, consumer, sliceState);
    producer.erase();
    return std::nullopt;
  }
  std::string reason;
  if (test.value == FusionResult::FailPrecondition) {
    reason = "failed precondition for fusion (e.g. same block)";
  } else if (test.value == FusionResult::FailBlockDependence) {
    reason = "fusion would violate another dependence in block";
  } else if (test.value == FusionResult::FailFusionDependence) {
    reason = "fusion would reverse dependences between loops";
  } else if (test.value == FusionResult::FailComputationSlice) {
    reason = "unable to compute src loop computation slice";
  } else if (test.value == FusionResult::FailIncorrectSlice) {
    reason = "slice is computed, but it is incorrect";
  }
  return reason;
}

enum class DependenceType : uint8_t {
  NONE = 0,
  RAW = 1 << 1u,
  WAR = 1 << 2u,
  WAW = 1 << 3u,
};

DependenceType operator|(DependenceType a, DependenceType b) {
  return static_cast<DependenceType>(static_cast<uint8_t>(a) |
                                     static_cast<uint8_t>(b));
}

DependenceType operator&(DependenceType a, DependenceType b) {
  return static_cast<DependenceType>(static_cast<uint8_t>(a) &
                                     static_cast<uint8_t>(b));
}

// check dependencies between two affine.for loop nests up to a certain depth
// assume forOpA is source, forOpB is sink
// return a bitmask of DependenceType
static DependenceType checkDependencies(affine::AffineForOp forOpA,
                                        affine::AffineForOp forOpB,
                                        unsigned depth) {
  SmallVector<affine::MemRefAccess, 4> accA;
  SmallVector<affine::MemRefAccess, 4> accB;
  forOpA.walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface,
            memref::LoadOp, memref::StoreOp>(op)) {
      accA.emplace_back(op);
    }
  });
  forOpB.walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface,
            memref::LoadOp, memref::StoreOp>(op)) {
      accB.emplace_back(op);
    }
  });
  DependenceType ret = DependenceType::NONE;
  for (auto &a : accA) {
    for (auto &b : accB) {
      if (a.memref != b.memref) {
        continue; // different memrefs
      }
      if (!a.isStore() && !b.isStore()) {
        continue; // we don't care about rar
      }
      SmallVector<affine::DependenceComponent, 2> deps;
      if (affine::checkMemrefAccessDependence(a, b, depth, nullptr, &deps)
              .value == affine::DependenceResult::HasDependence) {
        if (a.isStore() && !b.isStore()) {
          ret = ret | DependenceType::RAW;
        } else if (!a.isStore() && b.isStore()) {
          ret = ret | DependenceType::WAR;
        } else if (a.isStore() && b.isStore()) {
          ret = ret | DependenceType::WAW;
        }
      }
    }
  }
  return ret;
}

DiagnosedSilenceableFailure
transform::ComputeAtOp::apply(transform::TransformRewriter &rewriter,
                              transform::TransformResults &results,
                              transform::TransformState &states) {
  // validate input operation handles
  auto targetLoops = states.getPayloadOps(getTargetLoop());
  auto computeOps = states.getPayloadOps(getComputeStage());
  if (!llvm::hasSingleElement(targetLoops) ||
      !llvm::hasSingleElement(computeOps)) {
    return emitSilenceableError() << "expected single operands";
  }
  Operation *targetOp = *targetLoops.begin();
  Operation *computeOp = *computeOps.begin();
  auto targetForOp = dyn_cast<affine::AffineForOp>(targetOp);
  if (!targetForOp) {
    return emitSilenceableError()
           << "expected an affine.for operation as target loop";
  }

  // 1. Get the root loop of the compute stage and target loop
  SmallVector<affine::AffineForOp> prodChain;
  auto currProd = computeOp->getParentOfType<affine::AffineForOp>();
  while (currProd) {
    prodChain.push_back(currProd);
    currProd = currProd->getParentOfType<affine::AffineForOp>();
  }
  if (!currProd) {
    return emitSilenceableError()
           << "compute stage operation is not inside any affine.for loop";
  }
  std::reverse(prodChain.begin(), prodChain.end());
  affine::AffineForOp computeRootForOp = prodChain.front();
  unsigned computeDepth = prodChain.size();

  SmallVector<affine::AffineForOp> targetChain;
  auto currTarget = targetForOp;
  while (currTarget) {
    targetChain.push_back(currTarget);
    currTarget = currTarget->getParentOfType<affine::AffineForOp>();
  }
  std::reverse(targetChain.begin(), targetChain.end());
  unsigned targetDepth = targetChain.size();
  affine::AffineForOp targetRootForOp = targetChain.front();

  if (targetRootForOp->getBlock() != computeRootForOp->getBlock()) {
    return emitSilenceableError()
           << "target loop and compute stage must be in the same block";
  }
  if (computeDepth < targetDepth) {
    return emitSilenceableError() << "producer is shallower than target axis";
  }

  SmallVector<Value, 4> computeIVs, targetIVs;
  for (auto l : prodChain) {
    computeIVs.push_back(l.getInductionVar());
  }
  for (auto l : targetChain) {
    targetIVs.push_back(l.getInductionVar());
  }

  // 2. Get loop bounds of the loop nests
  auto getConstantBounds = [](ArrayRef<affine::AffineForOp> chain,
                              unsigned depth) {
    SmallVector<int64_t, 4> lbs, ubs;
    for (unsigned i = 0; i < depth; ++i) {
      auto loop = chain[i];
      if (!loop.hasConstantBounds())
        return std::make_pair(SmallVector<int64_t, 4>{},
                              SmallVector<int64_t, 4>{});
      lbs.push_back(loop.getConstantLowerBound());
      ubs.push_back(loop.getConstantUpperBound());
    }
    return std::make_pair(lbs, ubs);
  };

  auto [prodLBs, prodUBs] = getConstantBounds(computeRootForOp, targetDepth);
  auto [targetLBs, targetUBs] = getConstantBounds(targetRootForOp, targetDepth);
  if (prodLBs.empty() || prodUBs.empty()) {
    return emitSilenceableError()
           << "only loops with constant bounds are supported";
  }

  // 3. check dependencies between the two loop nests
  auto depType = checkDependencies(computeRootForOp, targetForOp, targetDepth);

  // 4. try to merge the loop nests
  if ((depType & DependenceType::RAW) != DependenceType::NONE) {
    auto reason =
        tryAffineLoopFusion(computeRootForOp, targetForOp, targetDepth);
    if (reason.has_value()) {
      return emitSilenceableError()
             << "cannot merge the two loop nests: " << reason.value();
    }
  } else if (depType == DependenceType::NONE) {
    // two loop nests have no dependencies
    // we handle two cases:
    // 1) loop bounds are the same -> merge directly
    // 2) loop bounds are different but producer is a subset of consumer
    // other cases are not supported
    int64_t totalDiff = 0;
    for (unsigned i = 0; i < targetDepth; ++i) {
      totalDiff += std::abs(targetLBs[i] - prodLBs[i]) +
                   std::abs(targetUBs[i] - prodUBs[i]);
    }
    bool isSameDepth = (targetDepth == computeDepth);
    Operation *toMove =
        isSameDepth ? nullptr : prodChain[targetDepth].getOperation();

    if (totalDiff == 0) {
      // case 1: loop bounds are the same
      // simply move the computeOp into the target loop nest and update ivs
      if (isSameDepth) {
        Block *prodBody = prodChain.back().getBody();
        rewriter.eraseOp(prodBody->getTerminator());
        rewriter.inlineBlockBefore(prodBody, targetForOp.getBody(),
                                   targetForOp.getBody()->begin());
      } else {
        rewriter.moveOpBefore(toMove, targetForOp.getBody(),
                              targetForOp.getBody()->begin());
      }
      // replace ivs
      for (unsigned i = 0; i < targetDepth; ++i) {
        replaceAllUsesInRegionWith(computeIVs[i], targetIVs[i],
                                   targetForOp.getRegion());
      }
    } else {
      // case 2: check subset relationship
      for (unsigned i = 0; i < targetDepth; ++i) {
        if (prodLBs[i] < targetLBs[i] || prodUBs[i] > targetUBs[i]) {
          return emitSilenceableError() << "producer loop bounds must be a "
                                           "subset of target loop bounds";
        }
      }
      // construct integer set for affine.if
      rewriter.setInsertionPointToStart(targetForOp.getBody());
      SmallVector<AffineExpr, 4> exprs;
      for (unsigned i = 0; i < targetDepth; ++i) {
        exprs.push_back(rewriter.getAffineDimExpr(i) - prodLBs[i]);
        exprs.push_back(prodUBs[i] - 1 - rewriter.getAffineDimExpr(i));
      }
      SmallVector<bool, 4> flags(exprs.size(), false);
      auto set = IntegerSet::get(targetDepth, /*symbolCount*/ 0, exprs, flags);

      // construct affine.if
      auto ifOp = affine::AffineIfOp::create(
          rewriter, targetForOp.getLoc(), set, targetIVs, /*elseRegion*/ false);
      rewriter.setInsertionPointToStart(ifOp.getThenBlock());

      // move operations
      if (isSameDepth) {
        Block *prodBody = prodChain.back().getBody();
        rewriter.eraseOp(prodBody->getTerminator());
        rewriter.inlineBlockBefore(prodBody, ifOp.getThenBlock(),
                                   ifOp.getThenBlock()->begin());
      } else {
        rewriter.moveOpBefore(toMove, ifOp.getThenBlock(),
                              ifOp.getThenBlock()->begin());
      }
      // replace ivs
      for (unsigned i = 0; i < targetDepth; ++i) {
        replaceAllUsesInRegionWith(computeIVs[i], targetIVs[i],
                                   ifOp.getThenRegion());
      }
    }
    computeRootForOp.erase();
  }

  // 5. Perform store-to-load forwarding to eliminate redundant memory ops
  // find the func because affineScalarReplace needs it
  // TODO: we can optimize only the target loop nest region
  auto func = targetForOp->getParentOfType<func::FuncOp>();
  DominanceInfo dom(func);
  PostDominanceInfo pdom(func);
  AliasAnalysis aliasAnalysis(func);
  affine::affineScalarReplace(func, dom, pdom, aliasAnalysis);

  results.set(cast<OpResult>(getAttachedStage()), {targetForOp});
  return DiagnosedSilenceableFailure::success();
}

void transform::ComputeAtOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTargetLoopMutable(), effects);
  transform::consumesHandle(getComputeStageMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

/// --------------------------------------------------------------
/// Partition Op
/// --------------------------------------------------------------

DiagnosedSilenceableFailure
transform::PartitionOp::apply(transform::TransformRewriter &rewriter,
                              transform::TransformResults &results,
                              transform::TransformState &state) {
  auto payloadValues = state.getPayloadValues(getBuffer());
  SmallVector<Value, 4> transformed;

  for (Value value : payloadValues) {
    auto memrefType = dyn_cast<MemRefType>(value.getType());
    if (!memrefType) {
      return emitSilenceableError() << "expected memref type";
    }
    unsigned targetDim = getDim().value_or(0);
    unsigned rank = memrefType.getRank();
    if (targetDim >= rank) {
      return emitSilenceableError() << "dimension out of bounds";
    }
    unsigned factor = getFactor().value_or(0);
    if (targetDim > 0 && factor > memrefType.getShape()[targetDim - 1]) {
      return emitSilenceableError()
             << "partition factor is larger than the dimension size";
    }
    if (targetDim == 0) {
      for (unsigned i = 0; i < rank; ++i) {
        if (factor > memrefType.getShape()[i]) {
          return emitSilenceableError()
                 << "partition factor is larger than the dimension size";
        }
      }
    }
    auto type = getPartitionType();
    SmallVector<AffineExpr, 8> partitionIndices;
    SmallVector<AffineExpr, 8> addressIndices;

    for (unsigned i = 0; i < rank; ++i) {
      AffineExpr dimExpr = rewriter.getAffineDimExpr(i);
      AffineMap existingLayout = memrefType.getLayout().getAffineMap();
      bool hasExistingLayout = existingLayout.getNumResults() > rank;
      // dim=0 means all dims, dim>0 means specific dim (1-based)
      if (targetDim == 0 || (targetDim > 0 && i == targetDim - 1)) {
        if (type == allo::PartitionKindEnum::BlockPartition) {
          int64_t blockSize = (memrefType.getShape()[i] + factor - 1) / factor;
          partitionIndices.push_back(dimExpr.floorDiv(blockSize));
          addressIndices.push_back(dimExpr % blockSize);
        } else if (type == allo::PartitionKindEnum::CyclicPartition) {
          partitionIndices.push_back(dimExpr % factor);
          addressIndices.push_back(dimExpr.floorDiv(factor));
        } else if (type == allo::PartitionKindEnum::CompletePartition) {
          partitionIndices.push_back(dimExpr);
          addressIndices.push_back(rewriter.getAffineConstantExpr(0));
        }
      } else {
        if (hasExistingLayout) {
          partitionIndices.push_back(existingLayout.getResult(i));
          addressIndices.push_back(existingLayout.getResult(i + rank));
        } else {
          partitionIndices.push_back(rewriter.getAffineConstantExpr(0));
          addressIndices.push_back(dimExpr);
        }
      }
    }
    partitionIndices.append(addressIndices);
    auto layoutMap =
        AffineMap::get(rank, 0, partitionIndices, rewriter.getContext());
    auto newType =
        MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                        layoutMap, memrefType.getMemorySpace());
    rewriter.modifyOpInPlace(value.getDefiningOp()
                                 ? value.getDefiningOp()
                                 : value.getParentRegion()->getParentOp(),
                             [&]() { value.setType(newType); });

    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto func = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp());
      if (func) {
        auto inputTypes = llvm::to_vector(func.getArgumentTypes());
        inputTypes[arg.getArgNumber()] = newType;
        auto newFuncType = FunctionType::get(rewriter.getContext(), inputTypes,
                                             func.getResultTypes());
        func.setFunctionType(newFuncType);
      }
    }
    transformed.push_back(value);
  }
  results.setValues(cast<OpResult>(getPartitioned()), transformed);
  return DiagnosedSilenceableFailure::success();
}

void transform::PartitionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getBufferMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

/// --------------------------------------------------------------
/// ReuseAt Op
/// --------------------------------------------------------------

struct ExprCompare {
  static int64_t findConstantExpr(const AffineExpr &exp) {
    int64_t value = -1;
    // TODO: only support one constant now
    exp.walk([&](AffineExpr inner) {
      if (llvm::isa<AffineConstantExpr>(inner))
        value = llvm::cast<AffineConstantExpr>(inner).getValue();
    });
    return value;
  }
  bool operator()(const AffineExpr &exp1, const AffineExpr &exp2) const {
    int64_t val1 = findConstantExpr(exp1);
    int64_t val2 = findConstantExpr(exp2);
    return val1 < val2;
  }
};

static int findMemRefAxisFromIV(affine::AffineStoreOp storeOp, Value iv) {
  AffineMap map = storeOp.getAffineMap();
  auto operands = storeOp.getMapOperands();
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    AffineExpr expr = map.getResult(i);
    bool found = false;
    expr.walk([&](AffineExpr e) {
      if (auto d = dyn_cast<AffineDimExpr>(e)) {
        if (d.getPosition() < operands.size() &&
            operands[d.getPosition()] == iv) {
          found = true;
        }
      }
    });
    if (found)
      return static_cast<int>(i);
  }
  return -1;
}

DiagnosedSilenceableFailure
transform::ReuseAtOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  auto payloadValues = state.getPayloadValues(getTarget());
  auto payloadOps = state.getPayloadOps(getAxis());
  if (!llvm::hasSingleElement(payloadValues) ||
      !llvm::hasSingleElement(payloadOps)) {
    return emitSilenceableError() << "expected single operand";
  }
  Value target = *payloadValues.begin();
  auto memrefType = dyn_cast<MemRefType>(target.getType());
  if (!memrefType) {
    return emitSilenceableError() << "expected memref type";
  }
  auto reuseLoop = dyn_cast<affine::AffineForOp>(*payloadOps.begin());
  if (!reuseLoop) {
    return emitSilenceableError() << "expected an affine.for operation as axis";
  }

  unsigned int rank = memrefType.getRank();

  // Find root loop (stage)
  affine::AffineForOp rootForOp = reuseLoop;
  while (auto parent = rootForOp->getParentOfType<affine::AffineForOp>()) {
    rootForOp = parent;
  }

  // Find (non-)reduction loops
  SmallVector<affine::AffineForOp> nonReductionLoops;
  DenseMap<Value, int64_t> reductionVars;
  WalkResult result = rootForOp.walk([&](affine::AffineForOp forOp) {
    if (forOp.getStepAsInt() != 1 || !forOp.hasConstantBounds() ||
        forOp.getConstantLowerBound() != 0) {
      return WalkResult::interrupt();
    }
    if (!forOp->hasAttr("reduction") && !forOp->hasAttr("spatial") &&
        !forOp->hasAttr("buffer")) {
      nonReductionLoops.push_back(forOp);
    } else if (forOp->hasAttr("reduction")) {
      reductionVars[forOp.getInductionVar()] = forOp.getConstantUpperBound();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return emitSilenceableError()
           << "loops must have constant bounds (lb=0) and step 1";
  }
  std::reverse(nonReductionLoops.begin(), nonReductionLoops.end());

  // Find the requested loop axis
  int loopAxis = -1;
  for (size_t i = 0; i < nonReductionLoops.size(); ++i) {
    if (nonReductionLoops[i] == reuseLoop) {
      loopAxis = i;
      break;
    }
  }
  if (loopAxis == -1) {
    return emitSilenceableError()
           << "the axis loop is not found in non-reduction loops";
  }

  // 5) Get span of each dimension (also stride)
  SmallVector<SmallVector<AffineExpr>> originalLoadExprs(rank);
  int cntLoad = 0;
  DenseMap<AffineExpr, Value> dim2iv;

  reuseLoop.walk([&](affine::AffineLoadOp loadOp) {
    if (loadOp.getOperand(0) != target)
      return WalkResult::advance();
    cntLoad++;
    for (int i = 0; i < (int)rank; ++i) {
      originalLoadExprs[i].push_back(loadOp.getAffineMap().getResult(i));
    }
    OpBuilder builder(loadOp);
    for (auto operandItem : llvm::enumerate(loadOp.getMapOperands())) {
      dim2iv[builder.getAffineDimExpr(operandItem.index())] =
          operandItem.value();
    }
    return WalkResult::advance();
  });

  SmallVector<int64_t> spans;
  int64_t stride = 1;
  for (int i = 0; i < rank; ++i) {
    int64_t span = 0;
    // TODO: require strict load order
    if (originalLoadExprs[i].empty()) {
      spans.push_back(1);
      continue;
    }
    AffineExpr baseExpr = originalLoadExprs[i][0];
    int64_t baseCst = 0;
    if (llvm::isa<AffineDimExpr>(baseExpr)) {
      bool allAffineDimExpr = true;
      for (int j = 0; j < cntLoad; ++j) {
        auto diff = originalLoadExprs[i][j] - baseExpr;
        if (!llvm::isa<AffineDimExpr>(originalLoadExprs[i][j]))
          allAffineDimExpr = false;
        if (llvm::isa<AffineConstantExpr>(diff)) {
          span = std::max(
              span, llvm::dyn_cast<AffineConstantExpr>(diff).getValue() + 1);
        }
      }
      if (allAffineDimExpr &&
          reductionVars.count(dim2iv[llvm::dyn_cast<AffineDimExpr>(baseExpr)]) >
              0) {
        span = reductionVars[dim2iv[llvm::dyn_cast<AffineDimExpr>(baseExpr)]];
      }
    } else if (llvm::isa<AffineConstantExpr>(baseExpr)) {
      for (int j = 0; j < cntLoad; ++j) {
        auto diff = originalLoadExprs[i][j] - baseExpr;
        if (llvm::isa<AffineConstantExpr>(diff)) {
          span = std::max(
              span, llvm::dyn_cast<AffineConstantExpr>(diff).getValue() + 1);
        }
      }
    } else { // AffineBinaryOpExpr, reduction
      auto binaryExpr = llvm::dyn_cast<AffineBinaryOpExpr>(baseExpr);
      auto lhs = binaryExpr.getLHS();
      auto rhs = binaryExpr.getRHS();
      // d0 * s + d1, d1 is the reduction variable
      if (llvm::isa<AffineDimExpr>(rhs)) {
        auto dimExpr = llvm::dyn_cast<AffineDimExpr>(rhs);
        if (reductionVars.count(dim2iv[dimExpr]) > 0) {
          span = reductionVars[dim2iv[dimExpr]];
        }
      } else if (llvm::isa<AffineConstantExpr>(rhs)) {
        int64_t cst = llvm::dyn_cast<AffineConstantExpr>(rhs).getValue();
        if (baseCst == 0)
          baseCst = cst;
        span = std::max(span, cst - baseCst + 1);
      }
      if (llvm::isa<AffineBinaryOpExpr>(lhs)) {
        auto binLHS = llvm::dyn_cast<AffineBinaryOpExpr>(lhs);
        if (llvm::isa<AffineConstantExpr>(binLHS.getRHS()))
          stride =
              llvm::dyn_cast<AffineConstantExpr>(binLHS.getRHS()).getValue();
      }
    }
    if (span == 0)
      span = 1;
    spans.push_back(span);
  }

  // 6) Obtain AffineMaps of load instructions
  std::set<AffineExpr, ExprCompare> requestedVars;
  SmallVector<affine::AffineLoadOp> allLoadOps;
  std::map<int, int> dimBounds; // dim expr->reduction bound
  int axis = -1;
  int distance = -1;

  reuseLoop.walk([&](affine::AffineLoadOp loadOp) {
    if (loadOp.getOperand(0) != target)
      return WalkResult::advance();
    auto loadMap = loadOp.getAffineMap();
    unsigned numDims = loadMap.getNumDims();
    auto operands = loadOp.getMapOperands();
    int rDim = -1;
    int operandIdx = 0;
    for (int j = 0; j < loadMap.getNumResults(); ++j) {
      AffineExpr expr = loadMap.getResult(j);
      if (axis == -1) {
        if (llvm::isa<AffineDimExpr>(expr)) {
          if (operands[operandIdx++] ==
              nonReductionLoops[loopAxis].getInductionVar()) {
            axis = j;
          }
        } else if (llvm::isa<AffineBinaryOpExpr>(expr)) {
          auto targetIV = nonReductionLoops[loopAxis].getInductionVar();
          if (operands[operandIdx] == targetIV) {
            axis = j;
          } else if (!llvm::isa<BlockArgument>(operands[operandIdx]) &&
                     llvm::isa<affine::AffineApplyOp>(
                         operands[operandIdx].getDefiningOp())) {
            auto applyOp = llvm::dyn_cast<affine::AffineApplyOp>(
                operands[operandIdx].getDefiningOp());
            for (auto applyOpOperand : applyOp.getOperands()) {
              if (applyOpOperand == targetIV) {
                axis = j;
                break;
              }
            }
          }

          operandIdx++;
          int cntDim = 0;
          for (int i = 0; i < numDims; ++i)
            if (expr.isFunctionOfDim(i))
              cntDim++;
          if (cntDim > 1)
            if (operands[operandIdx++] ==
                nonReductionLoops[loopAxis].getInductionVar())
              axis = j;
        }
      }
      for (int i = 0; i < numDims; ++i) {
        if (expr.isFunctionOfDim(i) && reductionVars.count(operands[i]) > 0) {
          dimBounds[i] = reductionVars[operands[i]];
          if (j == axis) // target reuse axis
            rDim = i;
        }
      }
    }
    OpBuilder builder(loadOp);
    AffineExpr expr = loadMap.getResult(axis);
    auto insertLoadOp = [&](affine::AffineLoadOp loadOp) {
      unsigned size = allLoadOps.size();
      auto exp1 = loadOp.getAffineMap().getResult(axis);
      for (int i = 0; i < size; ++i) {
        int64_t val1 = ExprCompare::findConstantExpr(exp1);
        auto exp2 = allLoadOps[i].getAffineMap().getResult(axis);
        int64_t val2 = ExprCompare::findConstantExpr(exp2);
        if (val1 < val2) {
          allLoadOps.insert(allLoadOps.begin() + i, loadOp);
          return;
        }
      }
      allLoadOps.push_back(loadOp);
    };
    insertLoadOp(loadOp);
    if (rDim != -1) {
      int ub = reductionVars[operands[rDim]];
      distance = ub - 1;
      for (int j = 0; j < ub; j++) {
        auto ubCstExpr = builder.getAffineConstantExpr(j);
        auto newExpr = expr.replace(builder.getAffineDimExpr(rDim), ubCstExpr);
        requestedVars.insert(newExpr);
      }
    } else {
      requestedVars.insert(expr);
      auto var = expr - *(requestedVars.begin());
      distance = std::max(
          distance, (int)(llvm::dyn_cast<AffineConstantExpr>(var).getValue()));
    }
    return WalkResult::advance();
  });
  if (axis == -1) {
    return emitSilenceableError() << "Cannot find reuse axis";
  }

  // 7) Try to find reuse pattern
  bool canReuse = false;
  for (auto var : requestedVars) {
    if (requestedVars.find(var + 1) != requestedVars.end()) {
      canReuse = true;
      break;
    }
  }
  if (!canReuse) {
    return emitSilenceableError()
           << "Cannot find reuse pattern on axis " << std::to_string(loopAxis)
           << ". Only support stride 1 reuse pattern now";
  }

  // 8) Obtain indices and strides in load instructions
  SmallVector<AffineMap> allLoadAffineMaps;
  SmallVector<SmallVector<Value>> allLoadOperands;
  SmallVector<int> preRDim;
  SmallVector<int> preRDimAxis;
  int rDim = -1;
  auto baseVar = *(requestedVars.begin());

  for (auto loadOp : allLoadOps) {
    auto loadMap = loadOp.getAffineMap();
    auto var = loadMap.getResult(axis);
    auto diff = var - baseVar;

    auto getReductionDim = [&](AffineExpr expr) {
      for (auto item : dimBounds)
        if (expr.isFunctionOfDim(item.first))
          return item.first;
      return -1;
    };
    rDim = getReductionDim(diff);

    OpBuilder builder(loadOp);
    if (rDim != -1) { // is reduction
      return emitSilenceableError() << "Reduction reuse not fully implemented";
    } else {
      int loadRank = 0;
      int operandIdx = 0;
      auto operands = loadOp.getMapOperands();
      SmallVector<Value> memAffineIndices;
      SmallVector<AffineExpr> singleLoadAffineExpr;
      for (int i = 0; i < axis; ++i) {
        if (spans[i] > 1) {
          singleLoadAffineExpr.push_back(builder.getAffineDimExpr(loadRank++));
          memAffineIndices.push_back(operands[operandIdx]);
        }
      }
      if (llvm::isa<AffineConstantExpr>(diff)) {
        singleLoadAffineExpr.push_back(diff);
      } else {
        return emitSilenceableError() << "Cannot support non-constant stride";
      }
      for (unsigned int i = axis + 1; i < rank; ++i) {
        singleLoadAffineExpr.push_back(builder.getAffineDimExpr(loadRank++));
        memAffineIndices.push_back(operands[operandIdx++]);
      }
      auto affineMap = AffineMap::get(loadRank, 0, singleLoadAffineExpr,
                                      builder.getContext());
      if (std::find(allLoadAffineMaps.begin(), allLoadAffineMaps.end(),
                    affineMap) == allLoadAffineMaps.end()) {
        allLoadAffineMaps.push_back(affineMap);
        allLoadOperands.push_back(memAffineIndices);
      }
    }
  }

  // 9) Create reuse buffer
  SmallVector<int64_t> shape;
  for (int i = 0; i < axis; ++i)
    if (spans[i] > 1)
      shape.push_back(spans[i]);
  shape.push_back(distance + 1);
  for (unsigned int i = axis + 1; i < rank; ++i)
    shape.push_back(memrefType.getShape()[i]);

  rewriter.setInsertionPoint(rootForOp);
  auto buf = memref::AllocOp::create(
      rewriter, rootForOp.getLoc(),
      MemRefType::get(shape, memrefType.getElementType()));
  buf->setAttr("name", StringAttr::get(buf->getContext(),
                                       "reuse_" + std::to_string(loopAxis)));
  if (auto defOp = target.getDefiningOp()) {
    if (defOp->hasAttr("unsigned"))
      buf->setAttr("unsigned", rewriter.getUnitAttr());
  }

  // 11) Update loop bound
  auto origLoopBound = nonReductionLoops[loopAxis].getConstantUpperBound();
  nonReductionLoops[loopAxis].setConstantUpperBound(origLoopBound * stride +
                                                    distance);

  auto iv = nonReductionLoops[loopAxis].getInductionVar();
  for (auto &use : llvm::make_early_inc_range(iv.getUses())) {
    auto user = use.getOwner();
    if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(user)) {
      for (auto &cast_user :
           llvm::make_early_inc_range(indexCastOp.getResult().getUses())) {
        auto muliOp = dyn_cast<arith::MulIOp>(cast_user.getOwner());
        if (muliOp) {
          rewriter.setInsertionPoint(muliOp);
          Type dtype = muliOp.getResult().getType();
          auto cst =
              arith::ConstantOp::create(rewriter, muliOp.getLoc(), dtype,
                                        rewriter.getIntegerAttr(dtype, stride));
          auto subiOp = arith::SubIOp::create(rewriter, muliOp.getLoc(),
                                              indexCastOp.getResult(), cst);
          muliOp.replaceAllUsesWith(subiOp.getResult());
          muliOp.erase();
        }
      }
    }
  }

  // 12) Update store index
  SmallVector<Operation *> opToRemove;
  reuseLoop.walk([&](affine::AffineStoreOp op) {
    auto arrayType = llvm::dyn_cast<MemRefType>(op.getOperand(1).getType());
    if (arrayType.getRank() == 1 && arrayType.getShape()[0] == 1) {
      return WalkResult::advance();
    }
    rewriter.setInsertionPoint(op);
    SmallVector<AffineExpr> memAffineIndices;
    auto oldAffineMap = op.getAffineMap();
    for (unsigned int i = 0, e = oldAffineMap.getResults().size(); i < e; ++i) {
      AffineExpr idx;
      Value targetIV = nonReductionLoops[loopAxis].getInductionVar();
      int targetAxis = findMemRefAxisFromIV(op, targetIV);
      if ((int)i == targetAxis) {
        if (stride != 1) {
          auto strideCst = rewriter.getAffineConstantExpr(stride);
          idx = (oldAffineMap.getResult(i) - distance).floorDiv(strideCst);
        } else {
          idx = oldAffineMap.getResult(i) - distance;
        }
      } else
        idx = oldAffineMap.getResult(i);
      memAffineIndices.push_back(idx);
    }
    auto affineMap = AffineMap::get(arrayType.getRank(), 0, memAffineIndices,
                                    rewriter.getContext());
    affine::AffineStoreOp::create(rewriter, op->getLoc(), op.getOperand(0),
                                  op.getOperand(1), affineMap, op.getIndices());
    opToRemove.push_back(op);
    return WalkResult::advance();
  });
  for (auto op : opToRemove)
    op->erase();

  // Update if structure
  nonReductionLoops[loopAxis].walk([&](affine::AffineIfOp ifOp) {
    int operandIdx = -1;
    for (auto item : llvm::enumerate(ifOp.getOperands())) {
      if (item.value() == nonReductionLoops[loopAxis].getInductionVar()) {
        operandIdx = item.index();
        break;
      }
    }
    if (operandIdx == -1)
      return WalkResult::advance();

    auto condSet = ifOp.getIntegerSet();
    OpBuilder builder(ifOp);
    auto distanceCst = builder.getAffineConstantExpr(distance);
    SmallVector<AffineExpr> newConds;
    for (auto cond : condSet.getConstraints()) {
      bool sign = false;
      cond.walk([&](AffineExpr expr) {
        if (llvm::isa<AffineBinaryOpExpr>(expr) &&
            expr.getKind() == AffineExprKind::Mul) {
          auto binExpr = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
          if (llvm::isa<AffineConstantExpr>(binExpr.getRHS()) &&
              llvm::dyn_cast<AffineConstantExpr>(binExpr.getRHS()).getValue() ==
                  -1) {
            sign = true;
          }
        }
      });
      if (cond.isFunctionOfDim(operandIdx)) {
        if (!sign)
          newConds.push_back(cond - distanceCst -
                             builder.getAffineConstantExpr(stride - 1) *
                                 builder.getAffineDimExpr(operandIdx));
        else
          newConds.push_back(cond + distanceCst +
                             builder.getAffineConstantExpr(stride - 1) *
                                 builder.getAffineDimExpr(operandIdx));
      } else {
        newConds.push_back(cond);
      }
    }
    auto newCondSet = IntegerSet::get(condSet.getNumDims(), 0, newConds,
                                      condSet.getEqFlags());
    ifOp.setIntegerSet(newCondSet);
    return WalkResult::advance();
  });

  // 13) Rewrite original memref to load from buffer
  for (auto op : allLoadOps) {
    rewriter.setInsertionPoint(op);
    SmallVector<AffineExpr> loadAffineExpr;
    SmallVector<Value> memAffineIndices;
    SmallVector<Value> operands = op.getMapOperands();
    auto loadMap = op.getAffineMap();

    if (rDim == -1) {
      auto diff = loadMap.getResult(axis) - baseVar;
      loadAffineExpr.push_back(diff);
      int loadRank = 0;
      int operandIdx = 0;
      for (int i = 0; i < axis; ++i) {
        if (spans[i] > 1) {
          loadAffineExpr.push_back(loadMap.getResult(i));
        }
      }
      SmallVector<AffineExpr> dims;
      for (int i = 0; i < axis + 1; ++i) {
        auto expr = loadMap.getResult(i);
        if (!llvm::isa<AffineConstantExpr>(expr)) {
          operandIdx++;
          dims.push_back(rewriter.getAffineDimExpr(0)); // placeholder
        }
      }
      for (unsigned int i = axis + 1; i < rank; ++i) {
        dims.push_back(rewriter.getAffineDimExpr(loadRank++));
      }
      for (unsigned int i = axis + 1; i < rank; ++i) {
        auto expr = loadMap.getResult(i);
        auto new_expr = expr.replaceDims(dims);
        loadAffineExpr.push_back(new_expr);
        memAffineIndices.push_back(operands[operandIdx++]);
      }
      auto affineMap =
          AffineMap::get(loadRank, 0, loadAffineExpr, rewriter.getContext());
      rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(op, buf, affineMap,
                                                        memAffineIndices);
    }
  }

  results.setValues(cast<OpResult>(getResult()), {buf.getResult()});
  return DiagnosedSilenceableFailure::success();
}

void transform::ReuseAtOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTargetMutable(), effects);
  transform::consumesHandle(getAxisMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

/// --------------------------------------------------------------
/// Reshape Op
/// --------------------------------------------------------------

static AffineMap computeReshapeMap(OpBuilder &builder,
                                   ArrayRef<int64_t> oldShape,
                                   ArrayRef<int64_t> newShape,
                                   AffineMap oldMap) {
  unsigned oldRank = oldShape.size();
  unsigned newRank = newShape.size();
  MLIRContext *ctx = builder.getContext();

  // 1. compute old shape strides
  SmallVector<int64_t> oldStrides(oldRank);
  int64_t stride = 1;
  for (int64_t i = oldRank - 1; i >= 0; --i) {
    oldStrides[i] = stride;
    stride *= oldShape[i];
  }

  // 2. map old multi-dimensional indices to linear address
  auto linearAddr = builder.getAffineConstantExpr(0);
  for (unsigned i = 0; i < oldRank; ++i) {
    linearAddr = linearAddr + oldMap.getResult(i) * oldStrides[i];
  }

  // 3. map linear address to new multi-dimensional indices
  SmallVector<AffineExpr> newExprs;
  auto tempAddr = linearAddr;
  for (unsigned i = newRank - 1; i > 0; --i) {
    newExprs.push_back(tempAddr % newShape[i]);
    tempAddr = tempAddr.floorDiv(newShape[i]);
  }
  newExprs.push_back(tempAddr);
  std::reverse(newExprs.begin(), newExprs.end());

  return AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), newExprs,
                        ctx);
}

DiagnosedSilenceableFailure
transform::ReshapeOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  auto payloadValues = state.getPayloadValues(getTensor());
  SmallVector<Value> transformed;

  for (Value array : payloadValues) {
    auto oldType = dyn_cast<MemRefType>(array.getType());
    if (!oldType)
      return emitSilenceableError() << "expected memref type";
    auto newShape = llvm::to_vector(getNewshape());
    auto newType =
        MemRefType::get(newShape, oldType.getElementType(), oldType.getLayout(),
                        oldType.getMemorySpace());
    rewriter.modifyOpInPlace(array.getDefiningOp(),
                             [&]() { array.setType(newType); });

    for (auto user : llvm::make_early_inc_range(array.getUsers())) {
      rewriter.setInsertionPoint(user);
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(user)) {
        auto newMap = computeReshapeMap(rewriter, oldType.getShape(), newShape,
                                        loadOp.getAffineMap());
        auto newLoad = affine::AffineLoadOp::create(rewriter, loadOp.getLoc(),
                                                    loadOp.getMemRef(), newMap,
                                                    loadOp.getMapOperands());
        rewriter.replaceOp(loadOp, newLoad.getResult());
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(user)) {
        auto newMap = computeReshapeMap(rewriter, oldType.getShape(), newShape,
                                        storeOp.getAffineMap());
        affine::AffineStoreOp::create(
            rewriter, storeOp.getLoc(), storeOp.getValueToStore(),
            storeOp.getMemRef(), newMap, storeOp.getMapOperands());
        rewriter.eraseOp(storeOp);
      } else {
        return emitSilenceableError()
               << "only affine.load and affine.store users are supported";
      }
    }

    if (auto arg = dyn_cast<BlockArgument>(array)) {
      if (auto func = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp())) {
        auto inputTypes = llvm::to_vector(func.getArgumentTypes());
        inputTypes[arg.getArgNumber()] = newType;
        auto newFuncType = FunctionType::get(rewriter.getContext(), inputTypes,
                                             func.getResultTypes());
        func.setFunctionType(newFuncType);
      }
    }
    transformed.push_back(array);
  }
  results.setValues(cast<OpResult>(getReshaped()), transformed);
  return DiagnosedSilenceableFailure::success();
}

void transform::ReshapeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTensorMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}
