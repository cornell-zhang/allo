/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Dialect/AlloOps.h"
#include "allo/TransformOps/AlloTransformOps.h"
#include "allo/TransformOps/Utils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::allo;

///===----------------------------------------------------------------------===//
/// OutlineOp implementation
///===----------------------------------------------------------------------===//
/// Wraps the given operation `op` into an `scf.execute_region` operation. Uses
/// the provided rewriter for all operations to remain compatible with the
/// rewriting infra, as opposed to just splicing the op in place.
/// Supports operations with either zero or one region.
static scf::ExecuteRegionOp wrapInExecuteRegion(RewriterBase &b,
                                                Operation *op) {
  if (op->getNumRegions() > 1)
    return nullptr;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  scf::ExecuteRegionOp executeRegionOp =
      scf::ExecuteRegionOp::create(b, op->getLoc(), op->getResultTypes());
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&executeRegionOp.getRegion().emplaceBlock());
    Operation *clonedOp = nullptr;
    if (op->getNumRegions() == 0) {
      clonedOp = b.clone(*op);
    } else {
      clonedOp = b.cloneWithoutRegions(*op);
      Region &clonedRegion = clonedOp->getRegions().front();
      assert(clonedRegion.empty() && "expected empty region");
      b.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                           clonedRegion.end());
    }
    scf::YieldOp::create(b, op->getLoc(), clonedOp->getResults());
  }
  b.replaceOp(op, executeRegionOp.getResults());
  return executeRegionOp;
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(RewriterBase &rewriter, Operation *op,
                                Region &region) {
  assert(region.hasOneBlock() && "expected single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, /*argValues=*/{});
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Modified from
/// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/SCF/TransformOps/SCFTransformOps.cpp
/// to support outlining of arbitrary operations with at most one region
DiagnosedSilenceableFailure
transform::OutlineOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  SmallVector<Operation *, 4> kernels;
  SmallVector<Operation *, 4> calls;
  DenseMap<Operation *, SymbolTable> symbolTables;

  for (Operation *target : state.getPayloadOps(getTarget())) {
    Location loc = target->getLoc();
    if (target->getNumRegions() > 1) {
      return emitSilenceableFailure(target)
             << "expected target operation to have at most one region";
    }
    Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(target);
    auto exec = wrapInExecuteRegion(rewriter, target);
    if (!exec) {
      return emitSilenceableFailure(target)
             << "expected target operation to have at most one region";
    }
    func::CallOp call;
    auto outlined = outlineSingleBlockRegion(rewriter, loc, exec.getRegion(),
                                             getKernelName(), &call);
    if (failed(outlined)) {
      return emitSilenceableFailure(target)
             << "failed to outline the target operation";
    }

    if (symbolTableOp) {
      SymbolTable &symbolTable =
          symbolTables.try_emplace(symbolTableOp, symbolTableOp)
              .first->getSecond();
      symbolTable.insert(*outlined);
      call.setCalleeAttr(FlatSymbolRefAttr::get(*outlined));
    }
    call->setAttr(OpIdentifier,
                  rewriter.getStringAttr(getKernelName() + ".call"));
    // `scf.execute_region` is only an outlining helper container. Inline it
    // back so the final IR directly contains `allo.call`.
    replaceOpWithRegion(rewriter, exec, exec.getRegion());
    kernels.push_back(*outlined);
    calls.push_back(call);
  }
  results.set(cast<OpResult>(getKernel()), kernels);
  results.set(cast<OpResult>(getCall()), calls);
  return DiagnosedSilenceableFailure::success();
}

///==--------------------------------------------------------------------===//
/// ReorderOp implementation
///===-------------------------------------------------------------------===//

/// Checks if the given loops are in the same perfectly nested loop band.
/// Return the outermost loop if true. Otherwise returns null.
/// The input loops do not need to be contiguous, or sorted by depth
static affine::AffineForOp
inSamePerfectlyNestedLoopBand(ArrayRef<affine::AffineForOp> loops) {
  if (loops.empty())
    return {};
  if (loops.size() == 1)
    return {};
  // create a temp copy and sort by depth
  auto tmp = llvm::to_vector(loops);
  DenseMap<affine::AffineForOp, unsigned> depthMap;
  llvm::for_each(tmp, [&depthMap](auto op) {
    unsigned depth = 0;
    Operation *curr = op;
    while ((curr = curr->getParentOp()))
      depth++;
    depthMap[op] = depth;
  });
  llvm::sort(tmp,
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
    Operation *ptr = currLoop;
    while (ptr != nextLoop) {
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

DiagnosedSilenceableFailure
transform::LoopReorderOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {

  SmallVector<affine::AffineForOp> loops;
  // validate input operation handles
  for (Operation *payload : state.getPayloadOps(getLoops())) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(payload)) {
      loops.push_back(forOp);
    } else {
      std::string msg = "expected an affine.for operation.";
      if (isa<scf::ForOp>(payload)) {
        msg += " Try raise scf.for to affine.for before reordering.";
      }
      return emitSilenceableFailure(payload) << msg;
    }
  }
  if (loops.size() < 2) {
    return emitSilenceableError()
           << "at least two loops are required for reordering";
  }

  auto outermostLoop = inSamePerfectlyNestedLoopBand(loops);
  if (!outermostLoop) {
    return emitSilenceableError()
           << "loops must be in the same perfectly nested loop band";
  }
  SmallVector<affine::AffineForOp, 4> band;
  affine::getPerfectlyNestedLoops(band, outermostLoop);

  // Validate permutation size against the number of selected loops.
  if (getPermutation().size() != loops.size()) {
    return emitSilenceableError()
           << "the size of permutation must match the number of loops";
  }

  // Construct the permutation vector over selected loops.
  SmallVector<unsigned, 4> permutation;
  llvm::for_each(getPermutation(), [&](Attribute attr) {
    permutation.push_back(cast<IntegerAttr>(attr).getInt());
  });

  // Map selected loops to their original positions in the full perfect band.
  SmallVector<unsigned, 4> selectedOrgIndices;
  for (auto l : loops) {
    auto *it = llvm::find(band, l);
    if (it == band.end())
      return emitSilenceableError() << "selected loop is not in the loop band";
    unsigned idx = std::distance(band.begin(), it);
    selectedOrgIndices.push_back(idx);
  }

  // Build a full-band permutation map. Unselected loops keep identity mapping.
  SmallVector<unsigned, 4> permMap(band.size());
  std::iota(permMap.begin(), permMap.end(), 0u);

  for (unsigned i = 0; i < permutation.size(); ++i) {
    unsigned targetPos = selectedOrgIndices[i];
    unsigned srcPos = selectedOrgIndices[permutation[i]];
    permMap[targetPos] = srcPos;
  }

  // perform reordering
  if (!affine::isValidLoopInterchangePermutation(band, permMap)) {
    return emitSilenceableError() << "permutation violates legality "
                                     "constraints (e.g., data dependencies)";
  }
  affine::permuteLoops(band, permMap);
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::LoopReorderOp::verify() {
  // we cannot know the number of loops at verification time
  // so we only check the validity of the permutation itself
  unsigned nPerm = getPermutation().size();
  SmallVector<int64_t> permutation;
  llvm::for_each(getPermutation(), [&](Attribute attr) {
    permutation.push_back(cast<IntegerAttr>(attr).getInt());
  });
  for (unsigned i = 0; i < nPerm; ++i) {
    if (permutation[i] < 0 || permutation[i] >= static_cast<int64_t>(nPerm)) {
      return emitOpError("permutation index out of bounds: ") << permutation[i];
    }
    for (unsigned j = i + 1; j < nPerm; ++j) {
      if (permutation[i] == permutation[j]) {
        return emitOpError("permutation contains duplicate index: ")
               << permutation[i];
      }
    }
  }
  return success();
}

///===----------------------------------------------------------------------===//
/// SplitOp implementation
///===----------------------------------------------------------------------===//

/// Checks if the given split factor is valid for the given loop.
/// A valid split factor should be positive and smaller than the loop range.
/// only checks constant bounds loops
static bool checkSplitFactor(affine::AffineForOp loop, int64_t factor) {
  if (!loop.hasConstantBounds()) {
    return true;
  }
  int64_t lb = loop.getConstantLowerBound();
  int64_t ub = loop.getConstantUpperBound();
  int64_t range = ub - lb;
  return factor > 0 && factor < range;
}

static FailureOr<int64_t> stripCastInt(Value value) {
  Value current = value;
  while (true) {
    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      return failure();
    }
    if (isa<arith::IndexCastOp, arith::TruncIOp, arith::ExtUIOp,
            arith::ExtSIOp>(defOp)) {
      current = defOp->getOperand(0);
      continue;
    }
    IntegerAttr::ValueType cst;
    if (matchPattern(current, m_ConstantInt(&cst))) {
      return cst.getSExtValue();
    }
    return failure();
  }
}

static bool checkSplitFactor(scf::ForOp loop, int64_t factor) {
  auto lbOr = stripCastInt(loop.getLowerBound());
  auto ubOr = stripCastInt(loop.getUpperBound());
  if (failed(lbOr) || failed(ubOr)) {
    return true;
  }
  int64_t lb = *lbOr;
  int64_t ub = *ubOr;
  int64_t range = ub - lb;
  return factor > 0 && factor < range;
}

DiagnosedSilenceableFailure
transform::LoopSplitOp::applyToOne(transform::TransformRewriter &rewriter,
                                   Operation *target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  int64_t factor = getFactorAttr().getInt();
  if (factor <= 0) {
    return emitSilenceableFailure(getOperation())
           << "split factor must be positive";
  }
  // Case 1: affine.for loop
  if (auto forOp = dyn_cast<affine::AffineForOp>(target)) {
    auto symName = forOp->getAttrOfType<StringAttr>(OpIdentifier);
    if (!checkSplitFactor(forOp, factor)) {
      return emitSilenceableFailure(forOp)
             << "split factor is larger than or equal to the loop range";
    }

    // perform split
    // single loop is always perfectly nested
    SmallVector<affine::AffineForOp, 2> splitOps;
    if (failed(affine::tilePerfectlyNested(forOp, factor, &splitOps))) {
      return emitSilenceableFailure(forOp) << "failed to split the loop";
    }
    assert(splitOps.size() == 2 && "expected exactly two loops after tiling");

    // normalize loop
    auto outer = splitOps.front();
    auto inner = splitOps.back();
    if (failed(affine::normalizeAffineFor(outer)) ||
        failed(affine::normalizeAffineFor(inner))) {
      return emitSilenceableFailure(forOp) << "failed to normalize the loop";
    }

    AffineMap innerUb = inner.getUpperBoundMap();
    if (innerUb.isConstant() && innerUb.getNumInputs() != 0) {
      // simplify the upper bound if it's a constant map with unused symbols
      auto cstUb =
          dyn_cast<AffineConstantExpr>(innerUb.getResult(0)).getValue();
      rewriter.setInsertionPoint(inner);
      inner.setUpperBound({}, rewriter.getConstantAffineMap(cstUb));
    }

    // Sink affine.apply ops that are only used in the inner loop.
    for (auto applyOp :
         llvm::make_early_inc_range(outer.getOps<affine::AffineApplyOp>())) {
      bool allUsesInInner = llvm::all_of(applyOp->getUses(), [&](OpOperand &u) {
        return inner->isProperAncestor(u.getOwner());
      });
      if (allUsesInInner) {
        applyOp->moveBefore(&inner.getBody()->front());
      }
    }
    // set sym_name
    if (symName) {
      auto symStr = symName.getValue();
      inner->setAttr(OpIdentifier, rewriter.getStringAttr(symStr + ".inner"));
      outer->setAttr(OpIdentifier, rewriter.getStringAttr(symStr + ".outer"));
    }
    // record results
    results.push_back(outer);
    results.push_back(inner);
    return DiagnosedSilenceableFailure::success();
  }
  // Case 2: scf.for loop
  if (auto forOp = dyn_cast<scf::ForOp>(target)) {
    auto symName = forOp->getAttrOfType<StringAttr>(OpIdentifier);
    if (!checkSplitFactor(forOp, factor)) {
      return emitSilenceableFailure(forOp)
             << "split factor is larger than or equal to the loop range";
    }
    // perform split
    rewriter.setInsertionPoint(forOp);
    Value cst =
        arith::ConstantIndexOp::create(rewriter, forOp.getLoc(), factor);
    auto loops = tilePerfectlyNested(forOp, cst);
    if (loops.size() != 1) {
      return emitSilenceableFailure(forOp) << "failed to split the loop";
    }
    // set sym_name
    if (symName) {
      auto symStr = symName.getValue();
      forOp->setAttr(OpIdentifier, rewriter.getStringAttr(symStr + ".outer"));
      loops.back()->setAttr(OpIdentifier,
                            rewriter.getStringAttr(symStr + ".inner"));
    }
    // record results
    results.push_back(forOp);
    results.push_back(loops.front());
    return DiagnosedSilenceableFailure::success();
  }
  return emitSilenceableFailure(target)
         << "expected target operation to be an affine.for or scf.for loop";
}

///===----------------------------------------------------------------------===///
/// LoopTileOp implementation
///===----------------------------------------------------------------------===///
static unsigned getOperationDepth(Operation *op) {
  unsigned depth = 0;
  Operation *curr = op;
  while ((curr = curr->getParentOp()))
    depth++;
  return depth;
}

namespace {
template <typename ForOp> struct LoopWithFactor {
  ForOp loop;
  uint64_t factor;
};
} // namespace

/// Sort (loop, factor) pairs by loop depth and check they form a single
/// ancestor chain with unique loops.
template <typename ForOp>
static LogicalResult
sortAndCheckLoopFactorPairs(SmallVectorImpl<LoopWithFactor<ForOp>> &pairs) {
  DenseSet<Operation *> seenLoops;
  for (auto pair : pairs) {
    if (!seenLoops.insert(pair.loop).second)
      return failure();
  }

  llvm::sort(pairs, [](const LoopWithFactor<ForOp> &a,
                       const LoopWithFactor<ForOp> &b) {
    return getOperationDepth(a.loop) < getOperationDepth(b.loop);
  });

  // Check they belong to one loop nest.
  for (unsigned i = 0; i < pairs.size() - 1; ++i) {
    if (!pairs[i].loop->isProperAncestor(pairs[i + 1].loop))
      return failure();
  }
  return success();
}

template <typename ForOp>
static bool isContiguousPerfectBand(SmallVectorImpl<ForOp> &loops) {
  if (loops.size() <= 1)
    return true;
  for (unsigned i = 0; i < loops.size() - 1; ++i) {
    Block *body = loops[i].getBody();
    if (body->getOperations().size() != 2)
      return false;
    if (&body->front() != loops[i + 1].getOperation())
      return false;
  }
  return true;
}

static FailureOr<SmallVector<uint64_t, 4>> parseTileFactors(ArrayAttr attr) {
  SmallVector<uint64_t, 4> factors;
  factors.reserve(attr.size());
  for (Attribute a : attr) {
    int64_t factor = cast<IntegerAttr>(a).getInt();
    if (factor <= 0)
      return failure();
    factors.push_back(static_cast<uint64_t>(factor));
  }
  return factors;
}

template <typename ForOp>
static SmallVector<StringAttr, 4> collectLoopSymNames(ArrayRef<ForOp> loops) {
  SmallVector<StringAttr, 4> symNames;
  symNames.reserve(loops.size());
  for (ForOp loop : loops)
    symNames.push_back(loop->template getAttrOfType<StringAttr>(OpIdentifier));
  return symNames;
}

static void annotateTiledLoopSymNames(RewriterBase &rewriter,
                                      ArrayRef<StringAttr> inputSymNames,
                                      ArrayRef<Operation *> tileLoops,
                                      ArrayRef<Operation *> pointLoops) {
  assert(inputSymNames.size() == tileLoops.size() &&
         "expected one sym_name per tile loop");
  assert(inputSymNames.size() == pointLoops.size() &&
         "expected one sym_name per point loop");

  for (auto [symName, tileLoop, pointLoop] :
       llvm::zip_equal(inputSymNames, tileLoops, pointLoops)) {
    if (!symName)
      continue;
    StringRef base = symName.getValue();
    tileLoop->setAttr(OpIdentifier, rewriter.getStringAttr(base + ".tile"));
    pointLoop->setAttr(OpIdentifier, rewriter.getStringAttr(base + ".point"));
  }
}

DiagnosedSilenceableFailure
transform::LoopTileOp::apply(transform::TransformRewriter &rewriter,
                             transform::TransformResults &results,
                             transform::TransformState &state) {
  SmallVector<affine::AffineForOp, 4> affineLoops;
  SmallVector<scf::ForOp, 4> scfLoops;

  // Collect payload loops in handle iteration order.
  // This order is the semantic order for mapping input factors to loops.
  for (Operation *payload : state.getPayloadOps(getLoops())) {
    if (auto affineFor = dyn_cast<affine::AffineForOp>(payload)) {
      affineLoops.push_back(affineFor);
    } else if (auto scfFor = dyn_cast<scf::ForOp>(payload)) {
      scfLoops.push_back(scfFor);
    } else {
      return emitSilenceableFailure(payload)
             << "expected an affine.for or scf.for operation";
    }
  }

  // A single tile op must target one loop dialect only.
  if (!affineLoops.empty() && !scfLoops.empty()) {
    return emitSilenceableError()
           << "cannot mix affine.for and scf.for loops in the same tiling";
  }
  if (affineLoops.empty() && scfLoops.empty()) {
    return emitSilenceableError() << "expected at least one loop to tile";
  }

  // Parse and validate tile factors once for both affine/scf paths.
  auto maybeFactors = parseTileFactors(getFactors());
  if (failed(maybeFactors))
    return emitSilenceableError() << "tile factors must be positive";
  SmallVector<uint64_t, 4> factors = *maybeFactors;

  if (!affineLoops.empty()) {
    // Factors must be provided one-for-one with input loop handles.
    if (factors.size() != affineLoops.size()) {
      return emitSilenceableError()
             << "number of tile factors must match the number of loops";
    }

    // Semantic rule:
    // bind factors to loops by input-handle order, then sort pairs by depth.
    SmallVector<LoopWithFactor<affine::AffineForOp>, 4> loopFactors;
    loopFactors.reserve(affineLoops.size());
    for (auto [loop, factor] : llvm::zip_equal(affineLoops, factors))
      loopFactors.push_back({loop, factor});

    // Validate selected loops form a single nest and are not duplicated.
    // Sorting pairs preserves loop-factor associations after reordering.
    if (failed(sortAndCheckLoopFactorPairs(loopFactors))) {
      return emitSilenceableError()
             << "affine loops must be unique and in the same loop nest";
    }

    // Materialize depth-sorted loops/factors for downstream tiling APIs.
    SmallVector<affine::AffineForOp, 4> sortedLoops;
    SmallVector<uint64_t, 4> sortedFactors;
    sortedLoops.reserve(loopFactors.size());
    sortedFactors.reserve(loopFactors.size());
    for (const auto &it : loopFactors) {
      sortedLoops.push_back(it.loop);
      sortedFactors.push_back(it.factor);
    }
    SmallVector<StringAttr, 4> inputSymNames =
        collectLoopSymNames<affine::AffineForOp>(sortedLoops);

    SmallVector<Operation *, 4> tileLoops;
    SmallVector<Operation *, 4> pointLoops;

    // Choose perfect vs imperfect tiling based on structural contiguity.
    bool perfect = isContiguousPerfectBand<affine::AffineForOp>(sortedLoops);

    if (perfect) {
      // Perfect affine tiling creates [outer tile loops..., inner point
      // loops...].
      SmallVector<unsigned, 4> uFactors;
      uFactors.reserve(sortedFactors.size());
      for (uint64_t factor : sortedFactors) {
        if (factor > std::numeric_limits<unsigned>::max()) {
          return emitSilenceableError()
                 << "tile factor exceeds supported affine tile size range";
        }
        uFactors.push_back(static_cast<unsigned>(factor));
      }
      SmallVector<affine::AffineForOp, 8> tiledNest;
      SmallVector<affine::AffineForOp, 4> band = sortedLoops;
      if (failed(affine::tilePerfectlyNested(band, uFactors, &tiledNest)))
        return emitSilenceableFailure(sortedLoops.front())
               << "failed to tile affine perfectly nested loops";
      if (tiledNest.size() < sortedLoops.size() * 2)
        return emitSilenceableError()
               << "unexpected number of loops created by affine tiling";

      unsigned nLoops = sortedLoops.size();
      for (auto loop : tiledNest) {
        if (failed(affine::normalizeAffineFor(loop))) {
          return emitSilenceableFailure(loop)
                 << "failed to normalize tiled affine loop";
        }
      }

      // Canonicalize point-loop upper bounds
      for (auto loop : llvm::drop_begin(tiledNest, nLoops)) {
        AffineMap ubMap = loop.getUpperBoundMap();
        if (ubMap.isConstant() && ubMap.getNumInputs() != 0) {
          auto cstUb = cast<AffineConstantExpr>(ubMap.getResult(0)).getValue();
          rewriter.setInsertionPoint(loop);
          loop.setUpperBound({}, rewriter.getConstantAffineMap(cstUb));
        } else if (!ubMap.isConstant()) {
          if (ubMap.getNumResults() == 2 && ubMap.getNumInputs() == 1) {
            auto addMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                         ubMap.getResult(1));
            auto applyOp = dyn_cast_or_null<affine::AffineApplyOp>(
                loop.getUpperBoundOperands().front().getDefiningOp());
            if (!applyOp)
              continue;
            auto outerIV = applyOp.getOperand(0);
            AffineMap composed = addMap.compose(applyOp.getAffineMap());
            SmallVector<AffineExpr, 2> exprs{ubMap.getResult(0),
                                             composed.getResult(0)};
            AffineMap finalMap = AffineMap::get(
                /*dimCount=*/1, /*symbolCount=*/0, exprs,
                rewriter.getContext());
            loop.setUpperBound(outerIV, finalMap);
          }
        }
      }
      // sink affine.apply into point loops when all uses are inside.
      for (unsigned i = 0; i < nLoops; ++i) {
        auto outer = tiledNest[i];
        auto point = tiledNest[i + nLoops];
        for (auto applyOp : llvm::make_early_inc_range(
                 outer.getOps<affine::AffineApplyOp>())) {
          bool allUsesInPoint =
              llvm::all_of(applyOp->getUses(), [&](OpOperand &u) {
                return point->isProperAncestor(u.getOwner());
              });
          if (allUsesInPoint)
            applyOp->moveBefore(&point.getBody()->front());
        }
        tileLoops.push_back(outer);
        pointLoops.push_back(point);
      }
      // sink affine.apply to innermost point loops to make perfectly nested
      for (unsigned i = nLoops; i < tiledNest.size() - 1; ++i) {
        auto point = tiledNest[i];
        auto nextPoint = tiledNest[i + 1];
        for (auto applyOp : llvm::make_early_inc_range(
                 point.getOps<affine::AffineApplyOp>())) {
          bool allUsesInNextPoint =
              llvm::all_of(applyOp->getUses(), [&](OpOperand &u) {
                return nextPoint->isProperAncestor(u.getOwner());
              });
          if (allUsesInNextPoint)
            applyOp->moveBefore(&nextPoint.getBody()->front());
        }
      }
    } else {
      // Imperfect affine tiling strip-mines selected loops and sinks them
      // under the chosen target while preserving sorted loop order.
      auto point = affine::tile(sortedLoops, sortedFactors, sortedLoops.back());
      if (point.size() != sortedLoops.size()) {
        return emitSilenceableError()
               << "failed to tile affine imperfectly nested loops";
      }
      for (auto loop : sortedLoops)
        tileLoops.push_back(loop);
      for (auto loop : point)
        pointLoops.push_back(loop);
    }

    annotateTiledLoopSymNames(rewriter, inputSymNames, tileLoops, pointLoops);

    // Output handles are always reported in depth order.
    results.set(cast<OpResult>(getTileLoops()), tileLoops);
    results.set(cast<OpResult>(getPointLoops()), pointLoops);
    return DiagnosedSilenceableFailure::success();
  }

  if (!scfLoops.empty()) {
    // Factors must be provided one-for-one with input loop handles.
    if (factors.size() != scfLoops.size()) {
      return emitSilenceableError()
             << "number of tile factors must match the number of loops";
    }

    // Semantic rule:
    // bind factors to loops by input-handle order, then sort pairs by depth.
    SmallVector<LoopWithFactor<scf::ForOp>, 4> loopFactors;
    loopFactors.reserve(scfLoops.size());
    for (auto [loop, factor] : llvm::zip_equal(scfLoops, factors))
      loopFactors.push_back({loop, factor});

    // Validate selected loops form a single nest and are not duplicated.
    // Sorting pairs preserves loop-factor associations after reordering.
    if (failed(sortAndCheckLoopFactorPairs(loopFactors))) {
      return emitSilenceableError()
             << "scf loops must be unique and in the same loop nest";
    }

    // Materialize depth-sorted loops/factors for downstream tiling APIs.
    SmallVector<scf::ForOp, 4> sortedLoops;
    SmallVector<uint64_t, 4> sortedFactors;
    sortedLoops.reserve(loopFactors.size());
    sortedFactors.reserve(loopFactors.size());
    for (const auto &it : loopFactors) {
      sortedLoops.push_back(it.loop);
      sortedFactors.push_back(it.factor);
    }
    SmallVector<StringAttr, 4> inputSymNames =
        collectLoopSymNames<scf::ForOp>(sortedLoops);

    SmallVector<Operation *, 4> tileLoops;
    SmallVector<Operation *, 4> pointLoops;

    // Build runtime tile-size SSA values from sorted factors.
    SmallVector<Value, 4> sizeVals;
    sizeVals.reserve(sortedFactors.size());
    rewriter.setInsertionPoint(sortedLoops.front());
    for (uint64_t factor : sortedFactors) {
      if (factor > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return emitSilenceableError()
               << "tile factor exceeds supported scf tile size range";
      sizeVals.push_back(
          arith::ConstantIndexOp::create(rewriter, sortedLoops.front().getLoc(),
                                         static_cast<int64_t>(factor)));
    }

    // Choose perfect vs imperfect tiling based on structural contiguity.
    bool perfect = isContiguousPerfectBand<scf::ForOp>(sortedLoops);
    if (perfect) {
      // Perfect scf tiling returns only point loops; outer loops are updated
      // in-place and remain represented by sortedLoops.
      SmallVector<scf::ForOp, 8> point =
          tilePerfectlyNested(sortedLoops.front(), sizeVals);
      if (point.size() != sortedLoops.size()) {
        return emitSilenceableError()
               << "failed to tile scf perfectly nested loops";
      }
      for (auto loop : sortedLoops)
        tileLoops.push_back(loop);
      for (auto loop : point)
        pointLoops.push_back(loop);
    } else {
      // Imperfect scf tiling strip-mines selected loops and sinks them
      // under the chosen target while preserving sorted loop order.
      auto point = ::mlir::tile(sortedLoops, sizeVals, sortedLoops.back());
      if (point.size() != sortedLoops.size()) {
        return emitSilenceableError()
               << "failed to tile scf imperfectly nested loops";
      }
      for (auto loop : sortedLoops)
        tileLoops.push_back(loop);
      for (auto loop : point)
        pointLoops.push_back(loop);
    }

    annotateTiledLoopSymNames(rewriter, inputSymNames, tileLoops, pointLoops);

    // Output handles are always reported in depth order.
    results.set(cast<OpResult>(getTileLoops()), tileLoops);
    results.set(cast<OpResult>(getPointLoops()), pointLoops);
    return DiagnosedSilenceableFailure::success();
  }

  return emitSilenceableError() << "failed to tile loops";
}

///===----------------------------------------------------------------------===//
/// LoopFlattenOp implementation
///===----------------------------------------------------------------------===///

// modified from lib/Transforms/Utils/LoopUtils.cpp
static void coalesceLoops(MutableArrayRef<affine::AffineForOp> loops,
                          int64_t flattenedTripCount,
                          transform::TransformRewriter &rewriter) {
  // RAII helper to restore the insertion point.
  OpBuilder::InsertionGuard guard(rewriter);

  affine::AffineForOp innermost = loops.back();
  affine::AffineForOp outermost = loops.front();
  Location loc = outermost.getLoc();

  // 1. Store the upper bound of the outermost loop in a variable.
  SmallVector<int64_t, 4> ubs;
  for (auto loop : loops) {
    auto cstUb = loop.getConstantUpperBound();
    ubs.push_back(cstUb);
  }

  // 2. The flattened trip count is validated by the caller.
  outermost.setConstantUpperBound(flattenedTripCount);

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
  outermost.getBody()->getOperations().splice(
      Block::iterator(secondOutermostLoop.getOperation()),
      innermost.getBody()->getOperations());
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
    for (Operation *op : opToSink) {
      for (Operation *user : op->getUsers()) {
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
      for (Operation *op : opToSink) {
        op->moveBefore(&body->front());
      }
    }
  });
}

DiagnosedSilenceableFailure
transform::LoopFlattenOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {
  SmallVector<affine::AffineForOp, 4> loops;
  // validate input operation handles
  for (Operation *payload : state.getPayloadOps(getLoops())) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(payload)) {
      loops.push_back(forOp);
    } else {
      if (isa<scf::ForOp>(payload)) {
        return emitSilenceableFailure(payload)
               << "Try raise scf.for to affine.for before flattening.";
      }
      return emitSilenceableFailure(payload)
             << "expected an affine.for operation";
    }
  }
  if (loops.size() < 2) {
    results.set(cast<OpResult>(getResult()), loops);
    return DiagnosedSilenceableFailure::success();
  }

  auto namePrefix = loops.front()->getAttrOfType<StringAttr>(OpIdentifier);

  // Flatten supports unordered loop handles; normalize to depth order first.
  llvm::sort(loops, [](auto a, auto b) {
    return getOperationDepth(a) < getOperationDepth(b);
  });

  auto selectedOutermost = inSamePerfectlyNestedLoopBand(loops);
  if (!selectedOutermost) {
    return emitSilenceableError()
           << "loops must be in the same perfectly nested loop band";
  }

  // Flatten a contiguous band from selected outermost to selected innermost.
  SmallVector<affine::AffineForOp, 4> perfectBand;
  affine::getPerfectlyNestedLoops(perfectBand, selectedOutermost);
  auto *endIt = llvm::find(perfectBand, loops.back());
  if (endIt == perfectBand.end()) {
    return emitSilenceableError()
           << "failed to find selected innermost loop in perfect loop band";
  }
  SmallVector<affine::AffineForOp, 4> flattenBand(perfectBand.begin(),
                                                  std::next(endIt));

  // Current coalescing logic assumes normalized affine loops with constant
  // trip counts.
  int64_t flattenedTripCount = 1;
  for (auto loop : flattenBand) {
    if (loop.getStepAsInt() != 1 || !loop.hasConstantLowerBound() ||
        loop.getConstantLowerBound() != 0 || !loop.hasConstantUpperBound()) {
      return emitSilenceableError()
             << "flatten requires normalized affine.for loops with step=1, "
                "constant lower bound=0 and constant upper bound";
    }
    int64_t ub = loop.getConstantUpperBound();
    if (ub <= 0) {
      return emitSilenceableError()
             << "flatten requires positive constant upper bounds";
    }
    if (flattenedTripCount > std::numeric_limits<int64_t>::max() / ub) {
      return emitSilenceableError()
             << "flattened loop trip count overflows int64";
    }
    flattenedTripCount *= ub;
  }

  // perform flattening
  ::coalesceLoops(flattenBand, flattenedTripCount, rewriter);

  // set sym_name
  if (namePrefix) {
    flattenBand.front()->setAttr(
        OpIdentifier, rewriter.getStringAttr(namePrefix.getValue() + ".flat"));
  }
  // record results
  results.set(cast<OpResult>(getResult()), {flattenBand.front()});
  return DiagnosedSilenceableFailure::success();
}

///===----------------------------------------------------------------------===//
/// ComputeAt implementation
///===----------------------------------------------------------------------===///

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

namespace {
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
} // namespace

// check dependencies between two affine.for loop nests up to a certain depth
// assume forOpA is source, forOpB is sink
// return a bitmask of DependenceType
static FailureOr<DependenceType> checkDependencies(affine::AffineForOp forOpA,
                                                   affine::AffineForOp forOpB,
                                                   unsigned depth) {
  SmallVector<affine::MemRefAccess, 4> accA;
  SmallVector<affine::MemRefAccess, 4> accB;
  bool hasUnsupportedAccess = false;
  // Collect only affine accesses; non-affine memref accesses are conservatively
  // treated as unsupported so we can fail instead of mis-transforming.
  forOpA.walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
            op)) {
      accA.emplace_back(op);
    } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      hasUnsupportedAccess = true;
    }
  });
  forOpB.walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
            op)) {
      accB.emplace_back(op);
    } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      hasUnsupportedAccess = true;
    }
  });
  if (hasUnsupportedAccess)
    return failure();

  // Build a coarse dependence summary (RAW/WAR/WAW) between source and sink
  // loop nests, using affine dependence checks per matching memref access pair.
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
      auto depResult =
          affine::checkMemrefAccessDependence(a, b, depth, nullptr, &deps);
      if (depResult.value == affine::DependenceResult::Failure) {
        return failure();
      }
      if (depResult.value == affine::DependenceResult::HasDependence) {
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

namespace {
struct ConstantBoundsPrefix {
  SmallVector<int64_t, 4> lowerBounds;
  SmallVector<int64_t, 4> upperBounds;
};
} // namespace

static SmallVector<affine::AffineForOp, 4>
collectAffineLoopChain(Operation *op) {
  SmallVector<affine::AffineForOp, 4> chain;
  for (Operation *curr = op; curr; curr = curr->getParentOp()) {
    if (auto loop = dyn_cast<affine::AffineForOp>(curr))
      chain.push_back(loop);
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

static FailureOr<ConstantBoundsPrefix>
getConstantBoundsPrefix(ArrayRef<affine::AffineForOp> chain, unsigned depth) {
  if (chain.size() < depth)
    return failure();
  ConstantBoundsPrefix bounds;
  bounds.lowerBounds.reserve(depth);
  bounds.upperBounds.reserve(depth);
  for (unsigned i = 0; i < depth; ++i) {
    affine::AffineForOp loop = chain[i];
    if (!loop.hasConstantBounds())
      return failure();
    bounds.lowerBounds.push_back(loop.getConstantLowerBound());
    bounds.upperBounds.push_back(loop.getConstantUpperBound());
  }
  return bounds;
}

static bool hasSinglePathLoopPrefix(ArrayRef<affine::AffineForOp> chain,
                                    unsigned prefixDepth) {
  if (prefixDepth > chain.size())
    return false;
  for (unsigned i = 0; i + 1 < prefixDepth; ++i) {
    affine::AffineForOp current = chain[i];
    affine::AffineForOp next = chain[i + 1];
    Block *body = current.getBody();
    if (body->getOperations().size() != 2)
      return false;
    if (&body->front() != next.getOperation())
      return false;
  }
  return true;
}

static bool hasIdenticalBounds(const ConstantBoundsPrefix &a,
                               const ConstantBoundsPrefix &b) {
  return a.lowerBounds == b.lowerBounds && a.upperBounds == b.upperBounds;
}

static bool isSubsetBounds(const ConstantBoundsPrefix &producerBounds,
                           const ConstantBoundsPrefix &consumerBounds) {
  for (auto [prodLb, prodUb, consLb, consUb] : llvm::zip_equal(
           producerBounds.lowerBounds, producerBounds.upperBounds,
           consumerBounds.lowerBounds, consumerBounds.upperBounds)) {
    if (prodLb < consLb || prodUb > consUb)
      return false;
  }
  return true;
}

static IntegerSet buildSubsetGuardSet(OpBuilder &builder,
                                      const ConstantBoundsPrefix &bounds) {
  unsigned depth = bounds.lowerBounds.size();
  SmallVector<AffineExpr, 8> constraints;
  constraints.reserve(depth * 2);
  for (unsigned i = 0; i < depth; ++i) {
    AffineExpr dim = builder.getAffineDimExpr(i);
    constraints.push_back(dim - bounds.lowerBounds[i]);
    constraints.push_back((bounds.upperBounds[i] - 1) - dim);
  }
  SmallVector<bool, 8> isEq(constraints.size(), false);
  return IntegerSet::get(depth, /*symbolCount=*/0, constraints, isEq);
}

static void remapProducerIVPrefix(ArrayRef<affine::AffineForOp> producerChain,
                                  ArrayRef<affine::AffineForOp> consumerChain,
                                  unsigned depth, Region &region) {
  for (unsigned i = 0; i < depth; ++i) {
    affine::AffineForOp producer = producerChain[i];
    affine::AffineForOp consumer = consumerChain[i];
    replaceAllUsesInRegionWith(producer.getInductionVar(),
                               consumer.getInductionVar(), region);
  }
}

static bool hasBlockingSideEffectsBetween(Operation *before, Operation *after) {
  assert(before->getBlock() == after->getBlock() &&
         "expected operations in the same block");
  assert(before->isBeforeInBlock(after) &&
         "expected `before` to appear before `after`");

  for (Operation *curr = before->getNextNode(); curr && curr != after;
       curr = curr->getNextNode()) {
    if (!isMemoryEffectFree(curr))
      return true;
  }
  return false;
}

namespace {
struct ComputeAtAnalysis {
  Operation *producerOp = nullptr;
  affine::AffineForOp consumerLoop = nullptr;
  SmallVector<affine::AffineForOp, 4> producerChain;
  SmallVector<affine::AffineForOp, 4> consumerChain;
  affine::AffineForOp producerRoot = nullptr;
  affine::AffineForOp consumerRoot = nullptr;
  unsigned producerDepth = 0;
  unsigned consumerDepth = 0;
  SmallVector<Value, 4> consumerPrefixIVs;
};
} // namespace

static std::optional<std::string>
analyzeComputeAt(Operation *producerOp, affine::AffineForOp consumerLoop,
                 ComputeAtAnalysis &analysis) {
  // Normalize producer/consumer into loop-chain metadata once so execution
  // paths can focus on transformation mechanics.
  analysis.producerOp = producerOp;
  analysis.consumerLoop = consumerLoop;
  analysis.producerChain = collectAffineLoopChain(producerOp);
  if (analysis.producerChain.empty()) {
    return std::string("producer must be inside an affine.for loop nest");
  }

  analysis.consumerChain = collectAffineLoopChain(consumerLoop);
  if (analysis.consumerChain.empty()) {
    return std::string("expected consumer_loop to resolve to an affine.for");
  }

  analysis.producerDepth = analysis.producerChain.size();
  analysis.consumerDepth = analysis.consumerChain.size();
  analysis.producerRoot = analysis.producerChain.front();
  analysis.consumerRoot = analysis.consumerChain.front();

  analysis.consumerPrefixIVs.clear();
  analysis.consumerPrefixIVs.reserve(analysis.consumerDepth);
  for (affine::AffineForOp loop : analysis.consumerChain)
    analysis.consumerPrefixIVs.push_back(loop.getInductionVar());

  if (analysis.producerRoot == analysis.consumerRoot) {
    return std::string(
        "producer and consumer must belong to different root loop nests");
  }
  if (analysis.producerRoot->getBlock() != analysis.consumerRoot->getBlock()) {
    return std::string(
        "producer and consumer loop nests must be in the same block");
  }
  if (analysis.producerDepth < analysis.consumerDepth) {
    return std::string(
        "producer loop nest depth is shallower than consumer depth");
  }
  return std::nullopt;
}

static std::optional<std::string>
applyNoDependenceMove(transform::TransformRewriter &rewriter,
                      ComputeAtAnalysis &analysis) {
  unsigned consumerDepth = analysis.consumerDepth;
  unsigned producerDepth = analysis.producerDepth;

  // We only rewrite a single-path producer prefix; imperfect control flow in
  // this prefix would make region move/remap semantics ambiguous.
  unsigned prefixDepthToValidate =
      producerDepth == consumerDepth ? producerDepth : consumerDepth + 1;
  if (!hasSinglePathLoopPrefix(analysis.producerChain, prefixDepthToValidate)) {
    return std::string(
        "producer loop prefix to be rewritten must be perfectly nested");
  }

  // No-dependence move must preserve top-level order and cannot jump over
  // side-effecting operations between producer root and consumer root.
  if (!analysis.producerRoot->isBeforeInBlock(analysis.consumerRoot)) {
    return std::string(
        "producer root loop must appear before consumer root loop");
  }
  if (hasBlockingSideEffectsBetween(analysis.producerRoot,
                                    analysis.consumerRoot)) {
    return std::string("cannot move producer across side-effecting operations "
                       "between producer and consumer roots");
  }

  // No-dependence path currently supports constant-bound prefix reasoning only;
  // subset bounds are handled by generating an affine.if guard.
  FailureOr<ConstantBoundsPrefix> producerBounds =
      getConstantBoundsPrefix(analysis.producerChain, consumerDepth);
  FailureOr<ConstantBoundsPrefix> consumerBounds =
      getConstantBoundsPrefix(analysis.consumerChain, consumerDepth);
  if (failed(producerBounds) || failed(consumerBounds)) {
    return std::string("compute_at currently supports only constant-bounds "
                       "loops for no-dependence move");
  }

  bool identicalBounds = hasIdenticalBounds(*producerBounds, *consumerBounds);
  if (!identicalBounds && !isSubsetBounds(*producerBounds, *consumerBounds)) {
    return std::string("producer loop bounds must be identical to, or a "
                       "subset of, consumer bounds");
  }

  // Move producer body/subtree under consumer loop. If bounds are subset-only,
  // first materialize an affine.if so execution stays within producer domain.
  Block *destination = analysis.consumerLoop.getBody();
  Region *ivRemapRegion = &analysis.consumerLoop.getRegion();
  if (!identicalBounds) {
    rewriter.setInsertionPointToStart(analysis.consumerLoop.getBody());
    auto ifOp = affine::AffineIfOp::create(
        rewriter, analysis.consumerLoop.getLoc(),
        buildSubsetGuardSet(rewriter, *producerBounds),
        analysis.consumerPrefixIVs,
        /*withElseRegion=*/false);
    destination = ifOp.getThenBlock();
    ivRemapRegion = &ifOp.getThenRegion();
  }

  if (producerDepth == consumerDepth) {
    Block *producerInnermostBody = analysis.producerChain.back().getBody();
    Value consumerInnermostIV = analysis.consumerChain.back().getInductionVar();
    rewriter.eraseOp(producerInnermostBody->getTerminator());
    rewriter.inlineBlockBefore(producerInnermostBody, destination,
                               destination->begin(),
                               ValueRange{consumerInnermostIV});
  } else {
    Operation *producerSubtree =
        analysis.producerChain[consumerDepth].getOperation();
    rewriter.moveOpBefore(producerSubtree, destination, destination->begin());
  }

  unsigned remapDepth = consumerDepth;
  if (producerDepth == consumerDepth)
    remapDepth -= 1;
  // Rewrite producer IV uses to consumer IVs in the moved region prefix.
  remapProducerIVPrefix(analysis.producerChain, analysis.consumerChain,
                        remapDepth, *ivRemapRegion);
  analysis.producerRoot.erase();
  return std::nullopt;
}

static bool mayWriteAliasingMemref(Operation *op, Value memref,
                                   AliasAnalysis &aliasAnalysis) {
  if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(op))
    return !aliasAnalysis.alias(writeOp.getMemRef(), memref).isNo();

  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    iface.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      if (!isa<MemoryEffects::Write>(effect.getEffect()))
        continue;
      Value effectValue = effect.getValue();
      if (!effectValue)
        return true;
      if (!aliasAnalysis.alias(effectValue, memref).isNo())
        return true;
    }
    return false;
  }

  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Operation &nested : block) {
          if (mayWriteAliasingMemref(&nested, memref, aliasAnalysis))
            return true;
        }
      }
    }
    return false;
  }

  // Unknown ops are conservatively assumed to possibly write.
  return true;
}

static void runComputeAtPostCleanup(affine::AffineForOp consumerLoop) {
  // Run local store-to-load forwarding in the transformed loop nest only.
  // This avoids full-function affineScalarReplace on large kernels.
  Operation *scopeOp = nullptr;
  if (auto kernel = consumerLoop->getParentOfType<func::FuncOp>())
    scopeOp = kernel.getOperation();
  else
    scopeOp = consumerLoop->getParentOp();
  AliasAnalysis aliasAnalysis(scopeOp);

  SmallVector<affine::AffineReadOpInterface, 16> loads;
  consumerLoop.walk(
      [&](affine::AffineReadOpInterface loadOp) { loads.push_back(loadOp); });

  SmallVector<Operation *, 16> loadsToErase;
  for (affine::AffineReadOpInterface loadOp : loads) {
    Operation *load = loadOp.getOperation();
    if (!load || !load->getBlock())
      continue;

    Value loadMemref = loadOp.getMemRef();
    affine::MemRefAccess loadAccess(load);
    Operation *forwardingStore = nullptr;

    for (Operation *curr = load->getPrevNode(); curr;
         curr = curr->getPrevNode()) {
      if (auto storeOp = dyn_cast<affine::AffineWriteOpInterface>(curr)) {
        // Non-aliasing stores cannot affect this load.
        if (aliasAnalysis.alias(storeOp.getMemRef(), loadMemref).isNo())
          continue;

        affine::MemRefAccess storeAccess(curr);
        if (storeAccess == loadAccess)
          forwardingStore = curr;
        // Any aliasing store blocks the search in this block.
        break;
      }

      if (mayWriteAliasingMemref(curr, loadMemref, aliasAnalysis))
        break;
    }

    if (!forwardingStore)
      continue;
    Value storeValue =
        cast<affine::AffineWriteOpInterface>(forwardingStore).getValueToStore();
    if (storeValue.getType() != loadOp.getValue().getType())
      continue;
    loadOp.getValue().replaceAllUsesWith(storeValue);
    loadsToErase.push_back(load);
  }

  for (Operation *load : loadsToErase)
    load->erase();
}

DiagnosedSilenceableFailure
transform::ComputeAtOp::apply(transform::TransformRewriter &rewriter,
                              transform::TransformResults &results,
                              transform::TransformState &states) {
  (void)results;

  auto consumerLoops = states.getPayloadOps(getConsumerLoop());
  auto producers = states.getPayloadOps(getProducer());
  if (!llvm::hasSingleElement(consumerLoops) ||
      !llvm::hasSingleElement(producers)) {
    return emitSilenceableError()
           << "expected exactly one producer and one consumer loop";
  }

  Operation *producerOp = *producers.begin();
  auto consumerLoop = dyn_cast<affine::AffineForOp>(*consumerLoops.begin());
  if (!consumerLoop) {
    return emitSilenceableError()
           << "expected consumer_loop to resolve to an affine.for";
  }

  ComputeAtAnalysis analysis;
  if (auto reason = analyzeComputeAt(producerOp, consumerLoop, analysis)) {
    return emitSilenceableError() << *reason;
  }

  // Classify producer->consumer dependence first; this decides whether to use
  // affine fusion or conservative manual move.
  auto depTypeOr = checkDependencies(
      analysis.producerRoot, analysis.consumerLoop, analysis.consumerDepth);
  if (failed(depTypeOr)) {
    return emitSilenceableError()
           << "dependence analysis failed; refusing compute_at";
  }
  DependenceType depType = *depTypeOr;

  if ((depType & DependenceType::RAW) != DependenceType::NONE) {
    // RAW dependence requires true producer-consumer fusion to preserve
    // semantics while changing loop placement.
    auto reason = tryAffineLoopFusion(
        analysis.producerRoot, analysis.consumerLoop, analysis.consumerDepth);
    if (reason.has_value()) {
      return emitSilenceableError()
             << "cannot fuse producer and consumer loop nests: "
             << reason.value();
    }
  } else if (depType == DependenceType::NONE) {
    // No dependence: perform structurally-checked move/inline + IV remap.
    if (auto reason = applyNoDependenceMove(rewriter, analysis)) {
      return emitSilenceableError() << *reason;
    }
  } else {
    return emitSilenceableError()
           << "compute_at does not support WAR/WAW-only dependences";
  }

  runComputeAtPostCleanup(analysis.consumerLoop);

  // results.set(cast<OpResult>(getResult()), {targetForOp});
  return DiagnosedSilenceableFailure::success();
}

void transform::ComputeAtOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getProducerMutable(), effects);
  // consumer_loop is read-only and must remain reusable.
  onlyReadsHandle(getConsumerLoopMutable(), effects);
  modifiesPayload(effects);
}

///===----------------------------------------------------------------------===//
/// ReuseAt implementation
///===----------------------------------------------------------------------===///
// Extract the last constant term encountered in an affine expression.
// ReuseAt uses this as a lightweight ordering key for sliding-window loads.
static int64_t findConstantExprValue(const AffineExpr &exp) {
  int64_t value = -1;
  // TODO: only support one constant now
  exp.walk([&](AffineExpr inner) {
    if (llvm::isa<AffineConstantExpr>(inner))
      value = llvm::cast<AffineConstantExpr>(inner).getValue();
  });
  return value;
}

namespace {
struct ExprCompare {
  // Order expressions by their constant offset so we can process loads
  // from low-to-high offsets on the reuse axis.
  bool operator()(const AffineExpr &exp1, const AffineExpr &exp2) const {
    int64_t val1 = findConstantExprValue(exp1);
    int64_t val2 = findConstantExprValue(exp2);
    return val1 < val2;
  }
};
struct LoopRoleInfo {
  DenseSet<Value> spatialIVs;
  DenseSet<Value> reductionIVs;
  DenseMap<Value, int64_t> reductionUpperBounds;
};
} // namespace

static bool valueDependsOnTargetLoad(Value value, Value target,
                                     DenseMap<Value, bool> &cache,
                                     SmallPtrSetImpl<Value> &visiting) {
  if (auto it = cache.find(value); it != cache.end())
    return it->second;
  if (!visiting.insert(value).second)
    return false;

  bool depends = false;
  if (auto loadOp = value.getDefiningOp<affine::AffineLoadOp>()) {
    depends = loadOp.getMemRef() == target;
  } else if (Operation *defOp = value.getDefiningOp()) {
    depends = llvm::any_of(defOp->getOperands(), [&](Value operand) {
      return valueDependsOnTargetLoad(operand, target, cache, visiting);
    });
  }

  visiting.erase(value);
  cache[value] = depends;
  return depends;
}

// ReuseAt assumes canonical affine loops: lb=0, step=1, constant bounds.
static bool requireNormalizedLoop(affine::AffineForOp forOp) {
  return forOp.getStepAsInt() == 1 && forOp.hasConstantBounds() &&
         forOp.getConstantLowerBound() == 0;
}

// Walk parent loops to the outermost loop of the selected axis loop.
static affine::AffineForOp getRootLoop(affine::AffineForOp loop) {
  affine::AffineForOp root = loop;
  while (auto parent = root->getParentOfType<affine::AffineForOp>())
    root = parent;
  return root;
}

// Resolve the axis handle to exactly one affine.for loop.
static FailureOr<affine::AffineForOp>
resolveAxisLoop(transform::TransformState &state, transform::ReuseAtOp op) {
  auto payloadOps = state.getPayloadOps(op.getAxis());
  if (!llvm::hasSingleElement(payloadOps))
    return failure();
  auto axisLoop = dyn_cast<affine::AffineForOp>(*payloadOps.begin());
  if (!axisLoop)
    return failure();
  return axisLoop;
}

// Classify each loop IV in the root nest:
// - spatial: contributes to store indexing
// - reduction: contributes to target-load indexing but not store indexing
// Also cache reduction loop upper bounds for span/distance derivation.
static LogicalResult
classifyLoopRoles(affine::AffineForOp rootForOp, Value target,
                  LoopRoleInfo &roles,
                  SmallVectorImpl<affine::AffineForOp> &allLoops) {
  WalkResult walkResult = rootForOp.walk([&](affine::AffineForOp forOp) {
    if (!requireNormalizedLoop(forOp))
      return WalkResult::interrupt();
    allLoops.push_back(forOp);
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  DenseSet<Value> spatialCandidates;
  DenseMap<Value, bool> loadDependenceCache;
  rootForOp.walk([&](affine::AffineStoreOp storeOp) {
    SmallPtrSet<Value, 16> visiting;
    if (!valueDependsOnTargetLoad(storeOp.getValueToStore(), target,
                                  loadDependenceCache, visiting)) {
      return WalkResult::advance();
    }
    AffineMap storeMap = storeOp.getAffineMap();
    auto storeOperands = storeOp.getMapOperands();
    for (AffineExpr resultExpr : storeMap.getResults()) {
      for (affine::AffineForOp loop : allLoops) {
        if (affineExprUsesValue(resultExpr, storeOperands,
                                storeMap.getNumDims(),
                                loop.getInductionVar())) {
          spatialCandidates.insert(loop.getInductionVar());
        }
      }
    }
    return WalkResult::advance();
  });

  DenseSet<Value> loadCandidates;
  rootForOp.walk([&](affine::AffineLoadOp loadOp) {
    if (loadOp.getMemRef() != target)
      return WalkResult::advance();
    AffineMap loadMap = loadOp.getAffineMap();
    auto loadOperands = loadOp.getMapOperands();
    for (AffineExpr resultExpr : loadMap.getResults()) {
      for (affine::AffineForOp loop : allLoops) {
        if (affineExprUsesValue(resultExpr, loadOperands, loadMap.getNumDims(),
                                loop.getInductionVar())) {
          loadCandidates.insert(loop.getInductionVar());
        }
      }
    }
    return WalkResult::advance();
  });

  if (loadCandidates.empty())
    return failure();

  roles.spatialIVs = std::move(spatialCandidates);
  for (Value iv : loadCandidates) {
    if (!roles.spatialIVs.contains(iv))
      roles.reductionIVs.insert(iv);
  }

  for (affine::AffineForOp loop : allLoops) {
    Value iv = loop.getInductionVar();
    if (!roles.reductionIVs.contains(iv))
      continue;
    roles.reductionUpperBounds[iv] = loop.getConstantUpperBound();
  }
  return success();
}

// True if the loop IV is inferred as reduction-only in the current nest.
static bool isReductionLoop(affine::AffineForOp forOp,
                            const LoopRoleInfo &roles) {
  return roles.reductionIVs.contains(forOp.getInductionVar());
}

// True if the loop IV appears in spatial indexing (store side).
static bool isSpatialLoop(affine::AffineForOp forOp,
                          const LoopRoleInfo &roles) {
  return roles.spatialIVs.contains(forOp.getInductionVar());
}

// Stage 1 analysis:
// Collect load expressions under the selected axis loop and infer
// per-dimension span plus axis stride used by the later rewrites.
static std::optional<std::string>
analyzeSpanAndStride(affine::AffineForOp axisLoop, Value target, unsigned rank,
                     const DenseMap<Value, int64_t> &reductionUpperBounds,
                     RewriterBase &rewriter, SmallVectorImpl<int64_t> &spans,
                     int64_t &stride) {
  SmallVector<SmallVector<AffineExpr>, 8> loadExprsByResultDim(rank);
  DenseMap<AffineExpr, Value> dimExprToOperand;
  int numTargetLoads = 0;

  axisLoop.walk([&](affine::AffineLoadOp loadOp) {
    if (loadOp.getMemRef() != target)
      return WalkResult::advance();

    ++numTargetLoads;
    AffineMap loadMap = loadOp.getAffineMap();
    for (unsigned i = 0; i < rank; ++i)
      loadExprsByResultDim[i].push_back(loadMap.getResult(i));

    for (auto operandItem : llvm::enumerate(loadOp.getMapOperands())) {
      dimExprToOperand[rewriter.getAffineDimExpr(operandItem.index())] =
          operandItem.value();
    }
    return WalkResult::advance();
  });

  if (numTargetLoads == 0)
    // Without target loads in the selected scope, no reuse window can be built.
    return std::string("cannot find target loads under selected axis loop");

  auto computeSpanForResultDim = [&](unsigned dim) -> FailureOr<int64_t> {
    if (loadExprsByResultDim[dim].empty())
      return static_cast<int64_t>(1);

    int64_t span = 0;
    AffineExpr baseExpr = loadExprsByResultDim[dim][0];
    int64_t baseCst = 0;

    if (isa<AffineDimExpr>(baseExpr)) {
      bool allDimExpr = true;
      for (int j = 0; j < numTargetLoads; ++j) {
        AffineExpr expr = loadExprsByResultDim[dim][j];
        AffineExpr diff = expr - baseExpr;
        if (!isa<AffineDimExpr>(expr))
          allDimExpr = false;
        if (auto cst = dyn_cast<AffineConstantExpr>(diff))
          span = std::max(span, cst.getValue() + 1);
      }
      if (allDimExpr) {
        auto dimExpr = cast<AffineDimExpr>(baseExpr);
        auto operandIt = dimExprToOperand.find(dimExpr);
        if (operandIt != dimExprToOperand.end() &&
            reductionUpperBounds.count(operandIt->second) > 0) {
          span = reductionUpperBounds.lookup(operandIt->second);
        }
      }
    } else if (isa<AffineConstantExpr>(baseExpr)) {
      for (int j = 0; j < numTargetLoads; ++j) {
        AffineExpr diff = loadExprsByResultDim[dim][j] - baseExpr;
        if (auto cst = dyn_cast<AffineConstantExpr>(diff))
          span = std::max(span, cst.getValue() + 1);
      }
    } else if (auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(baseExpr)) {
      AffineExpr lhs = binaryExpr.getLHS();
      AffineExpr rhs = binaryExpr.getRHS();
      if (auto dimExpr = dyn_cast<AffineDimExpr>(rhs)) {
        auto operandIt = dimExprToOperand.find(dimExpr);
        if (operandIt != dimExprToOperand.end() &&
            reductionUpperBounds.count(operandIt->second) > 0) {
          span = reductionUpperBounds.lookup(operandIt->second);
        }
      } else if (auto cst = dyn_cast<AffineConstantExpr>(rhs)) {
        int64_t value = cst.getValue();
        if (baseCst == 0)
          baseCst = value;
        span = std::max(span, value - baseCst + 1);
      }

      if (auto lhsBinary = dyn_cast<AffineBinaryOpExpr>(lhs)) {
        if (auto cst = dyn_cast<AffineConstantExpr>(lhsBinary.getRHS()))
          stride = cst.getValue();
      }
    } else {
      // Unsupported expression shape means we cannot safely derive span/stride.
      return failure();
    }

    if (span == 0)
      span = 1;
    return span;
  };

  spans.clear();
  spans.reserve(rank);
  for (unsigned i = 0; i < rank; ++i) {
    FailureOr<int64_t> span = computeSpanForResultDim(i);
    if (failed(span))
      // Fail fast when span inference is ambiguous to avoid unsafe rewrites.
      return std::string("unsupported load expression form for reuse analysis");
    spans.push_back(*span);
  }

  return std::nullopt;
}

// Stage 2 analysis:
// Infer the reusable axis and distance window from target loads, and collect
// ordered loads + reduction-dimension metadata for legality checks.
static std::optional<std::string>
analyzeAxisReuseWindow(affine::AffineForOp axisLoop, Value target, Value axisIV,
                       const DenseMap<Value, int64_t> &reductionUpperBounds,
                       RewriterBase &rewriter, unsigned &axis,
                       int64_t &distance,
                       std::set<AffineExpr, ExprCompare> &requestedAxisExprs,
                       SmallVectorImpl<affine::AffineLoadOp> &orderedLoadOps,
                       DenseMap<unsigned, int64_t> &reductionDimBounds) {
  auto insertLoadByAxisOffset = [&](affine::AffineLoadOp loadOp) {
    unsigned size = orderedLoadOps.size();
    AffineExpr lhs = loadOp.getAffineMap().getResult(axis);
    for (unsigned i = 0; i < size; ++i) {
      AffineExpr rhs = orderedLoadOps[i].getAffineMap().getResult(axis);
      if (findConstantExprValue(lhs) < findConstantExprValue(rhs)) {
        orderedLoadOps.insert(orderedLoadOps.begin() + i, loadOp);
        return;
      }
    }
    orderedLoadOps.push_back(loadOp);
  };

  std::optional<unsigned> detectedAxis;
  axis = 0;
  distance = -1;
  requestedAxisExprs.clear();
  orderedLoadOps.clear();
  reductionDimBounds.clear();

  axisLoop.walk([&](affine::AffineLoadOp loadOp) {
    if (loadOp.getMemRef() != target)
      return WalkResult::advance();

    AffineMap loadMap = loadOp.getAffineMap();
    unsigned numDims = loadMap.getNumDims();
    ValueRange operands = loadOp.getMapOperands();
    std::optional<unsigned> reductionDimOnAxis;

    for (unsigned j = 0; j < loadMap.getNumResults(); ++j) {
      AffineExpr expr = loadMap.getResult(j);
      bool resultUsesAxis =
          affineExprUsesValue(expr, operands, numDims, axisIV);
      if (!detectedAxis && resultUsesAxis)
        detectedAxis = j;
      bool isAxisResult = detectedAxis && *detectedAxis == j;

      for (unsigned i = 0; i < numDims; ++i) {
        if (!expr.isFunctionOfDim(i) || i >= operands.size())
          continue;
        auto reductionIt = reductionUpperBounds.find(operands[i]);
        if (reductionIt == reductionUpperBounds.end())
          continue;
        reductionDimBounds[i] = reductionIt->second;
        if (isAxisResult)
          reductionDimOnAxis = i;
      }
    }

    if (!detectedAxis || *detectedAxis >= loadMap.getNumResults())
      // If no result dimension can be matched to axisIV, this load is not
      // reusable.
      return WalkResult::interrupt();
    axis = *detectedAxis;

    AffineExpr axisExpr = loadMap.getResult(axis);
    insertLoadByAxisOffset(loadOp);

    if (reductionDimOnAxis) {
      int64_t ub = reductionUpperBounds.lookup(operands[*reductionDimOnAxis]);
      distance = std::max(distance, ub - 1);
      for (int64_t j = 0; j < ub; ++j) {
        AffineExpr ubExpr = rewriter.getAffineConstantExpr(j);
        AffineExpr expandedExpr = axisExpr.replace(
            rewriter.getAffineDimExpr(*reductionDimOnAxis), ubExpr);
        requestedAxisExprs.insert(expandedExpr);
      }
    } else {
      requestedAxisExprs.insert(axisExpr);
      AffineExpr offset = axisExpr - *requestedAxisExprs.begin();
      auto cst = dyn_cast<AffineConstantExpr>(offset);
      if (!cst)
        return WalkResult::interrupt();
      distance = std::max(distance, cst.getValue());
    }
    return WalkResult::advance();
  });

  if (!detectedAxis || requestedAxisExprs.empty())
    // Require a concrete axis and at least one requested index expression.
    return std::string("cannot find reusable load axis");
  axis = *detectedAxis;

  return std::nullopt;
}

// Stage 3 analysis:
// Validate the inferred axis window is a supported pattern and materialize
// the base axis expression used by index remapping.
static std::optional<std::string> validateAxisReuseWindow(
    const std::set<AffineExpr, ExprCompare> &requestedAxisExprs,
    ArrayRef<affine::AffineLoadOp> orderedLoadOps, unsigned axis,
    const DenseMap<unsigned, int64_t> &reductionDimBounds,
    AffineExpr &baseAxisExpr) {
  bool hasStrideOneWindow = false;
  for (AffineExpr expr : requestedAxisExprs) {
    if (requestedAxisExprs.find(expr + 1) != requestedAxisExprs.end()) {
      hasStrideOneWindow = true;
      break;
    }
  }
  if (!hasStrideOneWindow)
    // Current implementation only supports consecutive stride-1 reuse windows.
    return std::string("cannot find stride-1 reuse pattern on selected axis");

  baseAxisExpr = *requestedAxisExprs.begin();
  auto hasReductionDimInExpr = [&](AffineExpr expr) {
    for (auto [dim, _] : reductionDimBounds) {
      if (expr.isFunctionOfDim(dim))
        return true;
    }
    return false;
  };

  for (affine::AffineLoadOp loadOp : orderedLoadOps) {
    AffineExpr diff = loadOp.getAffineMap().getResult(axis) - baseAxisExpr;
    if (hasReductionDimInExpr(diff))
      // Reduction-indexed axis reuse needs extra handling not implemented yet.
      return std::string("reduction reuse not fully implemented");
    if (!isa<AffineConstantExpr>(diff))
      // Non-constant deltas break fixed-window addressing assumptions.
      return std::string("cannot support non-constant stride");
  }

  return std::nullopt;
}

DiagnosedSilenceableFailure
transform::ReuseAtOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  // Stage 0: resolve transform handles to concrete payload IR objects.
  // Resolve concrete payloads first; ReuseAt requires a single memref and loop.
  auto targets = llvm::to_vector(state.getPayloadValues(getTarget()));
  if (targets.size() != 1) {
    return emitSilenceableError()
           << "expected target handle to resolve to exactly one payload value";
  }
  Value target = targets.front();
  MemRefType targetType = dyn_cast<MemRefType>(target.getType());
  if (!targetType)
    return emitSilenceableError()
           << "expected target to resolve to a memref value";

  auto axisLoopOr = resolveAxisLoop(state, *this);
  if (failed(axisLoopOr))
    return emitSilenceableError()
           << "expected axis to resolve to exactly one affine.for loop";
  affine::AffineForOp axisLoop = *axisLoopOr;
  Value axisIV = axisLoop.getInductionVar();
  unsigned rank = targetType.getRank();

  // Stage 0.5: classify loop roles and reject unsupported axis selections.
  affine::AffineForOp rootLoop = getRootLoop(axisLoop);
  LoopRoleInfo roles;
  SmallVector<affine::AffineForOp, 8> allLoops;
  if (failed(classifyLoopRoles(rootLoop, target, roles, allLoops))) {
    return emitSilenceableError()
           << "failed to classify loop roles; loops must be normalized and "
              "target must be loaded in the axis stage";
  }
  // The chosen axis must be spatial (store-indexing), not reduction-only.
  if (isReductionLoop(axisLoop, roles))
    return emitSilenceableError()
           << "selected axis loop is classified as a reduction loop";
  if (!isSpatialLoop(axisLoop, roles))
    return emitSilenceableError()
           << "selected axis loop is not classified as a spatial loop";

  DenseMap<Value, int64_t> reductionUpperBounds = roles.reductionUpperBounds;
  int64_t stride = 1;
  SmallVector<int64_t, 8> spans;
  // Stage 1: infer shape/stride information used by buffer allocation/remap.
  if (auto reason =
          analyzeSpanAndStride(axisLoop, target, rank, reductionUpperBounds,
                               rewriter, spans, stride)) {
    return emitSilenceableError() << *reason;
  }

  std::set<AffineExpr, ExprCompare> requestedAxisExprs;
  SmallVector<affine::AffineLoadOp, 8> orderedLoadOps;
  DenseMap<unsigned, int64_t> reductionDimBounds;
  unsigned axis = 0;
  int64_t distance = -1;
  // Stage 2: infer reusable axis and requested index window.
  if (auto reason = analyzeAxisReuseWindow(
          axisLoop, target, axisIV, reductionUpperBounds, rewriter, axis,
          distance, requestedAxisExprs, orderedLoadOps, reductionDimBounds)) {
    return emitSilenceableError() << *reason;
  }

  AffineExpr baseAxisExpr;
  // Stage 3: validate pattern constraints before mutating IR.
  if (auto reason =
          validateAxisReuseWindow(requestedAxisExprs, orderedLoadOps, axis,
                                  reductionDimBounds, baseAxisExpr)) {
    return emitSilenceableError() << *reason;
  }

  // Stage 4: materialize the reuse buffer shape and allocate it at root scope.
  SmallVector<int64_t, 8> reuseBufferShape;
  for (unsigned i = 0; i < axis; ++i) {
    if (spans[i] > 1)
      reuseBufferShape.push_back(spans[i]);
  }
  reuseBufferShape.push_back(distance + 1);
  for (unsigned i = axis + 1; i < rank; ++i)
    reuseBufferShape.push_back(targetType.getShape()[i]);

  rewriter.setInsertionPoint(rootLoop);
  auto reuseBuffer = memref::AllocOp::create(
      rewriter, rootLoop.getLoc(),
      MemRefType::get(reuseBufferShape, targetType.getElementType()));
  if (Operation *targetDef = target.getDefiningOp()) {
    if (auto targetSymName =
            targetDef->getAttrOfType<StringAttr>(OpIdentifier)) {
      reuseBuffer->setAttr(
          OpIdentifier, StringAttr::get(reuseBuffer->getContext(),
                                        targetSymName.getValue() + ".reuse"));
    }
  }

  // Stage 5: extend the axis loop domain and rebase dependent arithmetic users.
  // Extend loop bound to include the warmup distance for the reuse window.
  int64_t originalUpperBound = axisLoop.getConstantUpperBound();
  axisLoop.setConstantUpperBound(originalUpperBound * stride + distance);

  for (auto &use : llvm::make_early_inc_range(axisIV.getUses())) {
    auto castOp = dyn_cast<arith::IndexCastOp>(use.getOwner());
    if (!castOp)
      continue;
    for (auto &castUse :
         llvm::make_early_inc_range(castOp.getResult().getUses())) {
      auto mulOp = dyn_cast<arith::MulIOp>(castUse.getOwner());
      if (!mulOp)
        continue;
      rewriter.setInsertionPoint(mulOp);
      Type dtype = mulOp.getResult().getType();
      // Rebase axis-related arithmetic to the shifted reuse window coordinates.
      auto strideCst =
          arith::ConstantOp::create(rewriter, mulOp.getLoc(), dtype,
                                    rewriter.getIntegerAttr(dtype, stride));
      auto shifted = arith::SubIOp::create(rewriter, mulOp.getLoc(),
                                           castOp.getResult(), strideCst);
      mulOp.replaceAllUsesWith(shifted.getResult());
      mulOp.erase();
    }
  }

  // Stage 6: rewrite stores so axis indices match the shifted/scaled loop
  // space.
  SmallVector<Operation *, 8> storesToErase;
  axisLoop.walk([&](affine::AffineStoreOp storeOp) {
    MemRefType storeType = dyn_cast<MemRefType>(storeOp.getMemRef().getType());
    if (!storeType ||
        (storeType.getRank() == 1 && storeType.getShape()[0] == 1)) {
      return WalkResult::advance();
    }

    rewriter.setInsertionPoint(storeOp);
    SmallVector<AffineExpr, 8> remappedIndices;
    AffineMap oldMap = storeOp.getAffineMap();
    int axisResultIdx = findMemRefAxisFromIVs(storeOp, axisIV);
    for (unsigned i = 0, e = oldMap.getNumResults(); i < e; ++i) {
      if (static_cast<int>(i) != axisResultIdx) {
        remappedIndices.push_back(oldMap.getResult(i));
        continue;
      }
      // Axis dimension is remapped by distance/stride; others keep original
      // form.
      if (stride != 1) {
        AffineExpr strideExpr = rewriter.getAffineConstantExpr(stride);
        remappedIndices.push_back(
            (oldMap.getResult(i) - distance).floorDiv(strideExpr));
      } else {
        remappedIndices.push_back(oldMap.getResult(i) - distance);
      }
    }

    AffineMap newMap = AffineMap::get(storeType.getRank(), 0, remappedIndices,
                                      rewriter.getContext());
    affine::AffineStoreOp::create(
        rewriter, storeOp->getLoc(), storeOp.getValueToStore(),
        storeOp.getMemRef(), newMap, storeOp.getIndices());
    storesToErase.push_back(storeOp);
    return WalkResult::advance();
  });
  for (Operation *store : storesToErase)
    store->erase();

  // Stage 7: update affine.if guards that directly depend on the axis IV.
  axisLoop.walk([&](affine::AffineIfOp ifOp) {
    int axisOperandIdx = -1;
    for (auto item : llvm::enumerate(ifOp.getOperands())) {
      if (item.value() == axisIV) {
        axisOperandIdx = static_cast<int>(item.index());
        break;
      }
    }
    if (axisOperandIdx == -1)
      return WalkResult::advance();

    IntegerSet condSet = ifOp.getIntegerSet();
    AffineExpr distanceExpr = rewriter.getAffineConstantExpr(distance);
    AffineExpr strideAdjust = rewriter.getAffineConstantExpr(stride - 1) *
                              rewriter.getAffineDimExpr(axisOperandIdx);

    SmallVector<AffineExpr, 8> updatedConds;
    for (AffineExpr cond : condSet.getConstraints()) {
      bool hasNegativeSign = false;
      cond.walk([&](AffineExpr expr) {
        auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
        if (!binary || binary.getKind() != AffineExprKind::Mul)
          return;
        if (auto cst = dyn_cast<AffineConstantExpr>(binary.getRHS()))
          hasNegativeSign = cst.getValue() == -1;
      });

      if (!cond.isFunctionOfDim(axisOperandIdx)) {
        updatedConds.push_back(cond);
      } else if (hasNegativeSign) {
        // Preserve inequality direction when the condition is negated.
        updatedConds.push_back(cond + distanceExpr + strideAdjust);
      } else {
        updatedConds.push_back(cond - distanceExpr - strideAdjust);
      }
    }

    IntegerSet updatedSet = IntegerSet::get(condSet.getNumDims(), 0,
                                            updatedConds, condSet.getEqFlags());
    ifOp.setIntegerSet(updatedSet);
    return WalkResult::advance();
  });

  // Stage 8: rewrite original target loads to read from the reuse buffer.
  for (affine::AffineLoadOp loadOp : orderedLoadOps) {
    rewriter.setInsertionPoint(loadOp);

    AffineMap loadMap = loadOp.getAffineMap();
    SmallVector<Value, 8> operands = llvm::to_vector(loadOp.getMapOperands());
    SmallVector<AffineExpr, 8> reuseLoadExprs;
    SmallVector<Value, 8> reuseLoadOperands;

    reuseLoadExprs.push_back(loadMap.getResult(axis) - baseAxisExpr);

    unsigned reducedRank = 0;
    unsigned operandIdx = 0;
    for (unsigned i = 0; i < axis; ++i) {
      if (spans[i] > 1)
        reuseLoadExprs.push_back(loadMap.getResult(i));
    }

    SmallVector<AffineExpr, 8> dims;
    for (unsigned i = 0; i < axis + 1; ++i) {
      AffineExpr expr = loadMap.getResult(i);
      if (!isa<AffineConstantExpr>(expr)) {
        ++operandIdx;
        dims.push_back(rewriter.getAffineDimExpr(0)); // placeholder
      }
    }
    for (unsigned i = axis + 1; i < rank; ++i)
      dims.push_back(rewriter.getAffineDimExpr(reducedRank++));

    for (unsigned i = axis + 1; i < rank; ++i) {
      AffineExpr expr = loadMap.getResult(i);
      reuseLoadExprs.push_back(expr.replaceDims(dims));
      reuseLoadOperands.push_back(operands[operandIdx++]);
    }

    AffineMap reuseLoadMap =
        AffineMap::get(reducedRank, 0, reuseLoadExprs, rewriter.getContext());
    rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
        loadOp, reuseBuffer, reuseLoadMap, reuseLoadOperands);
  }

  // Stage 9: return the allocated reuse buffer as transform result.
  results.set(cast<OpResult>(getResult()), {reuseBuffer.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

///===----------------------------------------------------------------------===///
/// LoopPipeline implementation
///===----------------------------------------------------------------------===///
DiagnosedSilenceableFailure
transform::TagPipelineOp::applyToOne(transform::TransformRewriter &rewriter,
                                     Operation *target,
                                     transform::ApplyToEachResultList &results,
                                     transform::TransformState &state) {
  if (!target || !isa<LoopLikeOpInterface>(target)) {
    return emitSilenceableError()
           << "expected target to resolve to exactly one loop-like operation";
  }
  auto ii = getIiAttr().getInt();
  if (ii <= 0) {
    return emitSilenceableError() << "expected ii to be a positive integer";
  }
  target->setAttr("pipeline.ii", getIiAttr());
  return DiagnosedSilenceableFailure::success();
}

///===----------------------------------------------------------------------===///
/// LoopUnroll implementation
///===----------------------------------------------------------------------===///
DiagnosedSilenceableFailure
transform::TagUnrollOp::applyToOne(transform::TransformRewriter &rewriter,
                                   Operation *target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  if (!target || !isa<LoopLikeOpInterface>(target)) {
    return emitSilenceableError()
           << "expected target to resolve to exactly one loop-like operation";
  }
  auto factor = getFactorAttr().getInt();
  if (factor < 0) {
    return emitSilenceableError()
           << "expected unroll factor to be a non-negative "
           << "integer (0 for full unroll)";
  }
  target->setAttr("unroll.f", getFactorAttr());
  return DiagnosedSilenceableFailure::success();
}
