/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Support/Utils.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>

using namespace mlir;
using namespace allo;

using AffineLoopBand = SmallVector<AffineForOp, 6>;

//===----------------------------------------------------------------------===//
// Loop transformation
//===----------------------------------------------------------------------===//

namespace mlir {
namespace allo {

struct ExprCompare {
  int findConstantExpr(const AffineExpr &exp) const {
    int value = -1;
    // TODO: only support one constant now
    exp.walk([&](AffineExpr inner) {
      if (llvm::isa<AffineConstantExpr>(inner))
        value = llvm::cast<AffineConstantExpr>(inner).getValue();
    });
    return value;
  }
  bool operator()(const AffineExpr &exp1, const AffineExpr &exp2) const {
    int val1 = findConstantExpr(exp1);
    int val2 = findConstantExpr(exp2);
    return val1 < val2;
  }
};

Attribute createZeroAttr(OpBuilder &builder, mlir::Type elementType) {
  if (llvm::isa<FloatType>(elementType))
    return builder.getFloatAttr(elementType, 0.0);
  if (llvm::isa<IntegerType>(elementType))
    return builder.getIntegerAttr(elementType, 0);
  return {};
}

LogicalResult runSplitting(func::FuncOp &f, SplitOp &splitOp) {
  // 1) Get the schedule
  unsigned int factor = splitOp.getFactor();
  auto loopHandle =
      dyn_cast<CreateLoopHandleOp>(splitOp.getLoop().getDefiningOp());
  auto opHandle =
      dyn_cast<CreateOpHandleOp>(loopHandle.getOp().getDefiningOp());
  const auto loop_name = loopHandle.getLoopName();
  const auto op_name = opHandle.getOpName();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 3) Find the requested loop
  bool isOuterMost = false;
  AffineLoopBand band;
  rootForOp->walk([&](AffineForOp forOp) {
    if (band.size() == 0 && loop_name == getLoopName(forOp)) {
      band.push_back(forOp);
      if (forOp->hasAttr("op_name"))
        isOuterMost = true;
    }
  });
  // handle exception
  if (band.size() == 0) {
    splitOp.emitError("Cannot find Loop ")
        << loop_name.str() << " in Stage " << op_name.str();
    return failure();
  }
  if (factor >= band[0].getConstantUpperBound()) {
    splitOp.emitError("The requested tiling factor (")
        << factor << ") is larger than the upper bound ("
        << band[0].getConstantUpperBound() << ") of the loop";
    return failure();
  }

  // 4) Split the loop
  SmallVector<unsigned, 6> tileSizes;
  tileSizes.push_back(factor);
  AffineLoopBand tiledNest;
  if (failed(tilePerfectlyNested(band, tileSizes, &tiledNest)))
    return failure();
  if (isOuterMost)
    rootForOp = tiledNest[0];

  // 5) Loop normalization
  // Note: 5) & 6) are used for making the loop bound constants
  //       Otherwise, loops are not perfectly nested
  if (failed(normalizeAffineFor(tiledNest[0])) ||
      failed(normalizeAffineFor(tiledNest[1])))
    return failure();
  auto ub = tiledNest[1].getUpperBound();
  auto ubMap = ub.getMap();
  if (ubMap.isConstant()) {
    // Exception case that cannot change loop bound:
    // #map1 = affine_map<(d0, d1) -> (7, -d0 + 1024)>
    // %5 = affine.apply #map0(%arg3)
    // affine.for %arg4 = 0 to min #map1(%5, %5)
    auto cstUb =
        llvm::dyn_cast<AffineConstantExpr>(ubMap.getResult(0)).getValue();
    OpBuilder opBuilder(tiledNest[1]);
    tiledNest[1].setUpperBound({}, opBuilder.getConstantAffineMap(cstUb));
  } else {
    // f.dump();
    // auto addMap =
    //     AffineMap::get(/*numDims=*/1, /*numSymbols=*/0, ubMap.getResult(1));
    // auto outerIV = tiledNest[1].getUpperBoundOperands()[0];
    // auto mulMap = tiledNest[1].getUpperBound().getMap();
    // auto composedMap = addMap.compose(mulMap);
    // SmallVector<AffineExpr> newExprs{ubMap.getResult(0),
    //                                  composedMap.getResult(0)};
    // auto finalMinMap = AffineMap::get(/*numDims=*/1, /*numSymbols=*/0,
    // newExprs,
    //                                   tiledNest[1].getContext());
    // tiledNest[1].setUpperBound(outerIV, finalMinMap);
  }

  // 6) Sink AffineApply Operations
  auto fstApply = *(tiledNest[0].getOps<AffineApplyOp>().begin());
  auto sndApply = *(tiledNest[1].getOps<AffineApplyOp>().begin());
  WalkResult result = rootForOp->walk(
      [&](AffineForOp forOp) -> WalkResult { // from the innermost
        sndApply->moveBefore(&(*forOp.getBody()->getOperations().begin()));
        // definition should come before reference
        bool isDominance = true;
        for (auto user : sndApply->getUsers()) {
          DominanceInfo domInfo;
          if (!domInfo.properlyDominates(sndApply->getResult(0), user)) {
            isDominance = false;
            break;
          }
        }
        if (isDominance)
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  if (result.wasInterrupted())
    fstApply->moveBefore(sndApply);

  // 7) Add names to new loops
  SmallVector<std::string, 6> newNameArr;
  newNameArr.push_back(loop_name.str() + ".outer");
  newNameArr.push_back(loop_name.str() + ".inner");
  setLoopNames(tiledNest, newNameArr);
  if (isOuterMost)
    setStageName(tiledNest[0], op_name);

  // 8) Create new loop handles
  auto firstOp = *(f.getOps<AffineForOp>().begin());
  OpBuilder builder(firstOp);
  auto outer = builder.create<CreateLoopHandleOp>(
      firstOp->getLoc(), LoopHandleType::get(firstOp->getContext()),
      opHandle.getResult(),
      StringAttr::get(firstOp->getContext(), newNameArr[0]));
  auto inner = builder.create<CreateLoopHandleOp>(
      firstOp->getLoc(), LoopHandleType::get(firstOp->getContext()),
      opHandle.getResult(),
      StringAttr::get(firstOp->getContext(), newNameArr[1]));

  // 9) Link the loop handles with SSA values
  splitOp.getResult(0).replaceAllUsesWith(outer);
  splitOp.getResult(1).replaceAllUsesWith(inner);

  return success();
}

LogicalResult runTiling(func::FuncOp &f, TileOp &tileOp) {
  // 1) Get the schedule
  unsigned int x_factor = tileOp.getXFactor();
  unsigned int y_factor = tileOp.getYFactor();
  auto loopHandle =
      dyn_cast<CreateLoopHandleOp>(tileOp.getXLoop().getDefiningOp());
  auto opHandle =
      dyn_cast<CreateOpHandleOp>(loopHandle.getOp().getDefiningOp());
  const auto x_loop =
      dyn_cast<CreateLoopHandleOp>(tileOp.getXLoop().getDefiningOp())
          .getLoopName();
  const auto y_loop =
      dyn_cast<CreateLoopHandleOp>(tileOp.getYLoop().getDefiningOp())
          .getLoopName();
  const auto op_name = opHandle.getOpName();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 3) Find the requested loops
  bool isOuterMost = false;
  SmallVector<StringRef, 6> nameArr;
  nameArr.push_back(x_loop);
  nameArr.push_back(y_loop);
  AffineLoopBand band;
  WalkResult result = rootForOp.walk([&](AffineForOp forOp) -> WalkResult {
    if (findContiguousNestedLoops(forOp, band, nameArr))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  // handle exception
  if (!result.wasInterrupted()) {
    tileOp.emitError("Cannot find contiguous nested loops starting from Loop ")
        << x_loop.str();
    return failure();
  }
  if (x_factor >= band[0].getConstantUpperBound()) {
    tileOp.emitError("The requested tiling factor (")
        << x_factor << ") is larger than the upper bound ("
        << band[0].getConstantUpperBound() << ") of the loop";
    return failure();
  }
  if (y_factor >= band[1].getConstantUpperBound()) {
    tileOp.emitError("The requested tiling factor (")
        << y_factor << ") is larger than the upper bound ("
        << band[1].getConstantUpperBound() << ") of the loop";
    return failure();
  }
  if (llvm::cast<AffineForOp>(band[0])->hasAttr("op_name"))
    isOuterMost = true;

  // 4) Tile the loops
  SmallVector<unsigned, 6> tileSizes;
  tileSizes.push_back(x_factor);
  tileSizes.push_back(y_factor);
  AffineLoopBand tiledNest;
  if (failed(tilePerfectlyNested(band, tileSizes, &tiledNest)))
    return failure();
  if (isOuterMost)
    rootForOp = tiledNest[0];

  // 5) Loop normalization
  // Note: 5) & 6) are used for making the loop bound constants
  //       Otherwise, loops are not perfectly nested
  for (int i = 0; i < 4; ++i)
    if (failed(normalizeAffineFor(tiledNest[i])))
      return failure();
  // the tiled factor loops are the inner two
  for (int i = 2; i < 4; ++i) {
    auto ub = tiledNest[i].getUpperBound();
    auto ubMap = ub.getMap();
    if (ubMap.isConstant()) {
      auto cstUb =
          llvm::dyn_cast<AffineConstantExpr>(ubMap.getResult(0)).getValue();
      OpBuilder opBuilder(tiledNest[i]);
      tiledNest[i].setUpperBound({}, opBuilder.getConstantAffineMap(cstUb));
    } else {
      auto addMap =
          AffineMap::get(/*numDims=*/1, /*numSymbols=*/0, ubMap.getResult(1));
      auto applyOp = dyn_cast<AffineApplyOp>(
          tiledNest[i].getUpperBoundOperands()[0].getDefiningOp());
      auto outerIV = applyOp.getOperand(0);
      auto mulMap = applyOp.getAffineMap();
      auto composedMap = addMap.compose(mulMap);
      SmallVector<AffineExpr> newExprs{ubMap.getResult(0),
                                       composedMap.getResult(0)};
      auto finalMinMap = AffineMap::get(/*numDims=*/1, /*numSymbols=*/0,
                                        newExprs, tiledNest[i].getContext());
      tiledNest[i].setUpperBound(outerIV, finalMinMap);
    }
  }

  // 6) Sink AffineApply Operations
  for (int i = 1; i >= 0; --i) { // from inner to outer
    auto fstApply = *(tiledNest[i].getOps<AffineApplyOp>().begin());
    auto sndApply = *(tiledNest[i + 2].getOps<AffineApplyOp>().begin());
    WalkResult result = rootForOp->walk(
        [&](AffineForOp forOp) -> WalkResult { // from the innermost
          sndApply->moveBefore(&(*forOp.getBody()->getOperations().begin()));
          // definition should come before reference
          bool isDominance = true;
          for (auto user : sndApply->getUsers()) {
            DominanceInfo domInfo;
            if (!domInfo.properlyDominates(sndApply->getResult(0), user)) {
              isDominance = false;
              break;
            }
          }
          if (isDominance)
            return WalkResult::interrupt();
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      fstApply->moveBefore(sndApply);
  }

  // 7) Add names to new loops
  SmallVector<std::string, 6> newNameArr;
  newNameArr.push_back(x_loop.str() + ".outer");
  newNameArr.push_back(x_loop.str() + ".inner");
  newNameArr.push_back(y_loop.str() + ".outer");
  newNameArr.push_back(y_loop.str() + ".inner");
  setLoopNames(tiledNest, newNameArr);
  if (isOuterMost)
    setStageName(tiledNest[0], op_name);

  // 8) Create new loop handles &
  //    Link the loop handles with SSA values
  auto firstOp = *(f.getOps<AffineForOp>().begin());
  OpBuilder builder(firstOp);
  for (int i = 0; i < 4; ++i) {
    auto handle = builder.create<CreateLoopHandleOp>(
        firstOp->getLoc(), LoopHandleType::get(firstOp->getContext()),
        opHandle.getResult(),
        StringAttr::get(firstOp->getContext(), newNameArr[i]));
    tileOp.getResult(i).replaceAllUsesWith(handle);
  }

  return success();
}

LogicalResult runReordering(func::FuncOp &f, ReorderOp &reorderOp) {
  // 1) Get the schedule
  const auto loopsToReorder = reorderOp.getLoops(); // operand_range
  auto loopHandle =
      dyn_cast<CreateLoopHandleOp>(loopsToReorder[0].getDefiningOp());
  const auto op_name =
      dyn_cast<CreateOpHandleOp>(loopHandle.getOp().getDefiningOp())
          .getOpName();
  if (loopsToReorder.size() < 2) {
    reorderOp.emitError("Should at least input 2 loops to be reordered");
    return failure();
  }

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 3) Get the maximal perfect nest
  //    This should be done first to resolve imperfect loops
  AffineLoopBand nest;
  getPerfectlyNestedLoops(nest, rootForOp);

  // 4) Traverse all the loops in the stage
  //    Get a mapping from loop name to id
  std::map<std::string, unsigned> oldName2ID;
  SmallVector<std::string> oldLoopNames;
  unsigned int curr_depth = 0;
  for (AffineForOp forOp : nest) {
    std::string loop_name = getLoopName(forOp).str();
    oldName2ID[loop_name] = curr_depth;
    oldLoopNames.push_back(loop_name);
    curr_depth++;
  }

  // 5) Traverse all the input arguments that need to be reordered and
  // construct permMap
  // Possible inputs:
  // a) # arguments = # loops: (i,j,k)->(k,j,i)
  // b) # arguments != # loops: input (k,i), but should be the same as a)

  // 5.1) Map input arguments to the corresponding loop names
  SmallVector<std::string> nameOfLoopsToReorder;
  for (auto loop : loopsToReorder) {
    nameOfLoopsToReorder.push_back(
        llvm::dyn_cast<StringAttr>(loop.getDefiningOp()->getAttr("loop_name"))
            .getValue()
            .str());
  }

  // 5.2) Make Case b) to Case a)
  //      i.e. fill in all the missing loops in Case b)
  SmallVector<std::string> nameOfAllLoopsWithNewOrder;
  unsigned int cntInArgs = 0;
  for (unsigned int i = 0, e = oldLoopNames.size(); i < e; ++i) {
    auto name = oldLoopNames[i];
    auto iterator = std::find(nameOfLoopsToReorder.begin(),
                              nameOfLoopsToReorder.end(), name);
    if (iterator != nameOfLoopsToReorder.end()) { // name in the arguments
      nameOfAllLoopsWithNewOrder.push_back(nameOfLoopsToReorder[cntInArgs++]);
    } else { // not in
      nameOfAllLoopsWithNewOrder.push_back(name);
    }
  }

  // 5.3) Traverse the original loop nests and create a new order (permMap) for
  // the loops, where permMap[i] means the ith loop in the original nests will
  // become the permMap[i]-th loop
  unsigned int outerMostIdx = 0;
  SmallVector<unsigned, 6> permMap;
  for (unsigned int i = 0, e = oldLoopNames.size(); i < e; ++i) {
    auto name = oldLoopNames[i];
    auto iterator = std::find(nameOfAllLoopsWithNewOrder.begin(),
                              nameOfAllLoopsWithNewOrder.end(), name);
    unsigned int idx = iterator - nameOfAllLoopsWithNewOrder.begin();
    permMap.push_back(idx);
    if (idx == 0) {
      outerMostIdx = i;
    }
  }

  // 6) Permute the loops
  // TODO: imperfect loops
  // Permute if the nest's size is consistent with the specified
  // permutation
  if (nest.size() >= 2 && nest.size() == permMap.size()) {
    if (outerMostIdx != 0)
      nest[0]->removeAttr("op_name");
    permuteLoops(nest, permMap);
  } else {
    reorderOp.emitError("Cannot permute the loops because the size of the "
                        "perfectly nested loop band (")
        << nest.size() << ") "
        << "is not consistent with the size of permutation mapping ("
        << permMap.size() << ")";
    return failure();
  }

  // 7) Rename the stage if the outermost loop moves inward
  if (outerMostIdx != 0) {
    nest[outerMostIdx]->setAttr(
        "op_name", StringAttr::get(nest[outerMostIdx]->getContext(), op_name));
  }

  return success();
}

LogicalResult runIntraKernelOpCheck(func::FuncOp &f, IntraKernelToOp &intraOp) {
  // 1) Get the schedule
  auto loopHandle =
      dyn_cast<CreateLoopHandleOp>(intraOp.getPeArray().getDefiningOp());
  auto opHandle =
      dyn_cast<CreateOpHandleOp>(loopHandle.getOp().getDefiningOp());
  const auto loop_name = loopHandle.getLoopName();
  const auto op_name = opHandle.getOpName();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 3) Find the requested loop
  bool isOuterMost = false;
  int dependency_distance = 0;
  AffineLoopBand band;
  rootForOp->walk([&](AffineForOp forOp) {
    if (band.size() == 0 && loop_name == getLoopName(forOp)) {
      band.push_back(forOp);
      if (forOp->hasAttr("op_name"))
        isOuterMost = true;
      if (forOp->hasAttr("dep_distance")) {
        dependency_distance =
            llvm::dyn_cast<IntegerAttr>(forOp->getAttr("dep_distance"))
                .getInt();
      }
    }
  });

  // 4) Get data movement direction as int array
  Attribute attr = intraOp->getAttr("pe_index");
  auto pe_index = llvm::dyn_cast<ArrayAttr>(attr).getValue();

  // 5) Verify the intra-kernel data movement schedule
  if (pe_index.size() == 0) {
    intraOp.emitError("Cannot move data to null PE");
    return failure();
  }
  if (dependency_distance != 0) {
    intraOp.emitError("Cannot move the loop with non-uniform dependency");
    return failure();
  }

  return success();
}

LogicalResult runUnfolding(func::FuncOp &f, UnfoldOp &unfoldOp) {
  // 1) Get the schedule
  auto optional_factor = unfoldOp.getFactor();
  unsigned int factor;

  if (optional_factor.has_value()) {
    factor = optional_factor.value();
  } else {
    factor = 0; // fully unroll
  }

  auto loopHandle =
      dyn_cast<CreateLoopHandleOp>(unfoldOp.getLoop().getDefiningOp());
  const auto loop_name = loopHandle.getLoopName();
  const auto op_name =
      dyn_cast<CreateOpHandleOp>(loopHandle.getOp().getDefiningOp())
          .getOpName();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 3) Find the requested loop and attach attribute
  WalkResult result = rootForOp.walk([&](AffineForOp forOp) -> WalkResult {
    if (loop_name == getLoopName(forOp)) {
      AffineLoopBand band{forOp};
      SmallVector<int, 6> attr_arr{(int)factor};
      setIntAttr(band, attr_arr, "unroll");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  // TODO: call Polymer/PoCC to inject loop information as attr
  // handle exception
  if (!result.wasInterrupted()) {
    unfoldOp.emitError("Cannot find Loop ") << loop_name.str();
    return failure();
  }

  return failure();
}

LogicalResult runUnrolling(func::FuncOp &f, UnrollOp &unrollOp) {
  // 1) Get the schedule
  auto optional_factor = unrollOp.getFactor();
  unsigned int factor;
  if (optional_factor.has_value()) {
    factor = optional_factor.value();
  } else {
    factor = 0; // fully unroll
  }
  auto loopHandle =
      dyn_cast<CreateLoopHandleOp>(unrollOp.getLoop().getDefiningOp());
  const auto loop_name = loopHandle.getLoopName();
  const auto op_name =
      dyn_cast<CreateOpHandleOp>(loopHandle.getOp().getDefiningOp())
          .getOpName();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 3) Find the requested loop and attach attribute
  WalkResult result = rootForOp.walk([&](AffineForOp forOp) -> WalkResult {
    if (loop_name == getLoopName(forOp)) {
      AffineLoopBand band{forOp};
      SmallVector<int, 6> attr_arr{(int)factor};
      setIntAttr(band, attr_arr, "unroll");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  // handle exception
  if (!result.wasInterrupted()) {
    unrollOp.emitError("Cannot find Loop ") << loop_name.str();
    return failure();
  }

  return success();
}

LogicalResult runParallel(func::FuncOp &f, ParallelOp &parallelOp) {
  // 1) Get the schedule
  auto loopHandle =
      dyn_cast<CreateLoopHandleOp>(parallelOp.getLoop().getDefiningOp());
  const auto loop_name = loopHandle.getLoopName();
  const auto op_name =
      dyn_cast<CreateOpHandleOp>(loopHandle.getOp().getDefiningOp())
          .getOpName();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 3) Find the requested loop and attach attribute
  WalkResult result = rootForOp.walk([&](AffineForOp forOp) -> WalkResult {
    if (loop_name == getLoopName(forOp)) {
      AffineLoopBand band{forOp};
      SmallVector<int, 6> attr_arr{1};
      setIntAttr(band, attr_arr, "parallel");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  // handle exception
  if (!result.wasInterrupted()) {
    parallelOp.emitError("Cannot find Loop ") << loop_name.str();
    return failure();
  }

  return success();
}

LogicalResult runPipelining(func::FuncOp &f, PipelineOp &pipelineOp) {
  // 1) Get the schedule
  auto optional_ii = pipelineOp.getIi();
  unsigned int ii;
  if (optional_ii.has_value()) {
    ii = optional_ii.value();
  } else {
    ii = 1;
  }
  auto loopHandle =
      dyn_cast<CreateLoopHandleOp>(pipelineOp.getLoop().getDefiningOp());
  const auto loop_name = loopHandle.getLoopName();
  const auto op_name =
      dyn_cast<CreateOpHandleOp>(loopHandle.getOp().getDefiningOp())
          .getOpName();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 3) Find the requested loop and attach attribute
  WalkResult result = rootForOp.walk([&](AffineForOp forOp) -> WalkResult {
    if (loop_name == getLoopName(forOp)) {
      AffineLoopBand band{forOp};
      SmallVector<int, 6> attr_arr{(int)ii};
      setIntAttr(band, attr_arr, "pipeline_ii");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  // handle exception
  if (!result.wasInterrupted()) {
    pipelineOp.emitError("Cannot find Loop ") << loop_name.str();
    return failure();
  }
  return success();
}

LogicalResult runThreadBind(func::FuncOp &f, ThreadBindOp &threadBindOp) {
  // 1) Get the schedule
  auto target_dim = threadBindOp.getDim();
  auto loopHandle =
      dyn_cast<CreateLoopHandleOp>(threadBindOp.getLoop().getDefiningOp());
  const auto loop_name = loopHandle.getLoopName();
  const auto op_name =
      dyn_cast<CreateOpHandleOp>(loopHandle.getOp().getDefiningOp())
          .getOpName();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 3) Find the requested loop and attach attribute
  WalkResult result = rootForOp.walk([&](AffineForOp forOp) -> WalkResult {
    if (loop_name == getLoopName(forOp)) {
      AffineLoopBand band{forOp};
      SmallVector<int, 6> attr_arr{(int)target_dim};
      setIntAttr(band, attr_arr, "thread_axis");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  // handle exception
  if (!result.wasInterrupted()) {
    threadBindOp.emitError("Cannot find Loop ") << loop_name.str();
    return failure();
  }
  return success();
}

// modified from lib/Transforms/Utils/LoopUtils.cpp
LogicalResult coalesceLoops(MutableArrayRef<AffineForOp> loops,
                            AffineForOp stageLoop) {
  if (loops.size() < 2)
    return failure();

  AffineForOp innermost = loops.back();
  AffineForOp outermost = loops.front();
  AffineBound ub = outermost.getUpperBound();
  Location loc = outermost.getLoc();
  OpBuilder builder(outermost);
  for (AffineForOp loop : loops) {
    // We only work on normalized loops.
    if (loop.getStep() != 1 || !loop.hasConstantLowerBound() ||
        loop.getConstantLowerBound() != 0)
      return failure();
    // TODO: support AffineMap loop bounds
    if (!loop.hasConstantUpperBound())
      return failure();
  }
  SmallVector<Value, 4> upperBoundSymbols;
  SmallVector<Value, 4> ubOperands(ub.getOperands().begin(),
                                   ub.getOperands().end());

  // 1. Store the upper bound of the outermost loop in a variable.
  // 2. Emit code computing the upper bound of the coalesced loop as product of
  // the number of iterations of all loops.
  int64_t prod = 1;
  for (AffineForOp loop : loops) {
    auto cstUb = loop.getConstantUpperBound();
    prod *= cstUb;
    auto cstOp = builder.create<arith::ConstantIndexOp>(loc, cstUb);
    upperBoundSymbols.push_back(cstOp);
    // hoist to the outermost
    cstOp->moveBefore(stageLoop);
  }
  outermost.setConstantUpperBound(prod);

  builder.setInsertionPointToStart(outermost.getBody());

  // 3. Remap induction variables. For each original loop, the value of the
  // induction variable can be obtained by dividing the induction variable of
  // the linearized loop by the total number of iterations of the loops nested
  // in it modulo the number of iterations in this loop (remove the values
  // related to the outer loops):
  //   iv_i = floordiv(iv_linear, product-of-loop-ranges-until-i) mod range_i.
  // Compute these iteratively from the innermost loop by creating a "running
  // quotient" of division by the range.
  Value previous = outermost.getInductionVar();
  SmallVector<Operation *> opToSink;
  for (unsigned idx = loops.size(); idx > 0; --idx) {
    if (idx != loops.size()) {
      SmallVector<Value, 4> operands;
      operands.push_back(previous);
      operands.push_back(upperBoundSymbols[idx]);
      previous = builder.create<AffineApplyOp>(
          loc,
          AffineMap::get(
              /*numDims=*/1, /*numSymbols=*/1,
              builder.getAffineDimExpr(0).floorDiv(
                  builder.getAffineSymbolExpr(0))),
          operands);
      opToSink.push_back(previous.getDefiningOp());
    }
    // Modified value of the induction variables of the nested loops after
    // coalescing.
    Value inductionVariable;
    if (idx == 1) {
      inductionVariable = previous;
    } else {
      SmallVector<Value, 4> applyOperands;
      applyOperands.push_back(previous);
      applyOperands.push_back(upperBoundSymbols[idx - 1]);
      inductionVariable = builder.create<AffineApplyOp>(
          loc,
          AffineMap::get(
              /*numDims=*/1, /*numSymbols=*/1,
              builder.getAffineDimExpr(0) % builder.getAffineSymbolExpr(0)),
          applyOperands);
      opToSink.push_back(inductionVariable.getDefiningOp());
    }
    replaceAllUsesInRegionWith(loops[idx - 1].getInductionVar(),
                               inductionVariable, loops.back().getRegion());
  }

  // 4. Move the operations from the innermost just above the second-outermost
  // loop, delete the extra terminator and the second-outermost loop.
  AffineForOp secondOutermostLoop = loops[1];
  innermost.getBody()->back().erase();
  outermost.getBody()->getOperations().splice(
      Block::iterator(secondOutermostLoop.getOperation()),
      innermost.getBody()->getOperations());
  secondOutermostLoop.erase();

  // 5. Sink AffineApply operations
  std::reverse(opToSink.begin(), opToSink.end());
  loops[0]->walk([&](AffineForOp forOp) -> WalkResult { // from the innermost
    bool isDominance = true;
    for (auto applyOp : opToSink) {
      applyOp->moveBefore(&(*forOp.getBody()->getOperations().begin()));
      // definition should come before reference
      for (auto user : applyOp->getUsers()) {
        DominanceInfo domInfo;
        if (!domInfo.properlyDominates(applyOp->getResult(0), user)) {
          isDominance = false;
          break;
        }
      }
    }
    if (isDominance)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return success();
}

// Notice allo.fuse (fuses nested loops) is different from affine.fuse,
// which fuses contiguous loops. This is actually the case of allo.compute_at.
LogicalResult runFusing(func::FuncOp &f, FuseOp &fuseOp) {
  // 1) Get the schedule
  const auto loopsToFuse = fuseOp.getLoops(); // operand_range
  unsigned int sizeOfFusedLoops = loopsToFuse.size();
  if (sizeOfFusedLoops < 2) {
    fuseOp.emitError("Should at least input 2 loops to be fused");
    return failure();
  }
  auto opHandle = dyn_cast<CreateLoopHandleOp>(loopsToFuse[0].getDefiningOp());
  const auto op_name =
      dyn_cast<CreateOpHandleOp>(opHandle.getOp().getDefiningOp()).getOpName();
  SmallVector<StringRef, 6> nameArr;
  for (auto loop : loopsToFuse) {
    nameArr.push_back(
        dyn_cast<CreateLoopHandleOp>(loop.getDefiningOp()).getLoopName());
  }

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 3) Find the requested loops
  bool isOuterMost = false;
  AffineLoopBand band;
  WalkResult result = rootForOp.walk([&](AffineForOp forOp) -> WalkResult {
    if (findContiguousNestedLoops(forOp, band, nameArr))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  // handle exception
  if (!result.wasInterrupted()) {
    fuseOp.emitError("Cannot find contiguous nested loops starting from Loop ")
        << nameArr[0].str()
        << ". Please specify the loop to be fused from outermost to innermost.";
    return failure();
  }
  if (band[0]->hasAttr("op_name"))
    isOuterMost = true;

  // 4) Construct new loop
  MutableArrayRef<AffineForOp> fusedLoops =
      llvm::MutableArrayRef(band.data(), sizeOfFusedLoops);
  if (failed(coalesceLoops(fusedLoops, rootForOp)))
    return failure();
  if (isOuterMost)
    rootForOp = fusedLoops[0];

  // 5) Constant propagation into the affine map
  SmallVector<Operation *> opToRemove;
  rootForOp.walk([&](AffineApplyOp applyOp) {
    auto applyMap = applyOp.getAffineMap();
    if (applyMap.getNumSymbols() == 0)
      return;
    if (auto cst = dyn_cast<arith::ConstantOp>(
            applyOp.getOperand(1).getDefiningOp())) { // get symbolic operand
      int cstVal = llvm::dyn_cast<IntegerAttr>(cst.getValue()).getInt();
      auto builder = OpBuilder(applyOp);
      SmallVector<AffineExpr> newDims{builder.getAffineDimExpr(0)};
      SmallVector<AffineExpr> newSymbols{builder.getAffineConstantExpr(cstVal)};
      auto newMap = applyMap.replaceDimsAndSymbols(newDims, newSymbols, 1, 0);
      auto newApplyOp = builder.create<AffineApplyOp>(
          applyOp.getLoc(), newMap, llvm::ArrayRef(applyOp.getOperand(0)));
      applyOp.getResult().replaceAllUsesWith(newApplyOp);
      opToRemove.push_back(applyOp);
    }
  });
  for (Operation *op : opToRemove) {
    op->erase();
  }

  // 6) Add name to the new loop
  std::string new_name;
  for (auto name : nameArr) {
    new_name += name.str() + "_";
  }
  new_name += "fused";
  setLoopName(fusedLoops[0], new_name);
  if (isOuterMost)
    setStageName(fusedLoops[0], op_name);

  // 7) Create new loop handles &
  //    Link the loop handles with SSA values
  auto firstOp = *(f.getOps<AffineForOp>().begin());
  OpBuilder builder(firstOp);
  auto fused = builder.create<CreateLoopHandleOp>(
      firstOp->getLoc(), LoopHandleType::get(firstOp->getContext()),
      opHandle.getResult(), StringAttr::get(firstOp->getContext(), new_name));
  fuseOp.getResult().replaceAllUsesWith(fused);

  return success();
}

LogicalResult runComputeAt(func::FuncOp &f, ComputeAtOp &computeAtOp) {
  // 1) Get the schedule
  const auto loop_name =
      dyn_cast<CreateLoopHandleOp>(computeAtOp.getAxis().getDefiningOp())
          .getLoopName();
  const auto producer_name =
      dyn_cast<CreateOpHandleOp>(computeAtOp.getProducer().getDefiningOp())
          .getOpName();
  const auto consumer_name =
      dyn_cast<CreateOpHandleOp>(computeAtOp.getConsumer().getDefiningOp())
          .getOpName();

  // 2) Traverse all the outer-most loops and find the requested one
  AffineForOp producerFor;
  AffineForOp consumerFor;
  std::pair<bool, bool> isFound{false, false};
  for (auto rootForOp : f.getOps<AffineForOp>()) {
    auto curr_name =
        llvm::dyn_cast<StringAttr>(rootForOp->getAttr("op_name")).getValue();
    if (producer_name == curr_name) {
      producerFor = rootForOp;
      isFound.first = true;
    } else if (consumer_name == curr_name) {
      consumerFor = rootForOp;
      isFound.second = true;
    }
  }
  if (!isFound.first || !isFound.second) {
    computeAtOp.emitError("Cannot find corresponding producer and consumer");
    return failure();
  }

  // 3) Find the requested loops
  int cnt_depth = 0;
  int requested_depth = 0;
  SmallVector<Value> consumerIVs;
  SmallVector<Value> producerIVs;
  consumerFor.walk([&](AffineForOp forOp) {
    cnt_depth++;
    if (!forOp->hasAttr("loop_name"))
      return WalkResult::advance();
    Attribute attr = forOp->getAttr("loop_name");
    if (loop_name == llvm::dyn_cast<StringAttr>(attr).getValue()) {
      requested_depth = cnt_depth;
    }
    consumerIVs.push_back(forOp.getInductionVar());
    return WalkResult::advance();
  });
  producerFor.walk([&](AffineForOp forOp) {
    producerIVs.push_back(forOp.getInductionVar());
  });
  std::reverse(consumerIVs.begin(), consumerIVs.end());
  std::reverse(producerIVs.begin(), producerIVs.end());
  requested_depth = cnt_depth - requested_depth + 1;

  // find loop bounds
  SmallVector<int64_t, 4> consumerLBs;
  SmallVector<int64_t, 4> consumerUBs;
  SmallVector<int64_t, 4> producerLBs;
  SmallVector<int64_t, 4> producerUBs;
  consumerFor.walk([&](AffineForOp forOp) {
    auto lbMap = forOp.getLowerBoundMap();
    auto ubMap = forOp.getUpperBoundMap();
    int64_t lb = lbMap.getConstantResults()[0];
    int64_t ub = ubMap.getConstantResults()[0];
    consumerLBs.push_back(lb);
    consumerUBs.push_back(ub);
    return WalkResult::advance();
  });
  producerFor.walk([&](AffineForOp forOp) {
    auto lbMap = forOp.getLowerBoundMap();
    auto ubMap = forOp.getUpperBoundMap();
    int64_t lb = lbMap.getConstantResults()[0];
    int64_t ub = ubMap.getConstantResults()[0];
    producerLBs.push_back(lb);
    producerUBs.push_back(ub);
  });
  SmallVector<int64_t, 4> LBdiff;
  SmallVector<int64_t, 4> UBdiff;
  int64_t all_diff = 0;
  for (unsigned i = 0; i < std::min(producerLBs.size(), consumerLBs.size());
       i++) {
    LBdiff.push_back(producerLBs[i] - consumerLBs[i]);
    UBdiff.push_back(producerUBs[i] - consumerUBs[i]);
    all_diff += LBdiff[i] + UBdiff[i];
  }

  // 4) Try to merge two loops
  // TODO: bug: 1) cannot support tensor type
  //            2) doesn't support memref.load, memref.store
  SmallVector<Dependency, 4> dependency;
  if (!analyzeDependency(producerFor, consumerFor, dependency)) {
    std::string err_msg =
        "Does not support compute_at of stage with if operation.";
    computeAtOp.emitWarning("analyzeDependency Failed: ") << err_msg;
  }

  if (dependency.size() > 0 && std::find(dependency.begin(), dependency.end(),
                                         Dependency::RAW) != dependency.end()) {
    FusionStrategy strategy(FusionStrategy::ProducerConsumer);
    // use existing MLIR pass
    ComputationSliceState sliceUnion;
    FusionResult result = canFuseLoops(producerFor, consumerFor,
                                       requested_depth, &sliceUnion, strategy);
    std::string err_msg;
    if (result.value == FusionResult::Success) {
      fuseLoops(producerFor, consumerFor, sliceUnion);
      producerFor.erase();
    } else if (result.value == FusionResult::FailPrecondition) {
      err_msg = "failed precondition for fusion (e.g. same block)";
    } else if (result.value == FusionResult::FailBlockDependence) {
      err_msg = "fusion would violate another dependence in block";
    } else if (result.value == FusionResult::FailFusionDependence) {
      err_msg = "fusion would reverse dependences between loops";
    } else if (result.value == FusionResult::FailComputationSlice) {
      err_msg = "unable to compute src loop computation slice";
    } else if (result.value == FusionResult::FailIncorrectSlice) {
      err_msg = "slice is computed, but it is incorrect";
    }
    if (result.value != FusionResult::Success) {
      computeAtOp.emitError("Cannot merge these two loops because ") << err_msg;
      return failure();
    }
    if (requested_depth < cnt_depth - 1)
      return success();
  } else if (dependency.size() == 0 && all_diff != 0) {
    // no dependency, just merge the loops
    // but the loop bounds are not the same
    // e.g. first loop bound is (10, 10), second loop bound is (12, 12)
    //     then the fused loop bound is (12, 12)
    //     we build if statements to guard the smaller loop's body operations
    // 4.1) Check producer and consumer loop bounds, which one has smaller
    // bound?
    bool LBconsistency = std::all_of(LBdiff.begin(), LBdiff.end(),
                                     [](int64_t x) { return x > 0; }) ||
                         std::all_of(LBdiff.begin(), LBdiff.end(),
                                     [](int64_t x) { return x <= 0; });
    bool UBconsistency = std::all_of(UBdiff.begin(), UBdiff.end(),
                                     [](int64_t x) { return x > 0; }) ||
                         std::all_of(UBdiff.begin(), UBdiff.end(),
                                     [](int64_t x) { return x <= 0; });
    if (!LBconsistency || !UBconsistency) {
      computeAtOp.emitError(
          "Cannot merge these two loops because one's loop bounds are not "
          "consistently larger or smaller than the other's loop bounds");
      return failure();
    }
    bool is_producer_smaller = true;
    for (int i = 0; i < requested_depth; i++) {
      if (producerUBs[i] - producerLBs[i] < consumerUBs[i] - consumerLBs[i]) {
        is_producer_smaller = true && is_producer_smaller;
      } else {
        is_producer_smaller = false;
      }
    }
    // 4.2) Build guard conditions
    SmallVector<AffineExpr> constraints;
    SmallVector<bool> eqFlags;
    SmallVector<Value, 4> setOperands;
    for (int depth_idx = 0; depth_idx < requested_depth; depth_idx++) {
      OpBuilder builder(consumerFor);
      auto producerLB = builder.getAffineConstantExpr(producerLBs[depth_idx]);
      auto producerUB =
          builder.getAffineConstantExpr(producerUBs[depth_idx] - 1);
      auto consumerLB = builder.getAffineConstantExpr(consumerLBs[depth_idx]);
      auto consumerUB =
          builder.getAffineConstantExpr(consumerUBs[depth_idx] - 1);
      if (is_producer_smaller) {
        constraints.push_back(builder.getAffineDimExpr(depth_idx) - producerLB);
        constraints.push_back(producerUB - builder.getAffineDimExpr(depth_idx));
        eqFlags.push_back(false);
        eqFlags.push_back(false);
      } else {
        constraints.push_back(builder.getAffineDimExpr(depth_idx) - consumerLB);
        constraints.push_back(consumerUB - builder.getAffineDimExpr(depth_idx));
        eqFlags.push_back(false);
        eqFlags.push_back(false);
      }
      setOperands.push_back(consumerIVs[depth_idx]);
    }

    // 4.3) Build if statement
    AffineForOp secondForOp = consumerFor;
    getLoop(secondForOp, loop_name);
    int curr_depth = 0;
    AffineForOp firstForOp;
    producerFor.walk([&](AffineForOp forOp) {
      if (curr_depth++ == cnt_depth - requested_depth) {
        firstForOp = forOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    auto &firstBody = firstForOp.getBody()->getOperations();
    auto &secondBody = secondForOp.getBody()->getOperations();

    OpBuilder builder(secondForOp);
    if (is_producer_smaller) {
      builder.setInsertionPoint(&secondBody.front());
    } else {
      builder.setInsertionPoint(&firstBody.back());
    }
    auto ifCondSet = IntegerSet::get(requested_depth /*dimCount*/,
                                     0 /*symbolCount*/, constraints, eqFlags);
    auto ifOp = builder.create<AffineIfOp>(
        consumerFor.getLoc(), ifCondSet, setOperands, /*withElseRegion*/ false);
    auto &ifThenBody = ifOp.getThenBlock()->getOperations();

    // 4.4) Build fused loop
    if (is_producer_smaller) {
      ifThenBody.splice(ifThenBody.begin(), firstBody, firstBody.begin(),
                        std::prev(firstBody.end()));
      for (int i = 0; i < requested_depth; ++i)
        producerIVs[i].replaceAllUsesWith(consumerIVs[i]);
      producerFor.erase();
    } else {
      ifThenBody.splice(ifThenBody.begin(), secondBody, secondBody.begin(),
                        std::prev(secondBody.end()));
      for (int i = 0; i < requested_depth; ++i)
        consumerIVs[i].replaceAllUsesWith(producerIVs[i]);
      // move producerFor to the end of consumerFor
      producerFor->moveAfter(consumerFor.getOperation());
      consumerFor.erase();
    }
    return success();

  } else {
    // strategy = FusionStrategy::Sibling;
    computeAtOp.emitWarning(
        "MLIR loop fusion pass failed. Attempt using Allo's loop fusion pass.");
    // get inner loops
    AffineForOp secondForOp = consumerFor;
    getLoop(secondForOp, loop_name);
    int curr_depth = 0;
    AffineForOp firstForOp;
    producerFor.walk([&](AffineForOp forOp) {
      if (curr_depth++ == cnt_depth - requested_depth) {
        firstForOp = forOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    auto &firstBody = firstForOp.getBody()->getOperations();
    auto &secondBody = secondForOp.getBody()->getOperations();
    // do not need affine.yield op, so that's why using std::prev
    secondBody.splice(secondBody.begin(), firstBody, firstBody.begin(),
                      std::prev(firstBody.end()));
    // update references
    for (int i = 0; i < requested_depth; ++i)
      producerIVs[i].replaceAllUsesWith(consumerIVs[i]);
    producerFor.erase();
    return success();
  }

  // 5) remove intermediate buffers & loads/stores
  SmallVector<Operation *, 10> opToRemove;
  memref::AllocOp alloc;
  AffineStoreOp targetStore;
  consumerFor.walk([&](AffineStoreOp store) {
    if (!store.getOperand(1).getDefiningOp())
      return WalkResult::advance();
    auto buf = dyn_cast<memref::AllocOp>(store.getOperand(1).getDefiningOp());
    if (buf->hasAttr("name") &&
        llvm::dyn_cast<StringAttr>(buf->getAttr("name")).getValue().str() ==
            producer_name) {
      alloc = buf;
      targetStore = store;
      opToRemove.push_back(store);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  consumerFor.walk([&](AffineLoadOp load) {
    if (load->hasAttr("from") &&
        llvm::dyn_cast<StringAttr>(load->getAttr("from")).getValue().str() ==
            producer_name) {
      // Check if load and targetStore are in the same block
      if (load->getBlock() == targetStore->getBlock()) {
        replaceAllUsesInRegionWith(load.getResult(), targetStore.getOperand(0),
                                   consumerFor.getRegion());
        opToRemove.push_back(load);
        return WalkResult::interrupt();
      } else { // load and targetStore are in different blocks
        if (opToRemove.size() > 0) {
          opToRemove.pop_back(); // remove targetStore
        }
        return WalkResult::advance();
      }
    }
    return WalkResult::advance();
  });
  if (alloc && alloc.getResult().use_empty()) {
    opToRemove.push_back(alloc);
  }
  for (Operation *op : opToRemove) {
    op->erase();
  }
  return success();
}

bool findArray(func::FuncOp &f, const Value &target, Value &ret_array) {
  if (!target.getDefiningOp()) { // in func args
    for (auto arg : f.getArguments()) {
      if (target == arg) { // found the corresponding array
        ret_array = arg;
        return true;
      }
    }
    return false;
  } else {
    ret_array = target;
    return true;
  }
}

// https://github.com/hanchenye/scalehls/blob/master/lib/Transforms/Directive/ArrayPartition.cpp
LogicalResult runPartition(func::FuncOp &f, PartitionOp &partitionOp,
                           Value &array) {
  // 1) Get the schedule
  // auto memref = partitionOp.getTarget(); // return a Value type
  auto kind = partitionOp.getPartitionKind();
  unsigned int target_dim = partitionOp.getDim();
  auto optional_factor = partitionOp.getFactor();
  int factor = 0;
  if (optional_factor.has_value()) {
    factor = optional_factor.value();
  } else {
    factor = -1;
    if (kind != PartitionKindEnum::CompletePartition) {
      partitionOp.emitError("Should pass in `factor' for array partition");
      return failure();
    }
  }

  // 2) Find the requested array
  // has been done in findArray

  // 3) Construct new memory layout map
  auto builder = Builder(array.getContext());
  auto arrayType = llvm::dyn_cast<MemRefType>(array.getType());
  auto layout = arrayType.getLayout().getAffineMap();

  // Walk through each dimension of the current memory
  SmallVector<AffineExpr, 4> partitionIndices;
  SmallVector<AffineExpr, 4> addressIndices;

  // first N: partition index
  // last N : physical index
  unsigned rank = arrayType.getRank();
  if (layout.getNumResults() != rank) {
    partitionOp.emitWarning("Partition on the array partitioned before. "
                            "The original layout map will be rewritten!");
  }
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (target_dim == 0 || (target_dim > 0 && dim == target_dim - 1)) {
      if (kind == PartitionKindEnum::CyclicPartition) {
        // original index:  0, 1, 2, 3
        // bank (factor 2): 0, 1, 0, 1
        partitionIndices.push_back(builder.getAffineDimExpr(dim) % factor);
        addressIndices.push_back(
            builder.getAffineDimExpr(dim).floorDiv(factor));
      } else if (kind == PartitionKindEnum::BlockPartition) {
        // * block factor N means partition into N blocks
        //   each block has shape[dim] / factor elements
        //   (not N elements in each block!)
        // original index:  0, 1, 2, 3
        // bank (factor 2): 0, 0, 1, 1
        auto blockFactor =
            (arrayType.getShape()[dim] + factor - 1) / factor; // ceil
        partitionIndices.push_back(
            builder.getAffineDimExpr(dim).floorDiv(blockFactor));
        addressIndices.push_back(builder.getAffineDimExpr(dim) % blockFactor);
      } else if (kind == PartitionKindEnum::CompletePartition) {
        // original index:  0, 1, 2, 3
        // bank (factor 2): 0, 1, 2, 3
        partitionIndices.push_back(builder.getAffineDimExpr(dim));
        addressIndices.push_back(builder.getAffineConstantExpr(0));
      } else {
        partitionOp.emitError("No this partition kind");
        return failure();
      }
    } else {
      if (layout.getNumResults() == rank) {
        partitionIndices.push_back(builder.getAffineConstantExpr(0));
        addressIndices.push_back(builder.getAffineDimExpr(dim));
      } else { // already had one layout map before
        partitionIndices.push_back(layout.getResult(dim));
        addressIndices.push_back(layout.getResult(dim + rank));
      }
    }
  }

  // Construct new layout map
  partitionIndices.append(addressIndices.begin(), addressIndices.end());
  auto layoutMap = AffineMap::get(arrayType.getRank(), 0, partitionIndices,
                                  builder.getContext());

  // Construct new array type
  auto newType =
      MemRefType::get(arrayType.getShape(), arrayType.getElementType(),
                      layoutMap, arrayType.getMemorySpace());

  // Set new type
  array.setType(newType);

  // 4) update function signature
  auto resultTypes = f.front().getTerminator()->getOperandTypes();
  auto inputTypes = f.front().getArgumentTypes();
  f.setType(builder.getFunctionType(inputTypes, resultTypes));

  return success();
}

LogicalResult runReplaceOp(func::FuncOp &f, ReplaceOp &replaceOp, Value &src,
                           Value &dst) {
  // 1) Get the schedule

  // 2) Find the requested array
  // has been done in findArray

  // 3) Replace all uses of src with dst
  src.replaceAllUsesWith(dst);
  src.getDefiningOp()->erase();

  return success();
}

LogicalResult runReuseAt(func::FuncOp &f, ReuseAtOp &reuseAtOp) {
  // 1) Get the schedule
  // target is the target tensor to reuse
  auto target = reuseAtOp.getTarget(); // return a Value type
  // the loop at which level the target tensor is reused
  auto loopHandle =
      dyn_cast<CreateLoopHandleOp>(reuseAtOp.getAxis().getDefiningOp());
  const auto loop_name = loopHandle.getLoopName();
  const auto op_name =
      dyn_cast<CreateOpHandleOp>(loopHandle.getOp().getDefiningOp())
          .getOpName();
  auto arrayType = llvm::dyn_cast<MemRefType>(target.getType());
  unsigned int rank = arrayType.getRank();

  // Determine whether the target array is unsigned
  bool is_unsigned = false;
  auto defOp = target.getDefiningOp();
  if (defOp) {
    if (defOp->hasAttr("unsigned"))
      is_unsigned = true;
  } else {
    // function argument
    if (f->hasAttr("itypes")) {
      auto top_itypes =
          llvm::dyn_cast<StringAttr>(f->getAttr("itypes")).getValue().str();
      int argIdx = 0;
      for (auto arg : f.getArguments()) {
        if (arg == target) {
          break;
        }
        argIdx++;
      }
      if (top_itypes[argIdx] == 'u')
        is_unsigned = true;
    }
  }

  // 2) Find the requested stage (loop nest)
  // the outermost loop of the target loop nest
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 3) Find the requested loop and get the axis id
  AffineForOp reuseLoop = rootForOp;
  int loopAxis = getLoop(reuseLoop, loop_name);
  if (loopAxis == -1) {
    f.emitError("Cannot find Loop ") << loop_name.str();
    return failure();
  }

  // 4) Find (non-)reduction loops
  // collect info about nonReductionLoops, spatial loops, and reduction loops
  AffineLoopBand nonReductionLoops;
  AffineLoopBand previousShiftLoops;
  // InductionVar -> Loop upper bound
  DenseMap<Value, int> reductionVars;
  WalkResult result = rootForOp.walk([&](AffineForOp forOp) {
    if (forOp.getStep() != 1 || !forOp.hasConstantLowerBound() ||
        forOp.getConstantLowerBound() != 0 || !forOp.hasConstantUpperBound()) {
      reuseAtOp.emitError("Loop ")
          << getLoopName(forOp).str()
          << " must have (1) constant bounds (2) constant step (3) zero "
             "lower bound";
      return WalkResult::interrupt();
    }
    if (!forOp->hasAttr("reduction") && !forOp->hasAttr("spatial") &&
        !forOp->hasAttr("buffer")) {
      nonReductionLoops.push_back(forOp);
    } else if (forOp->hasAttr("spatial")) {
      previousShiftLoops.push_back(forOp);
    } else if (forOp->hasAttr("reduction")) {
      reductionVars[forOp.getInductionVar()] = forOp.getConstantUpperBound();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  std::reverse(nonReductionLoops.begin(), nonReductionLoops.end());
  AffineForOp innerMostForOp = nonReductionLoops[nonReductionLoops.size() - 1];

  // 5) Get span of each dimension (also stride)
  //    e.g. d0, d0+1, d0+2, span is 2
  //         d0+d1, d1\in[0,2], span is 2
  SmallVector<SmallVector<AffineExpr>> originalLoadExprs;
  for (int i = 0; i < (int)rank; ++i) {
    SmallVector<AffineExpr> tmp;
    originalLoadExprs.push_back(tmp);
  }
  int cntLoad = 0;
  DenseMap<AffineExpr, Value> dim2iv; // dim -> induction var
  reuseLoop.walk([&](AffineLoadOp loadOp) {
    // Note: affine.load operation's affine map can have arbitrary operands
    // e.g. affine.load %v[%v0 + %v1, %v2 + %v3] has this affine map:
    // (d0, d1, d2, d3) -> (d0 + d1, d2 + d3)
    // the affine map operands are: %v0, %v1, %v2, %v3
    if (loadOp.getOperand(0) != target)
      return WalkResult::advance();
    cntLoad++;
    for (int i = 0; i < (int)rank; ++i) {
      // loadOp.getAffineMap().getResult(i) is the affine expression of the i-th
      // dimension. Such as d0+d1, d1+d2
      originalLoadExprs[i].push_back(loadOp.getAffineMap().getResult(i));
    }
    OpBuilder builder(loadOp);
    for (auto operandItem : llvm::enumerate(loadOp.getMapOperands())) {
      // operandItem.value() is the i-th affine map operand.
      dim2iv[builder.getAffineDimExpr(operandItem.index())] =
          operandItem.value();
    }
    return WalkResult::advance();
  });
  SmallVector<int> spans;
  int stride = 1;
  for (int i = 0; i < (int)rank; ++i) {
    int span = 0;
    // TODO: require strict load order
    AffineExpr baseExpr = originalLoadExprs[i][0];
    int baseCst = 0;
    if (llvm::isa<AffineDimExpr>(baseExpr)) {
      bool allAffineDimExpr = true;
      for (int j = 0; j < cntLoad; ++j) {
        auto diff = originalLoadExprs[i][j] - baseExpr;
        if (!llvm::isa<AffineDimExpr>(originalLoadExprs[i][j]))
          allAffineDimExpr = false;
        if (llvm::isa<AffineConstantExpr>(diff)) {
          span = std::max(
              span,
              (int)llvm::dyn_cast<AffineConstantExpr>(diff).getValue() + 1);
        } else {
          assert(1 == 0 && "Load order is not strict");
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
              span,
              (int)llvm::dyn_cast<AffineConstantExpr>(diff).getValue() + 1);
        } else {
          assert(1 == 0 && "Load order is not strict");
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
        int cst = llvm::dyn_cast<AffineConstantExpr>(rhs).getValue();
        if (baseCst == 0)
          baseCst = cst;
        span = std::max(span, cst - baseCst + 1);
      }
      if (llvm::isa<AffineBinaryOpExpr>(lhs)) {
        auto binLHS = llvm::dyn_cast<AffineBinaryOpExpr>(lhs);
        if (llvm::isa<AffineConstantExpr>(binLHS.getRHS()))
          stride =
              llvm::dyn_cast<AffineConstantExpr>(binLHS.getRHS()).getValue();
        else
          assert(1 == 0 && "Unsupported memref format");
      }
    }
    assert(span != 0 && "Span should not be 0");
    spans.push_back(span);
  }

  // 6) Obtain AffineMaps of load instructions
  // if i-th axis has reduction var before the reuse axis
  //  reductionLoopBound[i] should be the dimension size
  // if i-th axis has reduction var after the reuse axis
  //  target.shape[i] should be the dimension size
  std::set<AffineExpr, ExprCompare> requestedVars;
  SmallVector<AffineLoadOp> allLoadOps;
  std::map<int, int> dimBounds; // dim expr->reduction bound
  int axis = -1;
  int distance = -1;
  int numLoadOp = 0;
  // TODO: eliminate order in inputs
  reuseAtOp.emitWarning("Need to guarantee the loads have orders");
  reuseLoop.walk([&](AffineLoadOp loadOp) {
    if (loadOp.getOperand(0) != target)
      return WalkResult::advance();
    numLoadOp++;
    auto loadMap = loadOp.getAffineMap();
    int numDims = loadMap.getNumDims();
    auto operands = loadOp.getMapOperands();
    int rDim = -1;
    int operandIdx = 0;
    for (int j = 0; j < (int)loadMap.getNumResults(); ++j) {
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
            // if the loadOp's AffineMap operand at operandIdx
            // is the induction var of the reuse loop, then
            // the target reuse axis is the current j-th axis
            axis = j;
          }

          else if (!llvm::isa<BlockArgument>(operands[operandIdx]) &&
                   llvm::isa<AffineApplyOp>(
                       operands[operandIdx].getDefiningOp())) {
            // However, the reuse loop can be transformed
            // so the induction var may not directly be the
            // loadOp's AffineMap operand at operandIdx,
            // instead, it is a result of affine.apply
            auto applyOp = llvm::dyn_cast<AffineApplyOp>(
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
    assert(axis != -1);
    OpBuilder builder(loadOp);
    AffineExpr expr = loadMap.getResult(axis);
    auto insertLoadOp = [&](AffineLoadOp loadOp) {
      int size = allLoadOps.size();
      auto exp1 = loadOp.getAffineMap().getResult(axis);
      ExprCompare cmp;
      for (int i = 0; i < size; ++i) {
        int val1 = cmp.findConstantExpr(exp1);
        auto exp2 = allLoadOps[i].getAffineMap().getResult(axis);
        int val2 = cmp.findConstantExpr(exp2);
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
  assert(distance > -1);

  // 7) Try to find reuse pattern
  //    TODO: support more reuse patterns
  bool canReuse = false;
  auto baseVar = *(requestedVars.begin());
  for (auto var : requestedVars) {
    if (std::find(requestedVars.begin(), requestedVars.end(), var + 1) !=
        requestedVars.end()) {
      canReuse = true;
      break;
    }
  }
  if (!canReuse) {
    reuseAtOp.emitError("Cannot find reuse pattern on axis ")
        << std::to_string(loopAxis)
        << ". Only support stride 1 reuse pattern now";
    return failure();
  }

  // 8) Obtain indices and strides in load instructions
  SmallVector<AffineMap> allLoadAffineMaps;
  SmallVector<SmallVector<Value>> allLoadOperands;
  SmallVector<int> preRDim;
  SmallVector<int> preRDimAxis;
  int rDim = -1;
  AffineLoadOp originalLoadOp;
  bool resultFlag = true;
  for (auto loadOp : allLoadOps) {
    auto loadMap = loadOp.getAffineMap();
    // e.g. d0 d0+2, diff=2
    //      d0 d0+d1, diff=d1
    auto var = loadMap.getResult(axis);
    auto diff = var - baseVar;

    // find reduction dimension
    auto getReductionDim = [&](AffineExpr expr) {
      for (auto item : dimBounds)
        if (expr.isFunctionOfDim(item.first))
          return item.first;
      return -1;
    };
    rDim = getReductionDim(diff);

    // obtain load expressions
    OpBuilder builder(loadOp);
    if (rDim != -1) { // is reduction
      int ub = dimBounds[rDim];
      auto operands = loadOp.getMapOperands();
      originalLoadOp = loadOp;
      // expand the reduction axis
      for (int j = 0; j < ub; j++) {
        SmallVector<AffineExpr> singleLoadAffineExpr;
        SmallVector<Value> memAffineIndices;
        int loadRank = 0; // loadOp.getMapOperands().size();
        int operandIdx = 0;
        // TODO: better mapping machanism for high-dimensional tensors
        // i < axis
        for (int i = 0; i < axis; ++i) {
          auto expr = loadMap.getResult(i);
          // TODO: only suppose the expr is in the format of d0*c+d1
          int d = getReductionDim(expr);
          if (spans[i] > 1) {
            if (d != -1) {
              // reduction axis before reuse axis
              if (std::find(preRDim.begin(), preRDim.end(), d) ==
                  preRDim.end()) {
                preRDim.push_back(d);
                preRDimAxis.push_back(i);
              }
              singleLoadAffineExpr.push_back(
                  builder.getAffineDimExpr(loadRank++));
              operandIdx++;
              memAffineIndices.push_back(operands[operandIdx++]);
            } else { // AffineConstantExpr
              singleLoadAffineExpr.push_back(expr);
            }
          }
        }
        // i = axis
        // TODO: suppose the expr is d0*c+d1
        singleLoadAffineExpr.push_back(builder.getAffineConstantExpr(j));
        operandIdx++;
        // i > axis
        for (unsigned int i = axis + 1; i < rank; ++i) {
          auto expr = loadMap.getResult(i);
          if (llvm::isa<AffineBinaryOpExpr>(expr)) {
            singleLoadAffineExpr.push_back(
                builder.getAffineDimExpr(loadRank++));
            memAffineIndices.push_back(operands[operandIdx++]);
            operandIdx++;
          } else if (llvm::isa<AffineDimExpr>(expr)) {
            singleLoadAffineExpr.push_back(
                builder.getAffineDimExpr(loadRank++));
            memAffineIndices.push_back(operands[operandIdx++]);
          } else { // AffineConstantExpr
            singleLoadAffineExpr.push_back(expr);
          }
        }
        auto affineMap = AffineMap::get(
            loadRank /*rank*/, 0, singleLoadAffineExpr, builder.getContext());
        if (std::find(allLoadAffineMaps.begin(), allLoadAffineMaps.end(),
                      affineMap) == allLoadAffineMaps.end()) {
          allLoadAffineMaps.push_back(affineMap);
          allLoadOperands.push_back(memAffineIndices);
        }
      }
    } else {
      originalLoadOp = loadOp;
      int loadRank = 0;
      int operandIdx = 0;
      auto operands = loadOp.getMapOperands();
      SmallVector<Value> memAffineIndices;
      SmallVector<AffineExpr> singleLoadAffineExpr;
      // i < axis
      for (int i = 0; i < axis; ++i) {
        if (spans[i] > 1) {
          // placeholder
          singleLoadAffineExpr.push_back(builder.getAffineDimExpr(loadRank++));
          memAffineIndices.push_back(operands[operandIdx]);
        }
      }
      // i = axis
      if (llvm::isa<AffineConstantExpr>(diff)) {
        singleLoadAffineExpr.push_back(diff);
      } else {
        reuseAtOp.emitError("Cannot support non-constant stride");
        resultFlag = false;
        break;
      }
      // i > axis
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
  if (!resultFlag)
    return failure();

  // 9) Create reuse buffer
  //    e.g., %1 = memref.alloc() : memref<3xi32>
  SmallVector<int64_t> shape;
  // i < axis
  for (int i = 0; i < axis; ++i)
    if (spans[i] > 1)
      shape.push_back(spans[i]);
  // i = axis
  shape.push_back(distance + 1);
  // i > axis
  for (unsigned int i = axis + 1; i < rank; ++i)
    shape.push_back(arrayType.getShape()[i]);
  OpBuilder out_builder(rootForOp); // outside the stage
  auto buf = out_builder.create<memref::AllocOp>(
      rootForOp.getLoc(),
      MemRefType::get(
          shape,
          llvm::dyn_cast<MemRefType>(target.getType()).getElementType()));
  buf->setAttr("name", StringAttr::get(buf->getContext(),
                                       StringRef(op_name.str() + "_reuse_" +
                                                 std::to_string(loopAxis))));
  if (is_unsigned)
    buf->setAttr("unsigned", out_builder.getUnitAttr());

  // 10) link the result SSA with the buffer
  reuseAtOp.getResult().replaceAllUsesWith(buf);

  // 11) Update loop bound
  // TODO: support non-constant bound
  auto origLoopBound = nonReductionLoops[loopAxis].getConstantUpperBound();
  nonReductionLoops[loopAxis].setConstantUpperBound(origLoopBound * stride +
                                                    distance);
  // update the expressions using nonReductionLoops[loopAxis]'s induction var
  // new_iv = old_iv * stride + distance
  // => old_iv = (new_iv - distance) / stride
  // therefore, we need to replace all uses of the old_iv with (new_iv -
  // distance) / stride
  auto iv = nonReductionLoops[loopAxis].getInductionVar();
  // get all uses of the induction variable
  for (auto &use : iv.getUses()) {
    auto user = use.getOwner();
    // we expect a sequence of:
    // %iv = arith.index_cast %iv : index to some integer type
    // %stride = arith.constant stride : some integer type
    // %iv_stride = arith.muli %iv, %stride : some integer type
    // We need to replace the %iv_stride with
    // the newly created induction variable expression

    // if the user is an index_cast, we need to check the next user
    if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(user)) {
      for (auto &cast_user : indexCastOp.getResult().getUses()) {
        user = cast_user.getOwner();
        // if the user is a muli op, we have our target
        if (auto muliOp = dyn_cast<arith::MulIOp>(user)) {
          // bulid a new expression: iv - distance
          OpBuilder builder(muliOp);
          Type dtype = muliOp.getResult().getType();
          auto cstDistance = builder.create<arith::ConstantOp>(
              muliOp.getLoc(), dtype, builder.getIntegerAttr(dtype, distance));
          auto subIOp = builder.create<arith::SubIOp>(muliOp.getLoc(),
                                                      indexCastOp, cstDistance);
          // replace the muli op with the new induction variable expression
          muliOp.replaceAllUsesWith(subIOp.getResult());
          // remove the muli op
          muliOp.erase();
        }
      }
    }
  }

  // 12) Update store index, since some load/store will be created later, this
  // step is done in advance reduction case:
  //   skip the first store (to reduction variable)
  //     affine.store %0, %1[%c0] {to = "sum_rv"} : memref<1xi32>
  //   update the outer store
  //     affine.store %6, %3[%arg1, %arg2] : memref<10x8xi32>
  // non-reduction case:
  //   affine.store %9, %0[%arg1, %arg2] : memref<10x8xi32>
  // * index should be changed to [%arg1, %arg2 - 2]
  SmallVector<Operation *> opToRemove;
  reuseLoop.walk([&](AffineStoreOp op) {
    // skip reduction variable store
    auto arrayType = llvm::dyn_cast<MemRefType>(op.getOperand(1).getType());
    if (arrayType.getRank() == 1 && arrayType.getShape()[0] == 1) {
      return WalkResult::advance();
    }
    // update the store to output tensor
    OpBuilder rewriter(op);
    SmallVector<AffineExpr> memAffineIndices;
    auto oldAffineMap = op.getAffineMap();
    // we need to find the correct memref axis from loopAxis
    for (unsigned int i = 0, e = oldAffineMap.getResults().size(); i < e; ++i) {
      AffineExpr idx;
      Value targetIV = nonReductionLoops[loopAxis].getInductionVar();
      int targetAxis = findMemRefAxisFromIV(op, targetIV);
      if ((int)i == targetAxis)
        // the iteration space now is related to the input tensor
        if (stride != 1) {
          auto strideCst = rewriter.getAffineConstantExpr(stride);
          idx = (oldAffineMap.getResult(i) - distance).floorDiv(strideCst);
        } else {
          idx = oldAffineMap.getResult(i) - distance;
        }
      else
        idx = oldAffineMap.getResult(i);
      memAffineIndices.push_back(idx);
    }
    auto affineMap = AffineMap::get(arrayType.getRank() /*rank*/, 0,
                                    memAffineIndices, rewriter.getContext());
    rewriter.create<AffineStoreOp>(
        op->getLoc(), op.getOperand(0) /*valueToStore*/,
        op.getOperand(1) /*memref*/, affineMap, op.getIndices());
    opToRemove.push_back(op);
    return WalkResult::advance();
  });
  // also update `if` structure that uses this axis
  // e.g. #set1 = affine_set<(d0, d1, d2, d3) : (d0 + d1 >= 0,
  // -(d0 + d1) + 7 >= 0, d2 + d3 >= 0, -(d2 + d3) + 7 >= 0)>
  nonReductionLoops[loopAxis].walk([&](AffineIfOp ifOp) {
    int operandIdx = -1;
    for (auto item : llvm::enumerate(ifOp.getOperands())) {
      if (item.value() == nonReductionLoops[loopAxis].getInductionVar()) {
        operandIdx = item.index();
        break;
      }
    }
    // get the if condition
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
    auto newCondSet = IntegerSet::get(
        condSet.getNumDims() /*dimCount*/, 0 /*symbolCount*/,
        newConds /*ArrayRef<AffineExpr> constraints*/, condSet.getEqFlags());
    ifOp.setIntegerSet(newCondSet);
  });

  // 13) Rewrite original memref to load from buffer
  // reduction case:
  //   skip the first load (from reduction variable)
  //     %1 = affine.load %0[%c0] {from = "sum_rv"} : memref<1xi32>
  //   update the non-reduction load
  //     %7 = affine.load %arg0[%arg1, %arg2 + %arg3] : memref<10x10xi32>
  // * load should be changed to %buf[%arg3]
  // non-reduction case:
  //   %4 = affine.load %arg0[%arg1, %arg2 + 0,1,2] : memref<10x10xi32>
  // * load should be changed to %buf[0,1,2]
  // * buffer shifting will be done later
  for (auto op : allLoadOps) {
    OpBuilder rewriter(op);
    SmallVector<AffineExpr> loadAffineExpr;
    SmallVector<Value> memAffineIndices;
    SmallVector<Value> operands = op.getMapOperands();
    auto loadMap = op.getAffineMap();

    // obtain load expressions
    AffineLoadOp newLoad;
    if (rDim == -1) { // reuse the found rDim value
      auto diff = loadMap.getResult(axis) - baseVar;
      loadAffineExpr.push_back(diff);
      int loadRank = 0;
      int operandIdx = 0;
      // i < axis
      for (int i = 0; i < axis; ++i) {
        if (spans[i] > 1) {
          loadAffineExpr.push_back(loadMap.getResult(i));
        }
      }
      // i > axis
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
      auto affineMap = AffineMap::get(loadRank /*rank*/, 0, loadAffineExpr,
                                      rewriter.getContext());
      newLoad = rewriter.create<AffineLoadOp>(op->getLoc(), buf, affineMap,
                                              memAffineIndices);
    } else { // reduction
      int loadRank = 0;
      int operandIdx = 0;
      for (int i = 0; i < (int)rank; ++i) {
        auto expr = loadMap.getResult(i);
        // TODO: only suppose the expr is in the format of d0*c+d1, and d1 is
        // reduction axis
        if (i < axis) {
          if (spans[i] > 1) {
            if (llvm::isa<AffineBinaryOpExpr>(expr)) {
              loadAffineExpr.push_back(rewriter.getAffineDimExpr(loadRank++));
              operandIdx++;
            } else if (llvm::isa<AffineDimExpr>(expr)) {
              loadAffineExpr.push_back(rewriter.getAffineDimExpr(loadRank++));
            } else { // expr is a constant
              loadAffineExpr.push_back(expr);
            }
            memAffineIndices.push_back(operands[operandIdx++]);
          } else {
            // TODO: suppose no other reduction axis before `axis`
            if (!llvm::isa<AffineConstantExpr>(expr))
              operandIdx++;
          }
        } else if (i == axis) {
          loadAffineExpr.push_back(rewriter.getAffineDimExpr(loadRank++));
          if (llvm::isa<AffineBinaryOpExpr>(expr)) // put reduction dim
            operandIdx++;
          memAffineIndices.push_back(operands[operandIdx++]);
        } else { // i > axis
          if (llvm::isa<AffineBinaryOpExpr>(expr)) {
            auto dim0 = rewriter.getAffineDimExpr(loadRank++);
            auto dim1 = rewriter.getAffineDimExpr(loadRank++);
            if (stride != 1) {
              auto strideCst = rewriter.getAffineConstantExpr(stride);
              loadAffineExpr.push_back(dim0 * strideCst + dim1);
            } else {
              loadAffineExpr.push_back(dim0 + dim1);
            }
            memAffineIndices.push_back(operands[operandIdx++]);
            memAffineIndices.push_back(operands[operandIdx++]);
          } else if (llvm::isa<AffineDimExpr>(expr)) {
            loadAffineExpr.push_back(rewriter.getAffineDimExpr(loadRank++));
            memAffineIndices.push_back(operands[operandIdx++]);
          } else { // AffineConstantExpr
            loadAffineExpr.push_back(expr);
          }
        }
      }
      auto affineMap = AffineMap::get(loadRank /*rank*/, 0, loadAffineExpr,
                                      rewriter.getContext());
      newLoad = rewriter.create<AffineLoadOp>(op->getLoc(), buf, affineMap,
                                              memAffineIndices);
    }
    if (is_unsigned)
      newLoad->setAttr("unsigned", rewriter.getUnitAttr());
    op->replaceAllUsesWith(newLoad);
    opToRemove.push_back(op);
  }

  // 14) Create if structure
  //     only if the indices are inside the output tensor iteration space,
  //     results will be computed and written to output
  int cntIf = 0;
  nonReductionLoops[0].walk([&](AffineIfOp ifOp) { cntIf++; });
  nonReductionLoops[nonReductionLoops.size() - 1].walk(
      [&](AffineIfOp ifOp) { cntIf--; });
  AffineIfOp ifOp;
  if (!llvm::isa<AffineIfOp>(
          nonReductionLoops[loopAxis].getBody()->getOperations().front())) {
    OpBuilder builder(
        &(nonReductionLoops[loopAxis].getBody()->getOperations().front()));
    auto loc = nonReductionLoops[loopAxis]
                   .getBody()
                   ->getOperations()
                   .begin()
                   ->getLoc();
    // e.g. #set = affine_set<(d0, d1)[s0]: (d0 - 10 >= 0, s0 - d0 - 9 >= 0,
    //                                d1 - 10 >= 0, s0 - d1 - 9 >= 0)>
    SmallVector<AffineExpr> constraints{builder.getAffineDimExpr(0) - distance};
    SmallVector<bool> eqFlags{false};
    if (stride != 1) {
      auto strideCst = builder.getAffineConstantExpr(stride);
      auto distanceCst = builder.getAffineConstantExpr(distance);
      constraints.push_back((builder.getAffineDimExpr(0) - distanceCst) %
                            strideCst);
      eqFlags.push_back(true);
    }
    auto ifCondSet = IntegerSet::get(
        1 /*dimCount*/, 0 /*symbolCount*/,
        constraints /*ArrayRef<AffineExpr> constraints*/, eqFlags);
    SmallVector<Value, 4> setOperands{
        nonReductionLoops[loopAxis].getInductionVar()};
    ifOp = builder.create<AffineIfOp>(loc, ifCondSet, setOperands,
                                      /*withElseRegion=*/false);
    auto &innerMostBody =
        nonReductionLoops[loopAxis].getBody()->getOperations();
    auto &ifThenBody = ifOp.getThenBlock()->getOperations();
    ifThenBody.splice(ifThenBody.begin(), innerMostBody,
                      std::next(innerMostBody.begin()),
                      std::prev(innerMostBody.end()));
  } else {
    auto outerIfOp = llvm::cast<AffineIfOp>(
        innerMostForOp.getBody()->getOperations().front());
    // skip the first if statement
    OpBuilder builder(&(*(outerIfOp.getThenBlock()->getOperations().begin())));
    auto loc = outerIfOp.getThenBlock()->getOperations().begin()->getLoc();
    SmallVector<AffineExpr> constraints{builder.getAffineDimExpr(0) - distance};
    SmallVector<bool> eqFlags{false};
    if (stride != 1) {
      auto strideCst = builder.getAffineConstantExpr(stride);
      auto distanceCst = builder.getAffineConstantExpr(distance);
      constraints.push_back((builder.getAffineDimExpr(0) - distanceCst) %
                            strideCst);
      eqFlags.push_back(true);
    }
    auto ifCondSet = IntegerSet::get(
        1 /*dimCount*/, 0 /*symbolCount*/,
        constraints /*ArrayRef<AffineExpr> constraints*/, eqFlags);
    SmallVector<Value, 4> setOperands{
        nonReductionLoops[loopAxis].getInductionVar()};
    ifOp = builder.create<AffineIfOp>(loc, ifCondSet, setOperands,
                                      /*withElseRegion=*/false);
    auto &innerMostBody = outerIfOp.getThenBlock()->getOperations();
    auto &ifThenBody = ifOp.getThenBlock()->getOperations();
    ifThenBody.splice(ifThenBody.begin(), innerMostBody,
                      std::next(innerMostBody.begin()),
                      std::prev(innerMostBody.end()));
    ifOp = outerIfOp;
  }

  // 15) shift buffer elements & load from memory to buffer
  // reduction case:
  // non-reduction case:
  //   %2 = affine.load %1[1] : memref<3xi32>
  //   affine.store %2, %1[0] : memref<3xi32>
  //   %3 = affine.load %1[2] : memref<3xi32>
  //   affine.store %3, %1[1] : memref<3xi32>
  //   %4 = affine.load %arg0[%arg1, %arg2] : memref<10x10xi32>
  //   affine.store %4, %1[2] : memref<3xi32>
  OpBuilder builder(ifOp);
  Location loc = ifOp.getLoc();
  if (!llvm::isa<AffineIfOp>(ifOp.getThenBlock()->getOperations().front())) {
    loc = nonReductionLoops[loopAxis]
              .getBody()
              ->getOperations()
              .begin()
              ->getLoc();
    builder = OpBuilder(
        &(*(nonReductionLoops[loopAxis].getBody()->getOperations().begin())));
  } else {
    ifOp = llvm::cast<AffineIfOp>(
        innerMostForOp.getBody()->getOperations().front());
    loc = ifOp.getThenBlock()->getOperations().begin()->getLoc();
    builder = OpBuilder(&(*(ifOp.getThenBlock()->getOperations().begin())));
  }
  AffineLoopBand shiftForOps; // after reuse `axis`
  for (unsigned int i = loopAxis + 1; i < nonReductionLoops.size(); ++i) {
    auto ub = llvm::dyn_cast<MemRefType>(target.getType())
                  .getShape()[i - loopAxis + axis];
    shiftForOps.push_back(builder.create<AffineForOp>(loc, 0, ub));
    shiftForOps.back()->setAttr("spatial", builder.getUnitAttr());
    builder =
        OpBuilder(&(*(shiftForOps.back().getBody()->getOperations().begin())));
    loc = shiftForOps.back().getBody()->getOperations().begin()->getLoc();
  }
  AffineLoopBand reductionForOps; // before reuse `axis`
  for (int i = 0; i < axis; ++i) {
    if (spans[i] > 1) {
      reductionForOps.push_back(builder.create<AffineForOp>(loc, 0, spans[i]));
      reductionForOps.back()->setAttr("spatial", builder.getUnitAttr());
      builder = OpBuilder(
          &(*(reductionForOps.back().getBody()->getOperations().begin())));
      loc = reductionForOps.back().getBody()->getOperations().begin()->getLoc();
    }
  }

  std::size_t numLoad = allLoadAffineMaps.size();
  for (std::size_t loadCnt = 0; loadCnt < numLoad; ++loadCnt) {
    AffineLoadOp load;
    if (loadCnt < numLoad - 1) { // load from buffer
      if (allLoadOperands[loadCnt + 1].size() > 0)
        for (unsigned int j = 0; j < reductionForOps.size(); ++j) {
          allLoadOperands[loadCnt + 1][j] =
              reductionForOps[j].getInductionVar();
        }
      std::size_t size = allLoadOperands[loadCnt + 1].size();
      for (unsigned int j = size - shiftForOps.size(); j < size; ++j) {
        allLoadOperands[loadCnt + 1][j] =
            shiftForOps[j - size + shiftForOps.size()].getInductionVar();
      }
      load =
          builder.create<AffineLoadOp>(loc, buf, allLoadAffineMaps[loadCnt + 1],
                                       allLoadOperands[loadCnt + 1]);
    } else { // load from memory
      if (reductionForOps.size() > 0) {
        SmallVector<AffineExpr> loadAffineExpr;
        SmallVector<Value> memAffineIndices;
        auto operands = originalLoadOp.getMapOperands();
        auto loadMap = originalLoadOp.getAffineMap();
        int operandIdx = 0;
        int loadRank = 0;
        int RLCnt =
            0; // reduction loop count (shift buffer loops inside `axis`)
        int SLCnt = 0; // shift loop count (shift buffer loops outside `axis`)
        for (int i = 0; i < (int)rank; ++i) {
          auto expr = loadMap.getResult(i);
          if (i < axis) {
            if (spans[i] > 1) {
              if (llvm::isa<AffineBinaryOpExpr>(expr)) {
                auto dim0 = builder.getAffineDimExpr(loadRank++);
                auto dim1 = builder.getAffineDimExpr(loadRank++);
                loadAffineExpr.push_back(dim0 + dim1);
                memAffineIndices.push_back(
                    nonReductionLoops[i].getInductionVar());
                memAffineIndices.push_back(
                    reductionForOps[RLCnt++].getInductionVar());
                operandIdx++;
                operandIdx++;
              } else if (llvm::isa<AffineDimExpr>(expr)) { // single reduction
                loadAffineExpr.push_back(builder.getAffineDimExpr(loadRank++));
                memAffineIndices.push_back(
                    reductionForOps[RLCnt++].getInductionVar());
                operandIdx++;
              } else { // AffineConstantExpr
                loadAffineExpr.push_back(builder.getAffineDimExpr(loadRank++));
                memAffineIndices.push_back(
                    reductionForOps[RLCnt++].getInductionVar());
              }
            } else {
              loadAffineExpr.push_back(builder.getAffineDimExpr(loadRank++));
              memAffineIndices.push_back(operands[operandIdx++]);
            }
          } else if (i == axis) {
            loadAffineExpr.push_back(builder.getAffineDimExpr(loadRank++));
            memAffineIndices.push_back(operands[operandIdx++]);
            if (llvm::isa<AffineBinaryOpExpr>(expr))
              operandIdx++;
          } else {
            if (llvm::isa<AffineBinaryOpExpr>(expr)) {
              loadAffineExpr.push_back(builder.getAffineDimExpr(loadRank++));
              operandIdx++;
              memAffineIndices.push_back(
                  shiftForOps[SLCnt++].getInductionVar());
              operandIdx++;
            } else if (llvm::isa<AffineDimExpr>(expr)) {
              loadAffineExpr.push_back(builder.getAffineDimExpr(loadRank++));
              memAffineIndices.push_back(
                  shiftForOps[SLCnt++].getInductionVar());
              operandIdx++;
            } else { // AffineConstantExpr
              loadAffineExpr.push_back(expr);
            }
          }
        }
        auto affineMap =
            AffineMap::get(loadRank, 0, loadAffineExpr, builder.getContext());
        for (auto operand_item : llvm::enumerate(memAffineIndices)) {
          auto operand = operand_item.value();
          // if operand is block argument, skip it
          if (llvm::isa<BlockArgument>(operand))
            continue;

          // if the operand's defining op is an affine.apply op, we need to
          // create identical affine.apply ops for the load op
          // there might be a cascade of affine.apply ops, so we need to
          // clone them all
          if (llvm::isa<AffineApplyOp>(operand.getDefiningOp())) {
            SmallVector<Operation *, 4> affineApplyOps;
            SmallVector<Operation *, 2> worklist{operand.getDefiningOp()};
            while (!worklist.empty()) {
              auto front = worklist[0];
              worklist.erase(worklist.begin()); // pop front
              if (llvm::isa<AffineApplyOp>(front)) {
                affineApplyOps.push_back(front);
                for (auto opr : front->getOperands()) {
                  if (llvm::isa<BlockArgument>(opr))
                    continue;
                  if (llvm::isa<AffineApplyOp>(opr.getDefiningOp())) {
                    worklist.push_back(opr.getDefiningOp());
                  }
                }
              }
            }
            std::reverse(affineApplyOps.begin(), affineApplyOps.end());
            SmallVector<Operation *, 4> newAffineApplyOps;
            for (auto op : affineApplyOps) {
              auto cloned_op = builder.clone(*op);
              newAffineApplyOps.push_back(cloned_op);
            }
            // Make sure the new affine.apply ops have the correct
            // operands
            for (unsigned op_idx = 0; op_idx < affineApplyOps.size();
                 ++op_idx) {
              auto orgOp = affineApplyOps[op_idx];
              auto newOp = newAffineApplyOps[op_idx];
              for (unsigned opr_idx = 0; opr_idx < orgOp->getNumOperands();
                   ++opr_idx) {
                auto orgOpr = orgOp->getOperand(opr_idx);
                if (llvm::isa<BlockArgument>(orgOpr))
                  continue;
                // get the index of orgOpr.getDefiningOp() in affineApplyOps
                auto oprDefOp_idx =
                    getIndex(affineApplyOps, orgOpr.getDefiningOp());
                if (oprDefOp_idx == -1)
                  (*orgOp).emitError(
                      "operand's defining op is not found in affineApplyOps");
                // update the new affine.apply op's operands
                auto newOprDefOp = newAffineApplyOps[oprDefOp_idx];
                newOp->setOperand(opr_idx, newOprDefOp->getResult(0));
              }
            }
            // Update the operand in memAffineIndices
            auto lastNewAffineApplyOp = newAffineApplyOps.back();
            memAffineIndices[operand_item.index()] =
                lastNewAffineApplyOp->getResult(0);
          }
        }
        load = builder.create<AffineLoadOp>(loc, target, affineMap,
                                            memAffineIndices);
      } else {
        SmallVector<Value> memAffineIndices;
        for (int i = 0; i < (int)rank; ++i) {
          AffineExpr baseExpr = originalLoadExprs[i][0];
          // remove the reduction ivs from the baseExpr
          if (llvm::isa<AffineDimExpr>(baseExpr)) {
            memAffineIndices.push_back(dim2iv[baseExpr]);
          } else if (llvm::isa<AffineBinaryOpExpr>(baseExpr)) {
            auto expr = llvm::dyn_cast<AffineBinaryOpExpr>(baseExpr);
            // walk LHS
            expr.getLHS().walk([&](AffineExpr e) {
              if (llvm::isa<AffineDimExpr>(e)) {
                bool isReductionIV = reductionVars.count(dim2iv[e]);
                if (!isReductionIV)
                  memAffineIndices.push_back(dim2iv[e]);
              }
            });

            // walk RHS
            expr.getRHS().walk([&](AffineExpr e) {
              if (llvm::isa<AffineDimExpr>(e)) {
                bool isReductionIV = reductionVars.count(dim2iv[e]);
                if (!isReductionIV)
                  memAffineIndices.push_back(dim2iv[e]);
              }
            });
          } else {
            memAffineIndices.push_back(builder.create<arith::ConstantIndexOp>(
                loc, llvm::dyn_cast<AffineConstantExpr>(baseExpr).getValue()));
          }
        }

        std::size_t size = memAffineIndices.size();
        for (unsigned int j = size - shiftForOps.size(); j < size; ++j) {
          memAffineIndices[j] =
              shiftForOps[j - size + shiftForOps.size()].getInductionVar();
        }
        auto shape = llvm::dyn_cast<MemRefType>(target.getType()).getShape();
        for (int i = 0; i < axis; ++i) {
          if (shape[i] == 1)
            memAffineIndices[i] =
                builder.create<arith::ConstantIndexOp>(loc, 0);
        }
        if (shape.size() != memAffineIndices.size()) {
          reuseAtOp.emitError("ReuseAt failed at step 15: target.shape() != "
                              "memAffineIndices.size()");
        }
        load = builder.create<AffineLoadOp>(loc, target, memAffineIndices);
      }
    }
    if (is_unsigned)
      load->setAttr("unsigned", builder.getUnitAttr());

    // store the load result to buffer
    if (reductionForOps.size() > 0 && allLoadOperands[loadCnt].size() > 0)
      for (unsigned int j = 0; j < reductionForOps.size(); ++j) {
        allLoadOperands[loadCnt][j] = reductionForOps[j].getInductionVar();
      }
    std::size_t size = allLoadOperands[loadCnt].size();
    for (unsigned int j = size - shiftForOps.size(); j < size; ++j) {
      allLoadOperands[loadCnt][j] =
          shiftForOps[j - size + shiftForOps.size()].getInductionVar();
    }
    auto store = builder.create<AffineStoreOp>(
        loc, load, buf, allLoadAffineMaps[loadCnt], allLoadOperands[loadCnt]);
    if (is_unsigned)
      store->setAttr("unsigned", builder.getUnitAttr());
  }

  // 16) Remove all the useless operations
  for (Operation *op : opToRemove) {
    op->erase();
  }

  // 17) Merge loops with the same bound
  if (previousShiftLoops.size() > 0 && cntIf < 2) {
    // TODO: only support one shift loop now
    AffineForOp firstLoop = previousShiftLoops.back();
    AffineForOp secondLoop = nonReductionLoops[loopAxis];
    if (firstLoop.getConstantUpperBound() ==
        secondLoop.getConstantUpperBound()) {
      auto &firstBody = firstLoop.getBody()->getOperations();
      auto &secondBody = secondLoop.getBody()->getOperations();
      auto firstOpInSecondLoop = secondBody.begin();
      // do not need affine.yield op, so that's why using std::prev
      secondBody.splice(secondBody.begin(), firstBody, firstBody.begin(),
                        std::prev(firstBody.end()));
      firstLoop.getInductionVar().replaceAllUsesWith(
          secondLoop.getInductionVar());
      firstLoop.erase();
      auto parent = secondLoop->getParentOp();
      if (llvm::isa<AffineIfOp>(parent)) {
        auto ifOp = llvm::dyn_cast<AffineIfOp>(parent);
        auto &ifBody = ifOp.getThenBlock()->getOperations();
        auto &parentBody =
            nonReductionLoops[loopAxis - 1].getBody()->getOperations();
        parentBody.splice(parentBody.begin(), ifBody, ifBody.begin(),
                          std::prev(ifBody.end()));
        // skip the previous reuse part
        ifOp->moveBefore(&(*firstOpInSecondLoop));
        // move the rest into the if body
        auto &secondBody = secondLoop.getBody()->getOperations();
        ifBody.splice(ifBody.begin(), secondBody, firstOpInSecondLoop,
                      std::prev(secondBody.end()));
      }
    }
  }

  return success();
}

LogicalResult runBufferAt(func::FuncOp &f, BufferAtOp &bufferAtOp) {
  // 1) Get the schedule
  auto target = bufferAtOp.getTarget(); // return a Value type
  auto loopHandle =
      dyn_cast<CreateLoopHandleOp>(bufferAtOp.getAxis().getDefiningOp());
  const auto loop_name = loopHandle.getLoopName();
  const auto op_name =
      dyn_cast<CreateOpHandleOp>(loopHandle.getOp().getDefiningOp())
          .getOpName();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, op_name))) {
    f.emitError("Cannot find Stage ") << op_name.str();
    return failure();
  }

  // 2.1) Find the requested loop and get the axis id
  AffineForOp bufferLoop = rootForOp;
  int axis = getLoop(bufferLoop, loop_name);
  if (axis == -1) {
    f.emitError("Cannot find Loop ") << loop_name.str();
    return failure();
  }

  // 3) Obtain non-reduction loops and reduction loops
  AffineLoopBand band;
  SmallVector<StringRef, 6> nameArr;
  // TODO: test if the requested loop has the target tensor
  bool isFound = findContiguousNestedLoops(rootForOp, band, nameArr);
  if (!isFound) {
    bufferAtOp.emitError("Cannot find nested loops for buffer_at");
    return failure();
  }
  SmallVector<AffineForOp, 6> nonReductionForOps;
  SmallVector<StringRef, 6> nonReductionNameArr;
  int firstReductionIdx = -1;
  for (std::size_t i = 0, e = band.size(); i != e; ++i) {
    if (!band[i]->hasAttr("reduction")) {
      nonReductionForOps.push_back(band[i]);
      nonReductionNameArr.push_back(getLoopName(band[i]));
    } else {
      if (firstReductionIdx == -1)
        firstReductionIdx = i;
    }
  }
  if (firstReductionIdx == -1)
    firstReductionIdx = band.size() - 1;
  // handle exception
  if (axis >= 0 && ((std::size_t)(axis + 1) >= band.size())) {
    bufferAtOp.emitError("Cannot buffer at the inner-most loop: axis=")
        << std::to_string(axis)
        << " inner-most axis=" << std::to_string(band.size() - 1);
    return failure();
  }
  if (axis >= 0 && axis >= firstReductionIdx) {
    bufferAtOp.emitError("Cannot buffer inside the reduction loops: axis=")
        << std::to_string(axis)
        << ", first reduction axis=" << std::to_string(firstReductionIdx);
    return failure();
  }

  // 4) Create write buffer
  // e.g.:
  // without reordering: (0, 1, 2r)
  //   buf_at 0: 1;(1,2r);1 insert at all[axis+1] but take non-red[axis+1]
  //   var buf_at 1: c;2r;c inner-most non-red buf_at 2: x cannot buffer
  //   at the inner-most
  // with reordering: (0, 1r, 2)
  //   buf_at 0: 2;(1r,2);2 non-red[axis+1]
  //   buf_at 1: x cannot buffer inside reduction loop
  //   buf_at 2: x
  if (axis == firstReductionIdx - 1 &&
      (std::size_t)firstReductionIdx ==
          nonReductionForOps.size()) { // inner-most non-reduction loop &&
                                       // no non-reduction loops inside
    OpBuilder builder(band[firstReductionIdx]);
    Location loc_front = band[firstReductionIdx].getLoc();
    mlir::Type elementType =
        llvm::dyn_cast<MemRefType>(target.getType()).getElementType();
    SmallVector<Value, 4> memIndices;
    // a) Initialization
    // buffer only has one element
    auto buf = builder.create<memref::AllocOp>(
        loc_front, MemRefType::get({1}, elementType));
    auto zero = builder.create<arith::ConstantOp>(
        loc_front, elementType, builder.getZeroAttr(elementType));
    // no need to create an explicit loop
    auto idx = builder.create<arith::ConstantIndexOp>(loc_front, 0);
    memIndices.push_back(idx);
    builder.create<AffineStoreOp>(loc_front, zero, buf, memIndices);

    // link the result SSA with the buffer
    bufferAtOp.getResult().replaceAllUsesWith(buf);

    // b) Rewrite the original buffer
    // TODO: possible bug: replace uses before an untraversed op
    SmallVector<Operation *, 10> opToRemove;
    for (Operation &op : band[firstReductionIdx].getBody()->getOperations()) {
      memIndices.clear();
      if (auto load = dyn_cast<AffineLoadOp>(op)) {
        if (load.getOperand(0) != target)
          continue;
        OpBuilder mid_builder(&op);
        memIndices.push_back(idx);
        auto new_load =
            mid_builder.create<AffineLoadOp>(op.getLoc(), buf, memIndices);
        op.replaceAllUsesWith(new_load);
        opToRemove.push_back(&op);
      } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
        if (store.getOperand(1) != target)
          continue;
        OpBuilder mid_builder(&op);
        memIndices.push_back(idx);
        mid_builder.create<AffineStoreOp>(op.getLoc(), op.getOperand(0), buf,
                                          memIndices);
        opToRemove.push_back(&op);
      }
    }
    for (Operation *op : opToRemove) {
      op->erase();
    }

    // c) Write back
    //    no need to create an explicit loop
    memIndices.clear();
    memIndices.push_back(idx);
    auto load_from_buf =
        builder.create<AffineLoadOp>(loc_front, buf, memIndices);
    memIndices.clear();
    for (int i = 0; i < firstReductionIdx; ++i) {
      memIndices.push_back(band[i].getInductionVar());
    }
    builder.create<AffineStoreOp>(loc_front, load_from_buf, target, memIndices);

    // d) move the original loop in the middle
    band[firstReductionIdx]->moveBefore(load_from_buf);

  } else { // not the inner-most non-reduction axis
    OpBuilder builder(band[axis + 1]);
    Location loc_front = band[axis + 1].getLoc();
    SmallVector<int64_t> ubs;
    for (unsigned int i = axis + 1, e = nonReductionForOps.size(); i < e; ++i) {
      ubs.push_back(nonReductionForOps[axis + 1].getConstantUpperBound());
    }
    // TODO: support more data types
    mlir::Type elementType =
        llvm::dyn_cast<MemRefType>(target.getType()).getElementType();
    SmallVector<Value, 4> memIndices;
    // a) Initialization
    // a.1) Allocate buffer
    auto buf = builder.create<memref::AllocOp>(
        loc_front, MemRefType::get(ubs, elementType));
    auto zero = builder.create<arith::ConstantOp>(
        loc_front, elementType, builder.getZeroAttr(elementType));

    // a.2) Create initialization loop
    //      need to create an explicit loop
    SmallVector<AffineForOp> initLoops;
    initLoops.push_back(builder.create<AffineForOp>(loc_front, 0, ubs[0]));
    AffineForOp forOp = initLoops[0];
    for (unsigned int i = axis + 2, e = nonReductionForOps.size(); i < e; ++i) {
      OpBuilder init_builder(&(*(forOp.getBody()->getOperations().begin())));
      forOp = init_builder.create<AffineForOp>(
          forOp.getBody()->getOperations().begin()->getLoc(), 0,
          ubs[i - axis - 1]);
      initLoops.push_back(forOp);
    }

    // a.3) Do the initialization
    OpBuilder init_builder(&(
        *(initLoops[initLoops.size() - 1].getBody()->getOperations().begin())));
    for (auto forOp : initLoops) {
      memIndices.push_back(forOp.getInductionVar());
    }
    init_builder.create<AffineStoreOp>(initLoops[initLoops.size() - 1].getLoc(),
                                       zero, buf, memIndices);

    // b) Rewrite the original buffer
    SmallVector<Operation *, 10> opToRemove;
    band[axis + 1].walk([&](Operation *op) {
      memIndices.clear();
      if (auto load = dyn_cast<AffineLoadOp>(op)) {
        if (load.getOperand(0) != target)
          return;
        OpBuilder mid_builder(op);
        for (unsigned int i = axis + 1, e = nonReductionForOps.size(); i < e;
             ++i) {
          memIndices.push_back(nonReductionForOps[i].getInductionVar());
        }
        auto new_load =
            mid_builder.create<AffineLoadOp>(op->getLoc(), buf, memIndices);
        op->replaceAllUsesWith(new_load);
        opToRemove.push_back(op);
      } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
        if (store.getOperand(1) != target)
          return;
        OpBuilder mid_builder(op);
        for (unsigned int i = axis + 1, e = nonReductionForOps.size(); i < e;
             ++i) {
          memIndices.push_back(nonReductionForOps[i].getInductionVar());
        }
        mid_builder.create<AffineStoreOp>(op->getLoc(), op->getOperand(0), buf,
                                          memIndices);
        opToRemove.push_back(op);
      }
    });
    for (Operation *op : opToRemove) {
      op->erase();
    }

    // c) Write back
    // c.1) Create write back loop
    Location loc_back =
        std::prev(band[axis + 1].getBody()->getOperations().end())->getLoc();
    SmallVector<AffineForOp> writeBackLoops;
    writeBackLoops.push_back(builder.create<AffineForOp>(loc_back, 0, ubs[0]));
    forOp = writeBackLoops[0];
    for (unsigned int i = axis + 2, e = nonReductionForOps.size(); i < e; ++i) {
      OpBuilder back_builder(&(*(forOp.getBody()->getOperations().begin())));
      forOp = back_builder.create<AffineForOp>(
          forOp.getBody()->getOperations().begin()->getLoc(), 0,
          ubs[i - axis - 1]);
      writeBackLoops.push_back(forOp);
    }

    // c.2) Load from intermediate results
    OpBuilder back_builder(&(*(writeBackLoops[writeBackLoops.size() - 1]
                                   .getBody()
                                   ->getOperations()
                                   .begin())));
    memIndices.clear();
    for (auto forOp : writeBackLoops) {
      memIndices.push_back(forOp.getInductionVar());
    }
    auto load_from_buf = back_builder.create<AffineLoadOp>(
        writeBackLoops[writeBackLoops.size() - 1].getLoc(), buf, memIndices);

    // c.3) Store the results back to memory
    memIndices.clear();
    for (int i = 0; i < axis + 1; ++i) {
      memIndices.push_back(nonReductionForOps[i].getInductionVar());
    }
    for (auto forOp : writeBackLoops) {
      memIndices.push_back(forOp.getInductionVar());
    }
    back_builder.create<AffineStoreOp>(
        writeBackLoops[writeBackLoops.size() - 1].getLoc(), load_from_buf,
        target, memIndices);

    // d) Move the original loop between the two loops
    band[axis + 1]->moveBefore(writeBackLoops[0]);

    // e) Add names to loops
    SmallVector<std::string, 6> newNameArr;
    newNameArr.push_back(nonReductionNameArr[axis + 1].str() + "_init");
    newNameArr.push_back(nonReductionNameArr[axis + 1].str() + "_back");
    SmallVector<AffineForOp, 6> newLoops{initLoops[0], writeBackLoops[0]};
    setLoopNames(newLoops, newNameArr);
    initLoops[0]->setAttr("buffer", init_builder.getUnitAttr());
    writeBackLoops[0]->setAttr("buffer", back_builder.getUnitAttr());

    // f) Automatic pipelining
    SmallVector<AffineForOp, 6> twoLoops{
        initLoops[initLoops.size() - 1],
        writeBackLoops[writeBackLoops.size() - 1]};
    SmallVector<int, 6> II{1, 1};
    setIntAttr(twoLoops, II, "pipeline_ii");
  }

  return success();
}

LogicalResult runReshape(func::FuncOp &f, ReshapeOp &reshapeOp, Value &array) {
  // 1) Get the schedule
  auto oldType = llvm::dyn_cast<MemRefType>(array.getType());
  auto newType = llvm::dyn_cast<MemRefType>(reshapeOp.getOutput().getType());
  int oldRank = oldType.getRank();
  int newRank = newType.getRank();
  auto oldShape = oldType.getShape();
  auto newShape = newType.getShape();
  SmallVector<int64_t> prodOldShape;
  prodOldShape.push_back(1);
  for (int i = oldRank - 1; i >= 0; --i)
    prodOldShape.push_back(oldShape[i] * prodOldShape[oldRank - 1 - i]);

  // 2) Set new type
  array.setType(newType);

  // 3) Update memory access
  SmallVector<Operation *> opToRemove;
  for (auto user : array.getUsers()) {
    if (auto op = dyn_cast<AffineStoreOp>(user)) {
      OpBuilder rewriter(op);
      SmallVector<AffineExpr> memAffineIndices;
      memAffineIndices.clear();
      auto oldAffineMap = op.getAffineMap();
      auto linear_addr = rewriter.getAffineConstantExpr(0);
      for (int i = oldRank - 1; i >= 0; --i) {
        AffineExpr idx = oldAffineMap.getResult(i);
        linear_addr = idx * prodOldShape[oldRank - i - 1] + linear_addr;
      }
      for (int i = 1; i < newRank; ++i) {
        memAffineIndices.push_back(linear_addr % newShape[newRank - i]);
        linear_addr = linear_addr.floorDiv(newShape[newRank - i]);
      }
      memAffineIndices.push_back(linear_addr);
      std::reverse(memAffineIndices.begin(), memAffineIndices.end());
      auto affineMap = AffineMap::get(oldRank, 0 /* symbols */,
                                      memAffineIndices, rewriter.getContext());
      rewriter.create<AffineStoreOp>(
          op->getLoc(), op.getOperand(0) /*valueToStore*/,
          op.getOperand(1) /*memref*/, affineMap, op.getIndices());
      // remove original op
      opToRemove.push_back(op);
    } else if (auto op = dyn_cast<AffineLoadOp>(user)) {
      OpBuilder rewriter(op);
      SmallVector<AffineExpr> memAffineIndices;
      memAffineIndices.clear();
      auto oldAffineMap = op.getAffineMap();
      auto linear_addr = rewriter.getAffineConstantExpr(0);
      for (int i = oldRank - 1; i >= 0; --i) {
        AffineExpr idx = oldAffineMap.getResult(i);
        linear_addr = idx * prodOldShape[oldRank - i - 1] + linear_addr;
      }
      for (int i = 1; i < newRank; ++i) {
        memAffineIndices.push_back(linear_addr % newShape[newRank - i]);
        linear_addr = linear_addr.floorDiv(newShape[newRank - i]);
      }
      memAffineIndices.push_back(linear_addr);
      std::reverse(memAffineIndices.begin(), memAffineIndices.end());
      auto affineMap = AffineMap::get(oldRank, 0 /* symbols */,
                                      memAffineIndices, rewriter.getContext());
      auto load = rewriter.create<AffineLoadOp>(op->getLoc(),
                                                op.getOperand(0) /*memref*/,
                                                affineMap, op.getIndices());
      // remove original op
      op.getResult().replaceAllUsesWith(load);
      opToRemove.push_back(op);
    }
  }

  // 4) update function signature
  auto builder = Builder(array.getContext());
  auto resultTypes = f.front().getTerminator()->getOperandTypes();
  auto inputTypes = f.front().getArgumentTypes();
  f.setType(builder.getFunctionType(inputTypes, resultTypes));

  // 5) Remove all the useless operations
  for (Operation *op : opToRemove) {
    op->erase();
  }
  return success();
}

LogicalResult
runInterKernelDataPlacement(std::map<std::string, func::FuncOp> &funcMap,
                            Value &arrayToStream, int fifo_depth = -1) {
  // Construct new array type (add stream attribute)
  auto arrayType = llvm::dyn_cast<MemRefType>(arrayToStream.getType());
  auto shape = arrayType.getShape();
  if (fifo_depth == -1) {
    // a conversative estimation
    fifo_depth = 1;
    for (auto size : shape)
      fifo_depth *= size;
  }
  auto newType = MemRefType::get(
      arrayType.getShape(), arrayType.getElementType(), arrayType.getLayout(),
      StringAttr::get(arrayToStream.getDefiningOp()->getContext(),
                      "stream:" + std::to_string(fifo_depth)));

  // Set new type in the top function
  arrayToStream.setType(newType);

  // Set new types in stage functions
  for (auto user : arrayToStream.getUsers()) {
    // first locate the CallOp
    if (auto callOp = dyn_cast<func::CallOp>(user)) {
      // get stage function
      auto stage = funcMap[callOp.getCallee().str().substr(6)];
      for (unsigned argIdx = 0, e = user->getNumOperands(); argIdx < e;
           ++argIdx) {
        // find the corresponding array
        if (callOp.getArgOperands()[argIdx] == arrayToStream) {
          // first change argument type
          stage.getArgument(argIdx).setType(newType);
          // get new function input types
          llvm::SmallVector<mlir::Type> inputTypes;
          for (auto indexedArg :
               llvm::enumerate(stage.front().getArgumentTypes())) {
            if (indexedArg.index() != argIdx) {
              inputTypes.push_back(indexedArg.value());
            } else {
              inputTypes.push_back(newType);
            }
          }
          auto resultTypes = stage.front().getTerminator()->getOperandTypes();
          // update function signature
          stage.setType(
              FunctionType::get(stage.getContext(), inputTypes, resultTypes));
          break;
        }
      }
    }
  }
  return success();
}

LogicalResult runInterKernelDataPlacementSingleFunction(Value &arrayToStream,
                                                        int fifo_depth = -1) {
  // Construct new array type (add stream attribute)
  auto arrayType = llvm::dyn_cast<MemRefType>(arrayToStream.getType());
  auto shape = arrayType.getShape();
  if (fifo_depth == -1) {
    // a conversative estimation
    fifo_depth = 1;
    for (auto size : shape)
      fifo_depth *= size;
  }
  auto newType = MemRefType::get(
      arrayType.getShape(), arrayType.getElementType(), arrayType.getLayout(),
      StringAttr::get(arrayToStream.getDefiningOp()->getContext(),
                      "stream:" + std::to_string(fifo_depth)));

  // Set new type
  arrayToStream.setType(newType);
  return success();
}

template <class T, int opId>
void getInputMemRefs(AffineForOp stage, SmallVector<Value> &allMemrefs,
                     std::set<Operation *> &opToMove) {
  stage.walk([&](T op) {
    auto target = op.getOperand(opId);
    auto defOp = target.getDefiningOp();
    if (defOp && (llvm::isa<memref::GetGlobalOp>(defOp) ||
                  llvm::isa<allo::GetGlobalFixedOp>(defOp))) {
      opToMove.insert(defOp);
    } else {
      if (std::find(allMemrefs.begin(), allMemrefs.end(), target) ==
          allMemrefs.end())
        allMemrefs.push_back(target);
      for (unsigned argIdx = 1, e = op->getNumOperands(); argIdx < e;
           ++argIdx) {
        auto operand = op.getOperand(argIdx);
        auto memrefType = llvm::dyn_cast<MemRefType>(operand.getType());
        if (operand.getDefiningOp()) {
          if (memrefType && memrefType.getRank() == 1 &&
              memrefType.getShape()[0] == 1)
            continue; // sum reg needn't to be moved
          opToMove.insert(operand.getDefiningOp());
        }
      }
    }
  });
}

template <class T, int opId>
void getOutputMemRefs(AffineForOp stage, SmallVector<Value> &allMemrefs,
                      std::set<Operation *> &opToMove) {
  SmallVector<Value> memrefToRemove;
  const auto op_name =
      llvm::dyn_cast<StringAttr>(stage->getAttr("op_name")).getValue().str();
  stage.walk([&](T op) {
    auto target = op.getOperand(opId);
    if (std::find(allMemrefs.begin(), allMemrefs.end(), target) ==
        allMemrefs.end()) { // need to prevent adding the same memref again
      allMemrefs.push_back(target);
    } else {
      if (allMemrefs.size() == 1)
        return WalkResult::advance();
      auto memrefType = llvm::dyn_cast<MemRefType>(target.getType());
      if (target.getDefiningOp()) {
        memrefToRemove.push_back(target);
        if (memrefType && memrefType.getRank() == 1 &&
            memrefType.getShape()[0] == 1)
          return WalkResult::advance(); // sum reg needn't to be moved
        opToMove.insert(target.getDefiningOp());
      }
    }
    return WalkResult::advance();
  });
  for (auto target : memrefToRemove) {
    allMemrefs.erase(std::remove(allMemrefs.begin(), allMemrefs.end(), target),
                     allMemrefs.end());
  }
}

LogicalResult runOutline(ModuleOp &mod, func::FuncOp &f, OutlineOp &outlineOp) {
  // 1) Get the schedule
  auto stages = outlineOp.getStages();
  SmallVector<AffineForOp> rootForOps;
  SmallVector<Value> allMemrefs;
  std::vector<std::string> stageNames;
  std::set<Operation *> opToMove;
  for (auto stage : stages) {
    const auto op_name =
        llvm::dyn_cast<CreateOpHandleOp>(stage.getDefiningOp()).getOpName();
    stageNames.push_back(op_name.str());

    // 2) Find the requested stages
    AffineForOp rootForOp;
    if (failed(getStage(f, rootForOp, op_name))) {
      f.emitError("Cannot find Stage ") << op_name.str();
      return failure();
    }
    rootForOps.push_back(rootForOp);

    // 3) Find all load memrefs (inputs)
    getInputMemRefs<AffineLoadOp, 0>(rootForOp, allMemrefs, opToMove);
    getInputMemRefs<memref::LoadOp, 0>(rootForOp, allMemrefs, opToMove);
  }

  // 4) Find all store memrefs (outputs)
  for (auto rootForOp : rootForOps) {
    getOutputMemRefs<AffineStoreOp, 1>(rootForOp, allMemrefs, opToMove);
    getOutputMemRefs<memref::StoreOp, 1>(rootForOp, allMemrefs, opToMove);
  }
  SmallVector<Value> newMemrefs(allMemrefs);

  // 5) If the function has been built, directly call it
  if (outlineOp->hasAttr("unify")) {
    func::FuncOp targetFunc;
    auto preFuncName =
        llvm::dyn_cast<StringAttr>(outlineOp->getAttr("unify")).getValue();
    for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
      if (func.getName() == preFuncName) {
        targetFunc = func;
        break;
      }
    }
    assert(targetFunc && "Cannot find the target function");
    OpBuilder call_builder(rootForOps[rootForOps.size() - 1]);
    SmallVector<std::pair<AffineForOp, int>> loops;
    int cntIdx = allMemrefs.size();
    for (auto srcForOpItem : llvm::enumerate(rootForOps)) {
      int srcIdx = srcForOpItem.index();
      auto srcForOp = srcForOpItem.value();
      SmallVector<AffineForOp> srcLoops;
      getLoops(srcForOp, srcLoops);
      SmallVector<AffineForOp> targetLoops;
      for (auto targetForOpItem :
           llvm::enumerate(targetFunc.getOps<AffineForOp>())) {
        int targetIdx = targetForOpItem.index();
        if (targetIdx == srcIdx) {
          auto targetForOp = targetForOpItem.value();
          getLoops(targetForOp, targetLoops);
          break;
        }
      }
      assert(targetLoops.size() == srcLoops.size() && "Loop mismatch");
      for (auto it : llvm::zip(srcLoops, targetLoops)) {
        auto srcLoop = std::get<0>(it);
        auto targetLoop = std::get<1>(it);
        // get current CallOp's operands
        if (targetLoop.hasConstantUpperBound()) { // has not been parameterized
          if (srcLoop.getConstantUpperBound() !=
              targetLoop.getConstantUpperBound()) {
            auto srcUb = call_builder.create<arith::ConstantIndexOp>(
                srcLoop.getLoc(), srcLoop.getConstantUpperBound());
            allMemrefs.push_back(srcUb);
            loops.push_back({targetLoop, -1});
          } else {
            // no need to parameterize
          }
        } else {
          if (srcLoop.hasConstantUpperBound()) {
            auto srcUb = call_builder.create<arith::ConstantIndexOp>(
                srcLoop.getLoc(), srcLoop.getConstantUpperBound());
            allMemrefs.push_back(srcUb);
            int idx = -1;
            for (auto item : llvm::enumerate(targetFunc.getArguments())) {
              if (item.value() == targetLoop.getUpperBound().getOperand(0)) {
                idx = item.index();
                break;
              }
            }
            assert(idx != -1 && "Not found target IV");
            loops.push_back({targetLoop, idx});
          } else {
            assert(srcLoop.getUpperBound().getMap() ==
                       targetLoop.getUpperBound().getMap() &&
                   "map mismatch");
          }
        }
      }
    }
    // update previous CallOp
    for (auto callOp : f.getOps<func::CallOp>()) {
      if (callOp.getCallee() == targetFunc.getName()) {
        OpBuilder builder(callOp);
        int prevNumOperands = callOp.getNumOperands();
        SmallVector<Value> prevOperands(callOp.getOperands());
        int currIdx = cntIdx;
        for (auto item : loops) {
          auto targetLoop = item.first;
          int idx = item.second;
          auto targetUb = (idx == -1 ? builder.create<arith::ConstantIndexOp>(
                                           targetLoop.getLoc(),
                                           targetLoop.getConstantUpperBound())
                                     : prevOperands[idx]);
          if (currIdx < prevNumOperands) {
            callOp->setOperand(currIdx++, targetUb);
          } else {
            callOp->insertOperands(callOp.getNumOperands(), {targetUb});
          }
        }
      }
    }
    // update function arguments
    int currIdx = cntIdx;
    for (auto item : loops) {
      auto targetLoop = item.first;
      if (item.second == -1) { // has not been parameterized
        auto arg = targetFunc.front().insertArgument(
            currIdx++, IndexType::get(f.getContext()), targetLoop.getLoc());
        // update target loop bound
        targetLoop.setUpperBound({arg}, call_builder.getSymbolIdentityMap());
      } else {
        currIdx++;
      }
    }
    // update if structure
    SmallVector<AffineIfOp> srcIfOps;
    for (auto rootForOp : rootForOps) {
      rootForOp.walk([&](AffineIfOp ifOp) { srcIfOps.push_back(ifOp); });
    }
    SmallVector<AffineIfOp> targetIfOps;
    for (auto rootForOp : targetFunc.getOps<AffineForOp>()) {
      rootForOp.walk([&](AffineIfOp ifOp) { targetIfOps.push_back(ifOp); });
    }
    assert(srcIfOps.size() == targetIfOps.size() && "IfOp mismatch");
    bool isDifferent = false;
    bool isParameterized = true;
    for (auto it : llvm::zip(srcIfOps, targetIfOps)) {
      auto srcIfOp = std::get<0>(it);
      auto targetIfOp = std::get<1>(it);
      auto srcConds = srcIfOp.getIntegerSet().getConstraints();
      auto targetConds = targetIfOp.getIntegerSet().getConstraints();
      SmallVector<AffineExpr> newConds;
      int srcCst = 0;
      int targetCst = 0;
      for (auto itCond : llvm::zip(srcConds, targetConds)) {
        auto srcCond = std::get<0>(itCond);
        auto targetCond = std::get<1>(itCond);
        auto getConstant = [&](AffineExpr cond) {
          int cst = 0;
          cond.walk([&](AffineExpr expr) {
            if (llvm::isa<AffineBinaryOpExpr>(expr) &&
                expr.getKind() == AffineExprKind::Add) {
              auto binExpr = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
              if (llvm::isa<AffineConstantExpr>(binExpr.getRHS())) {
                cst = llvm::dyn_cast<AffineConstantExpr>(binExpr.getRHS())
                          .getValue();
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          });
          return cst;
        };
        srcCst = getConstant(srcCond);
        targetCst = getConstant(targetCond);
        // build new condition
        if (srcCst != targetCst) {
          isDifferent = true;
          if (!targetCond.isFunctionOfSymbol(0)) { // has not been parameterized
            isParameterized = false;
            auto newCond = targetCond -
                           call_builder.getAffineConstantExpr(targetCst) +
                           call_builder.getAffineSymbolExpr(0);
            newConds.push_back(newCond);
            // outlineOp.emitWarning("Parameterize if condition of ")
            //     << targetCond << " as " << newCond;
          }
        } else {
          newConds.push_back(targetCond);
        }
      }
      if (isDifferent) {
        // update function arguments
        auto ifCst = call_builder.create<arith::ConstantIndexOp>(
            targetFunc.getLoc(), srcCst);
        allMemrefs.push_back(ifCst);
      }
      if (!isParameterized) {
        targetFunc.front().addArgument(IndexType::get(f.getContext()),
                                       targetFunc.getLoc());
        // update previous CallOp
        for (auto callOp : f.getOps<func::CallOp>()) {
          if (callOp.getCallee() == targetFunc.getName()) {
            OpBuilder builder(callOp);
            auto targetUb = builder.create<arith::ConstantIndexOp>(
                targetFunc.getLoc(), targetCst);
            callOp->insertOperands(callOp.getNumOperands(), {targetUb});
          }
        }
        // update if operands
        auto newCondSet = IntegerSet::get(
            targetIfOp.getIntegerSet().getNumDims() /*dimCount*/,
            1 /*symbolCount*/, newConds /*ArrayRef<AffineExpr> constraints*/,
            targetIfOp.getIntegerSet().getEqFlags());
        SmallVector<Value> operands = targetIfOp.getOperands();
        operands.push_back(targetFunc.front().getArgument(
            targetFunc.front().getNumArguments() - 1));
        targetIfOp.setConditional(newCondSet, operands);
      }
    }
    // different memory access indices with stride
    SmallVector<AffineLoadOp> srcLoadOps;
    for (auto rootForOp : rootForOps) {
      rootForOp.walk(
          [&](AffineLoadOp loadOp) { srcLoadOps.push_back(loadOp); });
    }
    SmallVector<AffineLoadOp> targetLoadOps;
    for (auto rootForOp : targetFunc.getOps<AffineForOp>()) {
      rootForOp.walk(
          [&](AffineLoadOp loadOp) { targetLoadOps.push_back(loadOp); });
    }
    for (auto it : llvm::zip(srcLoadOps, targetLoadOps)) {
      auto srcLoadOp = std::get<0>(it);
      auto targetLoadOp = std::get<1>(it);
      auto srcLoadMap = srcLoadOp.getAffineMap();
      auto targetLoadMap = targetLoadOp.getAffineMap();
      SmallVector<AffineExpr> newExprs;
      isDifferent = false;
      isParameterized = false;
      int targetCst = 1;
      if (srcLoadMap != targetLoadMap) {
        bool isMod = false;
        for (auto item :
             llvm::zip(srcLoadMap.getResults(), targetLoadMap.getResults())) {
          auto expr = std::get<0>(item);
          auto targetExpr = std::get<1>(item);
          if (targetLoadMap.getNumSymbols() > 0 &&
              llvm::isa<AffineBinaryOpExpr>(targetExpr) &&
              targetExpr.getKind() != AffineExprKind::Add) {
            int cst = 1;
            if (llvm::isa<AffineBinaryOpExpr>(expr) &&
                expr.getKind() != AffineExprKind::Add)
              cst = llvm::dyn_cast<AffineConstantExpr>(
                        llvm::dyn_cast<AffineBinaryOpExpr>(expr).getRHS())
                        .getValue();
            auto cstOp = call_builder.create<arith::ConstantIndexOp>(
                targetFunc.getLoc(), cst);
            if (!isParameterized)
              allMemrefs.push_back(cstOp);
            isParameterized = true;
          } else if (llvm::isa<AffineBinaryOpExpr>(expr) &&
                     expr.getKind() != AffineExprKind::Add) {
            auto cst = llvm::dyn_cast<AffineConstantExpr>(
                           llvm::dyn_cast<AffineBinaryOpExpr>(expr).getRHS())
                           .getValue();
            auto cstOp = call_builder.create<arith::ConstantIndexOp>(
                targetFunc.getLoc(), cst);
            if (llvm::isa<AffineBinaryOpExpr>(targetExpr)) {
              targetCst =
                  llvm::dyn_cast<AffineConstantExpr>(
                      llvm::dyn_cast<AffineBinaryOpExpr>(targetExpr).getRHS())
                      .getValue();
            }
            if (!isDifferent)
              allMemrefs.push_back(cstOp);
            isDifferent = true;
            AffineExpr newExpr;
            if (expr.getKind() == AffineExprKind::Mul)
              newExpr = llvm::dyn_cast<AffineBinaryOpExpr>(expr).getLHS() *
                        call_builder.getAffineSymbolExpr(0);
            else if (expr.getKind() == AffineExprKind::Mod) {
              newExpr = llvm::dyn_cast<AffineBinaryOpExpr>(expr).getLHS() %
                        call_builder.getAffineSymbolExpr(0);
              isMod = true;
            } else
              assert(false && "Unexpected affine expr kind");
            newExprs.push_back(newExpr);
          } else {
            newExprs.push_back(expr);
          }
        }
        if (!isParameterized) {
          auto arg = targetFunc.front().addArgument(
              IndexType::get(f.getContext()), targetFunc.getLoc());
          targetLoadOp->insertOperands(targetLoadOp.getNumOperands(), {arg});
          auto map = AffineMap::get(targetLoadMap.getNumDims(), 1, newExprs,
                                    targetFunc.getContext());
          targetLoadOp->setAttr("map", AffineMapAttr::get(map));
          if (isMod) {
            // See the issue:
            // https://github.com/cornell-zhang/allo-dialect/issues/127
            OpBuilder builder(targetLoadOp);
            SmallVector<Value> indices(targetLoadOp.getIndices());
            int pos = -1;
            for (auto item :
                 llvm::enumerate(targetLoadOp.getAffineMap().getResults())) {
              auto expr = item.value();
              if (llvm::isa<AffineBinaryOpExpr>(expr) &&
                  expr.getKind() == AffineExprKind::Mod) {
                pos = item.index();
                break;
              }
            }
            assert(pos != -1 && "Mod op not found");
            auto modOp = builder.create<arith::RemSIOp>(
                targetLoadOp.getLoc(), indices[pos],
                indices[indices.size() - 1]);
            indices.pop_back();
            indices[pos] = modOp.getResult();
            auto loadOp = builder.create<memref::LoadOp>(
                targetLoadOp.getLoc(), targetLoadOp.getMemRef(), indices);
            targetLoadOp.getResult().replaceAllUsesWith(loadOp.getResult());
            targetLoadOp.erase();
          }
        }
      }
      // update previous CallOp
      if (isDifferent && !isParameterized) {
        for (auto callOp : f.getOps<func::CallOp>()) {
          if (callOp.getCallee() == targetFunc.getName()) {
            OpBuilder builder(callOp);
            auto cst = builder.create<arith::ConstantIndexOp>(
                targetFunc.getLoc(), targetCst);
            callOp->insertOperands(callOp.getNumOperands(), {cst});
          }
        }
      }
    }
    // Recursively update array types
    SmallVector<Value> newAllMemrefs(allMemrefs);
    for (auto *op : opToMove) {
      if (llvm::isa<memref::AllocOp>(op))
        newAllMemrefs.push_back(llvm::cast<memref::AllocOp>(op).getResult());
    }
    SmallVector<Value> newFuncArgs;
    for (auto arg : targetFunc.getArguments()) {
      newFuncArgs.push_back(arg);
    }
    for (auto alloc : targetFunc.getOps<memref::AllocOp>()) {
      newFuncArgs.push_back(alloc.getResult());
    }
    bool isChanged = true;
    while (isChanged) {
      isChanged = false;
      for (auto item : llvm::enumerate(llvm::zip(newAllMemrefs, newFuncArgs))) {
        auto idx = item.index();
        auto srcMemref = std::get<0>(item.value());
        auto targetMemref = std::get<1>(item.value());
        auto funcArgType = llvm::dyn_cast<MemRefType>(targetMemref.getType());
        auto arrayType = llvm::dyn_cast<MemRefType>(srcMemref.getType());
        if (!funcArgType || !arrayType) { // not a memref (index type)
          assert(funcArgType == arrayType && "Type mismatch");
          continue;
        }
        mlir::Type elementType = arrayType.getElementType();
        assert(elementType == funcArgType.getElementType() && "Type mismatch");
        auto funcArgShape = funcArgType.getShape();
        auto arrayShape = arrayType.getShape();
        SmallVector<int64_t> newShape;
        // pick the larger shape
        for (auto shape : llvm::enumerate(arrayShape)) {
          if (shape.value() < funcArgShape[shape.index()]) {
            newShape.push_back(funcArgShape[shape.index()]);
          } else {
            newShape.push_back(shape.value());
          }
        }
        assert(arrayType.getLayout().getAffineMap() ==
                   funcArgType.getLayout().getAffineMap() &&
               "Layout mismatch");
        auto newType =
            MemRefType::get(newShape, elementType, arrayType.getLayout(),
                            arrayType.getMemorySpace());
        if (newType != arrayType) {
          if (!srcMemref.getDefiningOp()) {
            outlineOp.emitError("Change memref of ")
                << srcMemref << " to a new type " << newType;
          } else {
            outlineOp.emitWarning("Change memref of ")
                << srcMemref << " to a new type " << newType;
          }
          srcMemref.setType(newType);
          isChanged = true;
        }
        if (newType != funcArgType) {
          outlineOp.emitWarning("Change argument ")
              << targetMemref << " of function " << targetFunc.getName()
              << " to a new type " << newType;
          targetMemref.setType(newType);
          isChanged = true;
        }
        // update previous call operations
        if (idx < allMemrefs.size()) {
          for (auto callOp : f.getOps<func::CallOp>()) {
            if (callOp.getCallee() == targetFunc.getName() &&
                callOp.getOperand(idx).getType() != newType) {
              if (!srcMemref.getDefiningOp())
                outlineOp.emitError("Change argument ")
                    << callOp.getOperand(idx) << " of CallOp " << callOp
                    << " to a new type " << newType;
              callOp.getOperand(idx).setType(newType);
            }
          }
        }
      }
    }
    // Double check previous call operations
    for (auto callOp : f.getOps<func::CallOp>()) {
      for (auto func : mod.getOps<func::FuncOp>()) {
        if (callOp.getCallee() == func.getName()) {
          for (int i = 0, size = func.getNumArguments(); i < size; ++i) {
            auto callMemrefType =
                llvm::dyn_cast<MemRefType>(callOp.getOperand(i).getType());
            auto funcMemrefType =
                llvm::dyn_cast<MemRefType>(func.getArgument(i).getType());
            if (callMemrefType != funcMemrefType) {
              outlineOp.emitWarning("Argument ")
                  << i << " of CallOp " << callOp
                  << " is not the same type as argument " << i
                  << " of function " << func.getName() << "\n";
              func.front().getArgument(i).setType(
                  callOp.getOperand(i).getType());
              auto resultTypes =
                  func.front().getTerminator()->getOperandTypes();
              auto inputTypes = func.front().getArgumentTypes();
              func.setType(Builder(func.getContext())
                               .getFunctionType(inputTypes, resultTypes));
            }
          }
          break;
        }
      }
    }
    // Fix function type
    auto resultTypes = targetFunc.front().getTerminator()->getOperandTypes();
    auto inputTypes = targetFunc.front().getArgumentTypes();
    targetFunc.setType(Builder(targetFunc.getContext())
                           .getFunctionType(inputTypes, resultTypes));
    resultTypes = f.front().getTerminator()->getOperandTypes();
    inputTypes = f.front().getArgumentTypes();
    f.setType(Builder(f.getContext()).getFunctionType(inputTypes, resultTypes));
    // Call the function
    call_builder.create<func::CallOp>(
        rootForOps[rootForOps.size() - 1].getLoc(), targetFunc, allMemrefs);
    for (auto rootForOp : rootForOps) {
      rootForOp.erase();
    }
    return success();
  }

  // 6) Create a new function
  auto builder = OpBuilder(f);
  AffineForOp targetForOp;
  SmallVector<mlir::Type> TypeArr;
  for (auto memref : newMemrefs)
    TypeArr.push_back(memref.getType());
  int axis = -1;
  if (outlineOp->hasAttr("axis")) {
    // Suppose only one stage is given
    assert(rootForOps.size() == 1 && "Only one stage is expected");
    auto loopName =
        llvm::dyn_cast<StringAttr>(outlineOp->getAttr("axis")).getValue();
    targetForOp = rootForOps[0];
    axis = getLoop(targetForOp, loopName);
    for (int i = 0; i <= axis; ++i)
      TypeArr.push_back(IndexType::get(f.getContext()));
  } else {
    targetForOp = rootForOps[rootForOps.size() - 1];
  }
  TypeRange argTypes(TypeArr);
  FunctionType funcType = builder.getFunctionType(argTypes, TypeRange{});
  std::string func_name = "Stage";
  for (auto op_name : stageNames) {
    func_name += "_" + op_name;
  }
  auto func =
      builder.create<func::FuncOp>(f.getLoc(), StringRef(func_name), funcType);
  func.setPrivate();
  // used for generating HLS ap_int/fixed types
  func->setAttr("bit", builder.getUnitAttr());
  // fix unsigned types
  std::string itypes = "";
  for (auto memref : allMemrefs) {
    if (memref.getDefiningOp()) {
      auto op = memref.getDefiningOp();
      if (op->hasAttr("unsigned"))
        itypes += "u";
      else
        itypes += "_";
    } else {
      if (f->hasAttr("itypes")) {
        auto top_itypes =
            llvm::dyn_cast<StringAttr>(f->getAttr("itypes")).getValue().str();
        int argIdx = 0;
        for (auto arg : f.getArguments()) {
          if (arg == memref) {
            break;
          }
          argIdx++;
        }
        itypes += top_itypes[argIdx];
      } else {
        itypes += "_";
      }
    }
  }
  func->setAttr("itypes", StringAttr::get(func.getContext(), itypes));
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  auto ret = builder.create<func::ReturnOp>(func->getLoc());

  if (outlineOp->hasAttr("axis")) {
    // 7) Create callop in the main function
    OpBuilder call_builder(targetForOp);
    call_builder = OpBuilder(&(targetForOp.getBody()->back()));
    AffineForOp innerLoop = rootForOps[0];
    int cntAxis = 0;
    while (cntAxis <= axis) {
      allMemrefs.push_back(innerLoop.getInductionVar());
      if (cntAxis == axis)
        break;
      for (auto loop : innerLoop.getOps<AffineForOp>()) {
        innerLoop = loop;
        break;
      }
      cntAxis++;
    }
    call_builder.create<func::CallOp>(targetForOp.getLoc(), func, allMemrefs);
    // 8) Move original stage to the new function
    auto &targetBody = targetForOp.getBody()->getOperations();
    auto &funcBody = func.front().getOperations();
    // the last two ops are yield and callop
    funcBody.splice(funcBody.begin(), targetBody, targetBody.begin(),
                    std::prev(std::prev(targetBody.end())));
    for (auto *op : opToMove) {
      op->moveBefore(rootForOps[0]);
    }
    // 9) Update memrefs
    for (auto item : llvm::enumerate(newMemrefs)) {
      auto newMemref = func.getArgument(item.index());
      auto oldMemref = item.value();
      replaceAllUsesInRegionWith(oldMemref, newMemref, func.getBody());
    }
    innerLoop = rootForOps[0];
    cntAxis = 0;
    while (cntAxis <= axis) {
      replaceAllUsesInRegionWith(innerLoop.getInductionVar(),
                                 func.getArgument(newMemrefs.size() + cntAxis),
                                 func.getBody());
      if (cntAxis == axis)
        break;
      for (auto loop : innerLoop.getOps<AffineForOp>()) {
        innerLoop = loop;
        break;
      }
      cntAxis++;
    }
  } else {
    // 7) Create callop in the main function
    OpBuilder call_builder(targetForOp);
    call_builder.create<func::CallOp>(targetForOp.getLoc(), func, allMemrefs);
    // 8) Move original stage to the new function
    for (auto rootForOp : rootForOps) {
      rootForOp->moveBefore(ret);
    }
    for (auto *op : opToMove) {
      op->moveBefore(rootForOps[0]);
    }
    // 9) Update memrefs
    for (auto item : llvm::enumerate(newMemrefs)) {
      auto newMemref = func.getArgument(item.index());
      auto oldMemref = item.value();
      for (auto rootForOp : rootForOps)
        replaceAllUsesInRegionWith(oldMemref, newMemref, rootForOp.getRegion());
    }
  }

  return success();
}

template <class T>
void updateMemrefAccess(Operation *&user, SmallVector<AffineExpr> &dimExprs) {
  if (auto op = dyn_cast<T>(user)) {
    auto oldAffineMap = op.getAffineMap();
    SmallVector<AffineExpr> memAffineIndices;
    for (auto dim : dimExprs) {
      auto pos = llvm::dyn_cast<AffineDimExpr>(dim).getPosition();
      memAffineIndices.push_back(oldAffineMap.getResult(pos));
    }
    auto newAffineMap =
        AffineMap::get(oldAffineMap.getNumDims(), 0 /* symbols */,
                       memAffineIndices, op->getContext());
    op->setAttr("map", AffineMapAttr::get(newAffineMap));
  }
}

LogicalResult runReform(func::FuncOp &f, ReformOp &reformOp, Value &array) {
  // 1) Get the schedule
  auto oldType = llvm::dyn_cast<MemRefType>(array.getType());
  auto oldShape = oldType.getShape();
  auto layoutMap =
      llvm::dyn_cast<AffineMapAttr>(reformOp->getAttr("layout")).getValue();

  // 2) Get new shape
  SmallVector<int64_t> newShape;
  SmallVector<AffineExpr> dimExprs;
  for (auto dim : layoutMap.getResults()) {
    newShape.push_back(
        oldShape[llvm::dyn_cast<AffineDimExpr>(dim).getPosition()]);
    dimExprs.push_back(dim);
  }

  // 3) Set new type
  mlir::Type elementType = oldType.getElementType();
  auto newType = MemRefType::get(newShape, elementType);
  array.setType(newType);

  // 4) Update memory access
  for (auto user : array.getUsers()) {
    updateMemrefAccess<AffineLoadOp>(user, dimExprs);
    updateMemrefAccess<AffineStoreOp>(user, dimExprs);
  }

  // 5) update function signature
  auto builder = Builder(array.getContext());
  auto resultTypes = f.front().getTerminator()->getOperandTypes();
  auto inputTypes = f.front().getArgumentTypes();
  f.setType(builder.getFunctionType(inputTypes, resultTypes));

  return success();
}

bool isAlloOp(Operation &op) {
  return llvm::isa<SplitOp, TileOp, ReorderOp, UnrollOp, UnfoldOp,
                   IntraKernelToOp, PipelineOp, ParallelOp, FuseOp, ComputeAtOp,
                   PartitionOp, ReuseAtOp, BufferAtOp, OutlineOp, ReshapeOp,
                   ReformOp, ThreadBindOp, InterKernelToOp, ReplaceOp>(op);
}

void eraseScheduleOp(func::FuncOp &f,
                     SmallVector<Operation *, 10> &opToRemove) {
  std::reverse(opToRemove.begin(), opToRemove.end());
  for (Operation *op : opToRemove) {
    op->erase();
  }
  SmallVector<Operation *, 10> handleToRemove;
  for (Operation &op : f.getOps()) {
    if (llvm::isa<allo::CreateLoopHandleOp, allo::CreateOpHandleOp>(op))
      handleToRemove.push_back(&op);
  }
  std::reverse(handleToRemove.begin(), handleToRemove.end());
  for (Operation *op : handleToRemove) {
    if (op->use_empty())
      op->erase();
  }
}

void applyCustomization(
    func::FuncOp &top_func,
    std::map<std::string, allo::CustomizationOp> &customizationMap,
    SmallVector<Operation *, 10> &opToRemove) {
  // skip if top_func has no body
  if (top_func.getBlocks().size() == 0)
    return;
  auto builder = OpBuilder::atBlockTerminator(&(top_func.getBody().front()));
  for (auto applyOp : top_func.getOps<allo::ApplyOp>()) {
    auto c = customizationMap[applyOp.getCallee().str()];
    DenseMap<Value, Value> arg2operand;
    for (auto item : llvm::enumerate(c.getArguments())) {
      arg2operand[item.value()] = applyOp.getOperand(item.index());
    }
    for (Operation &op : c.getOps()) {
      if (llvm::isa<allo::EndOp>(op))
        continue;
      mlir::IRMapping mapping;
      for (auto item : llvm::enumerate(op.getOperands())) {
        if (arg2operand.count(item.value()) > 0) {
          mapping.map(item.value(), arg2operand[item.value()]);
        }
      }
      builder.clone(op, mapping);
    }
    opToRemove.push_back(applyOp);
  }
  for (auto c : customizationMap) {
    opToRemove.push_back(c.second);
  }
}

bool applyLoopTransformationOnSingleFunction(
    ModuleOp &mod, func::FuncOp &f,
    std::map<std::string, allo::CustomizationOp> &customizationMap) {
  SmallVector<Operation *, 10> opToRemove;
  applyCustomization(f, customizationMap, opToRemove);
  // schedule should preverse orders, thus traverse one by one
  // the following shows the dispatching logic
  for (Operation &op : f.getOps()) {
    if (isAlloOp(op)) {
      if (auto new_op = dyn_cast<SplitOp>(op)) {
        if (failed(runSplitting(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<TileOp>(op)) {
        if (failed(runTiling(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<ReorderOp>(op)) {
        if (failed(runReordering(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<UnrollOp>(op)) {
        if (failed(runUnrolling(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<UnfoldOp>(op)) {
        if (failed(runUnfolding(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<IntraKernelToOp>(op)) {
        if (failed(runIntraKernelOpCheck(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<PipelineOp>(op)) {
        if (failed(runPipelining(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<ThreadBindOp>(op)) {
        if (failed(runThreadBind(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<ParallelOp>(op)) {
        if (failed(runParallel(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<FuseOp>(op)) {
        if (failed(runFusing(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<ComputeAtOp>(op)) {
        if (failed(runComputeAt(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<PartitionOp>(op)) {
        Value array;
        if (findArray(f, new_op.getTarget(), array)) {
          if (failed(runPartition(f, new_op, array)))
            return false;
        } else {
          return false;
        }
      } else if (auto new_op = dyn_cast<ReuseAtOp>(op)) {
        if (failed(runReuseAt(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<BufferAtOp>(op)) {
        if (failed(runBufferAt(f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<ReshapeOp>(op)) {
        Value array;
        if (findArray(f, new_op.getTarget(), array)) {
          if (failed(runReshape(f, new_op, array)))
            return false;
        } else {
          return false;
        }
      } else if (auto new_op = dyn_cast<ReformOp>(op)) {
        Value array;
        if (findArray(f, new_op.getTarget(), array)) {
          if (failed(runReform(f, new_op, array)))
            return false;
        } else {
          return false;
        }
      } else if (auto new_op = dyn_cast<InterKernelToOp>(op)) {
        Value array;
        auto optional_fifo_depth = new_op.getFifoDepth();
        unsigned int fifo_depth;
        if (optional_fifo_depth.has_value()) {
          fifo_depth = optional_fifo_depth.value();
        } else {
          fifo_depth = -1; // conservative assumption
        }
        if (findArray(f, new_op.getTarget(), array)) {
          if (failed(
                  runInterKernelDataPlacementSingleFunction(array, fifo_depth)))
            return false;
        } else {
          return false;
        }
      } else if (auto new_op = dyn_cast<OutlineOp>(op)) {
        if (failed(runOutline(mod, f, new_op)))
          return false;
      } else if (auto new_op = dyn_cast<ReplaceOp>(op)) {
        Value src, dst;
        if (findArray(f, new_op.getSrc(), src) &&
            findArray(f, new_op.getDst(), dst)) {
          if (failed(runReplaceOp(f, new_op, src, dst)))
            return false;
        } else {
          return false;
        }
      }
      opToRemove.push_back(&op);
    }
  }
  // remove schedule operations (from back to front) & legacy loop handles
  eraseScheduleOp(f, opToRemove);
  return true;
}

bool applyLoopTransformation(ModuleOp &mod) {
  std::map<std::string, allo::CustomizationOp> customizationMap;
  for (auto c : mod.getOps<allo::CustomizationOp>()) {
    customizationMap[c.getName().str()] = c;
  }
  // apply schedule
  for (func::FuncOp f : mod.getOps<func::FuncOp>()) {
    applyLoopTransformationOnSingleFunction(mod, f, customizationMap);
  }
  return true;
}

} // namespace allo
} // namespace mlir

namespace {

struct AlloLoopTransformation
    : public mlir::allo::impl::LoopTransformationBase<AlloLoopTransformation> {

  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLoopTransformation(mod))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace allo {

// Create A Loop Transformation Pass
std::unique_ptr<OperationPass<ModuleOp>> createLoopTransformationPass() {
  return std::make_unique<AlloLoopTransformation>();
}

} // namespace allo
} // namespace mlir
