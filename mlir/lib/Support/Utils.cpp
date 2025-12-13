/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modification: ScaleHLS
 * https://github.com/hanchenye/scalehls
 */

#include "allo/Support/Utils.h"
#include "allo/Dialect/AlloTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
using namespace mlir;
using namespace allo;

//===----------------------------------------------------------------------===//
// HLSCpp attribute utils
//===----------------------------------------------------------------------===//

/// Parse loop directives.
Attribute allo::getLoopDirective(Operation *op, std::string name) {
  return op->getAttr(name);
}

StringRef allo::getLoopName(AffineForOp &forOp) {
  if (forOp->hasAttr("loop_name"))
    return llvm::cast<StringAttr>(forOp->getAttr("loop_name")).getValue();
  else
    return "";
}

void allo::setLoopName(AffineForOp &forOp, std::string loop_name) {
  forOp->setAttr("loop_name", StringAttr::get(forOp->getContext(), loop_name));
}

void allo::setStageName(AffineForOp &forOp, StringRef op_name) {
  forOp->setAttr("op_name", StringAttr::get(forOp->getContext(), op_name));
}

std::vector<std::string> allo::split_names(const std::string &arg_names) {
  std::stringstream ss(arg_names);
  std::vector<std::string> args;
  while (ss.good()) {
    std::string substr;
    getline(ss, substr, ',');
    args.push_back(substr);
  }
  return args;
}

/// Parse other attributes.
SmallVector<int64_t, 8> allo::getIntArrayAttrValue(Operation *op,
                                                   StringRef name) {
  SmallVector<int64_t, 8> array;
  if (auto arrayAttr = op->getAttrOfType<ArrayAttr>(name)) {
    for (auto attr : arrayAttr)
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr))
        array.push_back(intAttr.getInt());
      else
        return SmallVector<int64_t, 8>();
    return array;
  } else
    return SmallVector<int64_t, 8>();
}

bool allo::setIntAttr(SmallVector<AffineForOp, 6> &forOps,
                      const SmallVector<int, 6> &attr_arr,
                      const std::string attr_name) {
  assert(forOps.size() == attr_arr.size());
  unsigned cnt_loop = 0;
  for (AffineForOp newForOp : forOps) {
    newForOp->setAttr(
        attr_name,
        IntegerAttr::get(
            IntegerType::get(newForOp->getContext(), 32,
                             IntegerType::SignednessSemantics::Signless),
            attr_arr[cnt_loop]));
    cnt_loop++;
  }
  return true;
}

bool allo::setLoopNames(SmallVector<AffineForOp, 6> &forOps,
                        const SmallVector<std::string, 6> &nameArr) {
  assert(forOps.size() == nameArr.size());
  unsigned cnt_loop = 0;
  for (AffineForOp newForOp : forOps) {
    newForOp->setAttr("loop_name", StringAttr::get(newForOp->getContext(),
                                                   nameArr[cnt_loop]));
    cnt_loop++;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Memory and loop analysis utils
//===----------------------------------------------------------------------===//

LogicalResult allo::getStage(func::FuncOp &func, AffineForOp &forOp,
                             StringRef op_name) {
  for (auto rootForOp : func.getOps<AffineForOp>()) {
    if (op_name ==
        llvm::dyn_cast<StringAttr>(rootForOp->getAttr("op_name")).getValue()) {
      forOp = rootForOp;
      return success();
    }
  }
  return failure();
}

void recursivelyFindLoop(AffineForOp forOp, int depth, StringRef loop_name,
                         AffineForOp &retForOp, int &retDepth,
                         SmallVector<AffineForOp> &loops);

void recursivelyFindLoopWithIf(AffineIfOp ifOp, int depth, StringRef loop_name,
                               AffineForOp &retForOp, int &retDepth,
                               SmallVector<AffineForOp> &loops) {
  for (auto nextForOp : ifOp.getThenBlock()->getOps<AffineForOp>())
    recursivelyFindLoop(nextForOp, depth + 1, loop_name, retForOp, retDepth,
                        loops);
  for (auto nextIfOp : ifOp.getThenBlock()->getOps<AffineIfOp>())
    recursivelyFindLoopWithIf(nextIfOp, depth, loop_name, retForOp, retDepth,
                              loops);
}

void recursivelyFindLoop(AffineForOp forOp, int depth, StringRef loop_name,
                         AffineForOp &retForOp, int &retDepth,
                         SmallVector<AffineForOp> &loops) {
  loops.push_back(forOp);
  if (getLoopName(forOp) == loop_name) {
    retForOp = forOp;
    retDepth = depth;
    return;
  }
  for (auto nextForOp : forOp.getOps<AffineForOp>())
    recursivelyFindLoop(nextForOp, depth + 1, loop_name, retForOp, retDepth,
                        loops);
  for (auto ifOp : forOp.getOps<AffineIfOp>())
    recursivelyFindLoopWithIf(ifOp, depth, loop_name, retForOp, retDepth,
                              loops);
}

int allo::getLoop(AffineForOp &forOp, StringRef loop_name) {
  // return the axis id
  AffineForOp currentLoop = forOp;
  int cnt = -1;
  SmallVector<AffineForOp> loops;
  recursivelyFindLoop(currentLoop, 0, loop_name, forOp, cnt, loops);
  return cnt;
}

void allo::getLoops(AffineForOp &forOp, SmallVector<AffineForOp> &forOpList) {
  int cnt = -1;
  recursivelyFindLoop(forOp, 0, "_placeholder_", forOp, cnt, forOpList);
}

bool allo::findContiguousNestedLoops(const AffineForOp &rootAffineForOp,
                                     SmallVector<AffineForOp, 6> &resForOps,
                                     SmallVector<StringRef, 6> &nameArr,
                                     int depth, bool countReductionLoops) {
  // depth = -1 means traverses all the inner loops
  AffineForOp forOp = rootAffineForOp;
  unsigned int sizeNameArr = nameArr.size();
  if (sizeNameArr != 0)
    depth = sizeNameArr;
  else if (depth == -1)
    depth = 0x3f3f3f3f;
  resForOps.clear();
  for (int i = 0; i < depth; ++i) {
    if (!forOp) {
      if (depth != 0x3f3f3f3f)
        return false;
      else // reach the inner-most loop
        return true;
    }

    Attribute attr = forOp->getAttr("loop_name");
    const StringRef curr_loop = llvm::dyn_cast<StringAttr>(attr).getValue();
    if (sizeNameArr != 0 && curr_loop != nameArr[i])
      return false;

    if (forOp->hasAttr("reduction") == 1 && !countReductionLoops) {
      i--;
    } else {
      resForOps.push_back(forOp);
      if (sizeNameArr == 0)
        nameArr.push_back(curr_loop);
    }
    Block &body = forOp.getRegion().front();
    // if (body.begin() != std::prev(body.end(), 2)) // perfectly nested
    //   break;

    forOp = dyn_cast<AffineForOp>(&body.front());
  }
  return true;
}

/// Collect all load and store operations in the block and return them in "map".
// void allo::getMemAccessesMap(Block &block, MemAccessesMap &map) {
//   for (auto &op : block) {
//     if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
//       map[MemRefAccess(&op).memref].push_back(&op);

//     else if (op.getNumRegions()) {
//       // Recursively collect memory access operations in each block.
//       for (auto &region : op.getRegions())
//         for (auto &block : region)
//           getMemAccessesMap(block, map);
//     }
//   }
// }

// Check if the lhsOp and rhsOp are in the same block. If so, return their
// ancestors that are located at the same block. Note that in this check,
// AffineIfOp is transparent.
std::optional<std::pair<Operation *, Operation *>>
allo::checkSameLevel(Operation *lhsOp, Operation *rhsOp) {
  // If lhsOp and rhsOp are already at the same level, return true.
  if (lhsOp->getBlock() == rhsOp->getBlock())
    return std::pair<Operation *, Operation *>(lhsOp, rhsOp);

  // Helper to get all surrounding AffineIfOps.
  auto getSurroundIfs =
      ([&](Operation *op, SmallVector<Operation *, 4> &nests) {
        nests.push_back(op);
        auto currentOp = op;
        while (true) {
          if (auto parentOp = currentOp->getParentOfType<AffineIfOp>()) {
            nests.push_back(parentOp);
            currentOp = parentOp;
          } else
            break;
        }
      });

  SmallVector<Operation *, 4> lhsNests;
  SmallVector<Operation *, 4> rhsNests;

  getSurroundIfs(lhsOp, lhsNests);
  getSurroundIfs(rhsOp, rhsNests);

  // If any parent of lhsOp and any parent of rhsOp are at the same level,
  // return true.
  for (auto lhs : lhsNests)
    for (auto rhs : rhsNests)
      if (lhs->getBlock() == rhs->getBlock())
        return std::pair<Operation *, Operation *>(lhs, rhs);

  return std::optional<std::pair<Operation *, Operation *>>();
}

/// Returns the number of surrounding loops common to 'loopsA' and 'loopsB',
/// where each lists loops from outer-most to inner-most in loop nest.
unsigned allo::getCommonSurroundingLoops(Operation *A, Operation *B,
                                         AffineLoopBand *band) {
  SmallVector<AffineForOp, 4> loopsA, loopsB;
  getAffineForIVs(*A, &loopsA);
  getAffineForIVs(*B, &loopsB);

  unsigned minNumLoops = std::min(loopsA.size(), loopsB.size());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (loopsA[i] != loopsB[i])
      break;
    ++numCommonLoops;
    if (band != nullptr)
      band->push_back(loopsB[i]);
  }
  return numCommonLoops;
}

/// Calculate the upper and lower bound of "bound" if possible.
std::optional<std::pair<int64_t, int64_t>>
allo::getBoundOfAffineBound(AffineBound bound) {
  auto boundMap = bound.getMap();
  if (boundMap.isSingleConstant()) {
    auto constBound = boundMap.getSingleConstantResult();
    return std::pair<int64_t, int64_t>(constBound, constBound);
  }

  // For now, we can only handle one result affine bound.
  if (boundMap.getNumResults() != 1)
    return std::optional<std::pair<int64_t, int64_t>>();

  auto context = boundMap.getContext();
  SmallVector<int64_t, 4> lbs;
  SmallVector<int64_t, 4> ubs;
  for (auto operand : bound.getOperands()) {
    // Only if the affine bound operands are induction variable, the calculation
    // is possible.
    if (!isAffineForInductionVar(operand))
      return std::optional<std::pair<int64_t, int64_t>>();

    // Only if the owner for op of the induction variable has constant bound,
    // the calculation is possible.
    auto ifOp = getForInductionVarOwner(operand);
    if (!ifOp.hasConstantBounds())
      return std::optional<std::pair<int64_t, int64_t>>();

    auto lb = ifOp.getConstantLowerBound();
    auto ub = ifOp.getConstantUpperBound();
    auto step = ifOp.getStepAsInt();

    lbs.push_back(lb);
    ubs.push_back(ub - 1 - (ub - 1 - lb) % step);
  }

  // TODO: maybe a more efficient algorithm.
  auto operandNum = bound.getNumOperands();
  SmallVector<int64_t, 16> results;
  for (unsigned i = 0, e = pow(2, operandNum); i < e; ++i) {
    SmallVector<AffineExpr, 4> replacements;
    for (unsigned pos = 0; pos < operandNum; ++pos) {
      if (i >> pos % 2 == 0)
        replacements.push_back(getAffineConstantExpr(lbs[pos], context));
      else
        replacements.push_back(getAffineConstantExpr(ubs[pos], context));
    }
    auto newExpr =
        bound.getMap().getResult(0).replaceDimsAndSymbols(replacements, {});

    if (auto constExpr = llvm::dyn_cast<AffineConstantExpr>(newExpr))
      results.push_back(constExpr.getValue());
    else
      return std::optional<std::pair<int64_t, int64_t>>();
  }

  auto minmax = std::minmax_element(results.begin(), results.end());
  return std::pair<int64_t, int64_t>(*minmax.first, *minmax.second);
}

/// Return the layout map of "memrefType".
AffineMap allo::getLayoutMap(MemRefType memrefType) {
  // Check whether the memref has layout map.
  auto memrefMaps = memrefType.getLayout();
  if (memrefMaps.getAffineMap().isIdentity())
    return (AffineMap) nullptr;

  return memrefMaps.getAffineMap();
}

bool allo::isFullyPartitioned(MemRefType memrefType, int axis) {
  if (memrefType.getRank() == 0)
    return true;

  bool fullyPartitioned = false;
  if (auto layoutMap = getLayoutMap(memrefType)) {
    SmallVector<int64_t, 8> factors;
    getPartitionFactors(memrefType, &factors);

    // Case 1: Use floordiv & mod
    auto shapes = memrefType.getShape();
    if (axis == -1) // all the dimensions
      fullyPartitioned =
          factors == SmallVector<int64_t, 8>(shapes.begin(), shapes.end());
    else
      fullyPartitioned = factors[axis] == shapes[axis];

    // Case 2: Partition index is an identity function
    if (axis == -1) {
      bool flag = true;
      for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
        auto expr = layoutMap.getResult(dim);
        if (!llvm::isa<AffineDimExpr>(expr)) {
          flag = false;
          break;
        }
      }
      fullyPartitioned |= flag;
    } else {
      auto expr = layoutMap.getResult(axis);
      fullyPartitioned |= llvm::isa<AffineDimExpr>(expr);
    }
  }

  return fullyPartitioned;
}

// Calculate partition factors through analyzing the "memrefType" and return
// them in "factors". Meanwhile, the overall partition number is calculated and
// returned as well.
int64_t allo::getPartitionFactors(MemRefType memrefType,
                                  SmallVector<int64_t, 8> *factors) {
  auto shape = memrefType.getShape();
  auto layoutMap = getLayoutMap(memrefType);
  int64_t accumFactor = 1;

  for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
    int64_t factor = 1;

    if (layoutMap) {
      auto expr = layoutMap.getResult(dim);

      if (auto binaryExpr = llvm::dyn_cast<AffineBinaryOpExpr>(expr))
        if (auto rhsExpr =
                llvm::dyn_cast<AffineConstantExpr>(binaryExpr.getRHS())) {
          if (expr.getKind() == AffineExprKind::Mod)
            factor = rhsExpr.getValue();
          else if (expr.getKind() == AffineExprKind::FloorDiv)
            factor = (shape[dim] + rhsExpr.getValue() - 1) / rhsExpr.getValue();
        }
    }

    accumFactor *= factor;
    if (factors != nullptr)
      factors->push_back(factor);
  }

  return accumFactor;
}

/// This is method for finding the number of child loops which immediatedly
/// contained by the input operation.
static unsigned getChildLoopNum(Operation *op) {
  unsigned childNum = 0;
  for (auto &region : op->getRegions())
    for (auto &block : region)
      for (auto &op : block)
        if (isa<AffineForOp>(op))
          ++childNum;

  return childNum;
}

/// Get the whole loop band given the innermost loop and return it in "band".
static void getLoopBandFromInnermost(AffineForOp forOp, AffineLoopBand &band) {
  band.clear();
  AffineLoopBand reverseBand;

  auto currentLoop = forOp;
  while (true) {
    reverseBand.push_back(currentLoop);

    auto parentLoop = currentLoop->getParentOfType<AffineForOp>();
    if (!parentLoop)
      break;

    if (getChildLoopNum(parentLoop) == 1)
      currentLoop = parentLoop;
    else
      break;
  }

  band.append(reverseBand.rbegin(), reverseBand.rend());
}

/// Get the whole loop band given the outermost loop and return it in "band".
/// Meanwhile, the return value is the innermost loop of this loop band.
AffineForOp allo::getLoopBandFromOutermost(AffineForOp forOp,
                                           AffineLoopBand &band) {
  band.clear();
  auto currentLoop = forOp;
  while (true) {
    band.push_back(currentLoop);

    if (getChildLoopNum(currentLoop) == 1)
      currentLoop = *currentLoop.getOps<AffineForOp>().begin();
    else
      break;
  }
  return band.back();
}

/// Collect all loop bands in the "block" and return them in "bands". If
/// "allowHavingChilds" is true, loop bands containing more than 1 other loop
/// bands are also collected. Otherwise, only loop bands that contains no child
/// loops are collected.
void allo::getLoopBands(Block &block, AffineLoopBands &bands,
                        bool allowHavingChilds) {
  bands.clear();
  block.walk([&](AffineForOp loop) {
    auto childNum = getChildLoopNum(loop);

    if (childNum == 0 || (childNum > 1 && allowHavingChilds)) {
      AffineLoopBand band;
      getLoopBandFromInnermost(loop, band);
      bands.push_back(band);
    }
  });
}

void allo::getArrays(Block &block, SmallVectorImpl<Value> &arrays,
                     bool allowArguments) {
  // Collect argument arrays.
  if (allowArguments)
    for (auto arg : block.getArguments()) {
      if (llvm::isa<MemRefType>(arg.getType()))
        arrays.push_back(arg);
    }

  // Collect local arrays.
  for (auto &op : block.getOperations()) {
    if (llvm::isa<memref::AllocaOp, memref::AllocOp>(op))
      arrays.push_back(op.getResult(0));
  }
}

std::optional<unsigned> allo::getAverageTripCount(AffineForOp forOp) {
  if (auto optionalTripCount = getConstantTripCount(forOp))
    return optionalTripCount.value();
  else {
    // TODO: A temporary approach to estimate the trip count. For now, we take
    // the average of the upper bound and lower bound of trip count as the
    // estimated trip count.
    auto lowerBound = getBoundOfAffineBound(forOp.getLowerBound());
    auto upperBound = getBoundOfAffineBound(forOp.getUpperBound());

    if (lowerBound && upperBound) {
      auto lowerTripCount =
          upperBound.value().second - lowerBound.value().first;
      auto upperTripCount =
          upperBound.value().first - lowerBound.value().second;
      return (lowerTripCount + upperTripCount + 1) / 2;
    } else
      return std::optional<unsigned>();
  }
}

bool allo::checkDependence(Operation *A, Operation *B) {
  return true;
  // TODO: Fix the following
  //   AffineLoopBand commonLoops;
  //   unsigned numCommonLoops = getCommonSurroundingLoops(A, B, &commonLoops);

  //   // Traverse each loop level to find dependencies.
  //   for (unsigned depth = numCommonLoops; depth > 0; depth--) {
  //     // Skip all parallel loop level.
  //     if (auto loopAttr = getLoopDirective(commonLoops[depth - 1]))
  //       if (loopAttr.getParallel())
  //         continue;

  //     FlatAffineValueConstraints depConstrs;
  //     DependenceResult result = checkMemrefAccessDependence(
  //         MemRefAccess(A), MemRefAccess(B), depth, &depConstrs,
  //         /*dependenceComponents=*/nullptr);
  //     if (hasDependence(result))
  //       return true;
  //   }

  //   return false;
}

static bool gatherLoadOpsAndStoreOps(AffineForOp forOp,
                                     SmallVectorImpl<Operation *> &loadOps,
                                     SmallVectorImpl<Operation *> &storeOps) {
  bool hasIfOp = false;
  forOp.walk([&](Operation *op) {
    if (auto load = dyn_cast<AffineReadOpInterface>(op))
      loadOps.push_back(op);
    else if (auto load = dyn_cast<memref::LoadOp>(op))
      loadOps.push_back(op);
    else if (auto store = dyn_cast<AffineWriteOpInterface>(op))
      storeOps.push_back(op);
    else if (auto store = dyn_cast<memref::StoreOp>(op))
      storeOps.push_back(op);
    else if (llvm::isa<AffineIfOp>(op))
      hasIfOp = true;
  });
  return !hasIfOp;
}

bool allo::analyzeDependency(const AffineForOp &forOpA,
                             const AffineForOp &forOpB,
                             SmallVectorImpl<Dependency> &dependency) {
  SmallVector<Operation *, 4> readOpsA;
  SmallVector<Operation *, 4> writeOpsA;
  SmallVector<Operation *, 4> readOpsB;
  SmallVector<Operation *, 4> writeOpsB;

  if (!gatherLoadOpsAndStoreOps(forOpA, readOpsA, writeOpsA)) {
    return false;
  }

  if (!gatherLoadOpsAndStoreOps(forOpB, readOpsB, writeOpsB)) {
    return false;
  }

  DenseSet<Value> OpAReadMemRefs;
  DenseSet<Value> OpAWriteMemRefs;
  DenseSet<Value> OpBReadMemRefs;
  DenseSet<Value> OpBWriteMemRefs;

  for (Operation *op : readOpsA) {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      OpAReadMemRefs.insert(loadOp.getMemRef());
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      OpAReadMemRefs.insert(loadOp.getMemRef());
    }
  }

  for (Operation *op : writeOpsA) {
    if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      OpAWriteMemRefs.insert(storeOp.getMemRef());
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      OpAWriteMemRefs.insert(storeOp.getMemRef());
    }
  }

  for (Operation *op : readOpsB) {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      OpBReadMemRefs.insert(loadOp.getMemRef());
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      OpBReadMemRefs.insert(loadOp.getMemRef());
    }
  }

  for (Operation *op : writeOpsB) {
    if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      OpBWriteMemRefs.insert(storeOp.getMemRef());
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      OpBWriteMemRefs.insert(storeOp.getMemRef());
    }
  }

  for (Value memref : OpBReadMemRefs) {
    if (OpAWriteMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::RAW);
    else if (OpAReadMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::RAR);
  }

  for (Value memref : OpBWriteMemRefs) {
    if (OpAWriteMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::WAW);
    else if (OpAReadMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::WAR);
  }

  return true;
}

//===----------------------------------------------------------------------===//
// PtrLikeMemRefAccess Struct Definition
//===----------------------------------------------------------------------===//

PtrLikeMemRefAccess::PtrLikeMemRefAccess(Operation *loadOrStoreOpInst) {
  Operation *opInst = nullptr;
  SmallVector<Value, 4> indices;

  if (auto loadOp = dyn_cast<AffineReadOpInterface>(loadOrStoreOpInst)) {
    memref = loadOp.getMemRef();
    opInst = loadOrStoreOpInst;
    auto loadMemrefType = loadOp.getMemRefType();

    indices.reserve(loadMemrefType.getRank());
    for (auto index : loadOp.getMapOperands()) {
      indices.push_back(index);
    }
  } else {
    assert(isa<AffineWriteOpInterface>(loadOrStoreOpInst) &&
           "Affine read/write op expected");
    auto storeOp = cast<AffineWriteOpInterface>(loadOrStoreOpInst);
    opInst = loadOrStoreOpInst;
    memref = storeOp.getMemRef();
    auto storeMemrefType = storeOp.getMemRefType();

    indices.reserve(storeMemrefType.getRank());
    for (auto index : storeOp.getMapOperands()) {
      indices.push_back(index);
    }
  }

  // Get affine map from AffineLoad/Store.
  AffineMap map;
  if (auto loadOp = dyn_cast<AffineReadOpInterface>(opInst))
    map = loadOp.getAffineMap();
  else
    map = cast<AffineWriteOpInterface>(opInst).getAffineMap();

  SmallVector<Value, 8> operands(indices.begin(), indices.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);

  accessMap.reset(map, operands);
}

bool PtrLikeMemRefAccess::operator==(const PtrLikeMemRefAccess &rhs) const {
  if (memref != rhs.memref || impl != rhs.impl)
    return false;

  if (impl == rhs.impl && impl && rhs.impl)
    return true;

  AffineValueMap diff;
  AffineValueMap::difference(accessMap, rhs.accessMap, &diff);
  return llvm::all_of(diff.getAffineMap().getResults(),
                      [](AffineExpr e) { return e == 0; });
}

// Returns the index of 'op' in its block.
inline static unsigned getBlockIndex(Operation &op) {
  unsigned index = 0;
  for (auto &opX : *op.getBlock()) {
    if (&op == &opX)
      break;
    ++index;
  }
  return index;
}

// Returns a string representation of 'sliceUnion'.
std::string
allo::getSliceStr(const mlir::affine::ComputationSliceState &sliceUnion) {
  std::string result;
  llvm::raw_string_ostream os(result);
  // Slice insertion point format [loop-depth, operation-block-index]
  unsigned ipd = mlir::affine::getNestingDepth(&*sliceUnion.insertPoint);
  unsigned ipb = getBlockIndex(*sliceUnion.insertPoint);
  os << "insert point: (" << std::to_string(ipd) << ", " << std::to_string(ipb)
     << ")";
  assert(sliceUnion.lbs.size() == sliceUnion.ubs.size());
  os << " loop bounds: ";
  for (unsigned k = 0, e = sliceUnion.lbs.size(); k < e; ++k) {
    os << '[';
    sliceUnion.lbs[k].print(os);
    os << ", ";
    sliceUnion.ubs[k].print(os);
    os << "] ";
  }
  return os.str();
}

Value allo::castInteger(OpBuilder builder, Location loc, Value input,
                        Type srcType, Type tgtType, bool is_signed) {
  int oldWidth = llvm::dyn_cast<IntegerType>(srcType).getWidth();
  int newWidth = llvm::dyn_cast<IntegerType>(tgtType).getWidth();
  Value casted;
  if (newWidth < oldWidth) {
    // trunc
    casted = builder.create<arith::TruncIOp>(loc, tgtType, input);
  } else if (newWidth > oldWidth) {
    // extend
    if (is_signed) {
      casted = builder.create<arith::ExtSIOp>(loc, tgtType, input);
    } else {
      casted = builder.create<arith::ExtUIOp>(loc, tgtType, input);
    }
  } else {
    casted = input;
  }
  return casted;
}

/* CastIntMemRef
 * Allocate a new Int MemRef of target width and build a
 * AffineForOp loop nest to load, cast, store the elements
 * from oldMemRef to newMemRef.
 */
Value allo::castIntMemRef(OpBuilder &builder, Location loc, Value &oldMemRef,
                          size_t newWidth, bool unsign, bool replace,
                          const Value &dstMemRef) {
  // If newWidth == oldWidth, no need to cast.
  if (newWidth ==
      llvm::dyn_cast<IntegerType>(
          llvm::dyn_cast<MemRefType>(oldMemRef.getType()).getElementType())
          .getWidth()) {
    return oldMemRef;
  }
  // first, alloc new memref
  MemRefType oldMemRefType = llvm::dyn_cast<MemRefType>(oldMemRef.getType());
  Type newElementType = builder.getIntegerType(newWidth);
  MemRefType newMemRefType =
      llvm::dyn_cast<MemRefType>(oldMemRefType.clone(newElementType));
  Value newMemRef;
  if (!dstMemRef) {
    newMemRef = builder.create<memref::AllocOp>(loc, newMemRefType);
  }
  // replace all uses
  if (replace)
    oldMemRef.replaceAllUsesWith(newMemRef);
  // build loop nest
  SmallVector<int64_t, 4> lbs(oldMemRefType.getRank(), 0);
  SmallVector<int64_t, 4> steps(oldMemRefType.getRank(), 1);
  size_t oldWidth =
      llvm::dyn_cast<IntegerType>(
          llvm::dyn_cast<MemRefType>(oldMemRef.getType()).getElementType())
          .getWidth();
  buildAffineLoopNest(
      builder, loc, lbs, oldMemRefType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        Value v = nestedBuilder.create<AffineLoadOp>(loc, oldMemRef, ivs);
        Value casted;
        if (newWidth < oldWidth) {
          // trunc
          casted =
              nestedBuilder.create<arith::TruncIOp>(loc, newElementType, v);
        } else if (newWidth > oldWidth) {
          // extend
          if (unsign) {
            casted =
                nestedBuilder.create<arith::ExtUIOp>(loc, newElementType, v);
          } else {
            casted =
                nestedBuilder.create<arith::ExtSIOp>(loc, newElementType, v);
          }
        } else {
          casted = v; // no cast happened
        }
        if (dstMemRef) {
          nestedBuilder.create<AffineStoreOp>(loc, casted, dstMemRef, ivs);
        } else {
          nestedBuilder.create<AffineStoreOp>(loc, casted, newMemRef, ivs);
        }
      });
  return newMemRef;
}

bool mlir::allo::replace(std::string &str, const std::string &from,
                         const std::string &to) {
  size_t start_pos = str.find(from);
  if (start_pos == std::string::npos)
    return false;
  str.replace(start_pos, from.length(), to);
  return true;
}

Value mlir::allo::castToF64(OpBuilder &rewriter, const Value &src,
                            bool hasUnsignedAttr) {
  Type t = src.getType();
  Type I64 = rewriter.getIntegerType(64);
  Type F64 = rewriter.getF64Type();
  Value casted;
  if (llvm::isa<IndexType>(t)) {
    Type I32 = rewriter.getIntegerType(32);
    Value intValue =
        rewriter.create<arith::IndexCastOp>(src.getLoc(), I32, src);
    return castToF64(rewriter, intValue, hasUnsignedAttr);
  } else if (llvm::isa<IntegerType>(t)) {
    size_t iwidth = t.getIntOrFloatBitWidth();
    if (t.isUnsignedInteger() or hasUnsignedAttr) {
      Value widthAdjusted;
      if (iwidth < 64) {
        widthAdjusted = rewriter.create<arith::ExtUIOp>(src.getLoc(), I64, src);
      } else if (iwidth > 64) {
        widthAdjusted =
            rewriter.create<arith::TruncIOp>(src.getLoc(), I64, src);
      } else {
        widthAdjusted = src;
      }
      casted =
          rewriter.create<arith::UIToFPOp>(src.getLoc(), F64, widthAdjusted);
    } else { // signed and signless integer
      Value widthAdjusted;
      if (iwidth < 64) {
        widthAdjusted = rewriter.create<arith::ExtSIOp>(src.getLoc(), I64, src);
      } else if (iwidth > 64) {
        widthAdjusted =
            rewriter.create<arith::TruncIOp>(src.getLoc(), I64, src);
      } else {
        widthAdjusted = src;
      }
      casted =
          rewriter.create<arith::SIToFPOp>(src.getLoc(), F64, widthAdjusted);
    }
  } else if (llvm::isa<FloatType>(t)) {
    unsigned width = llvm::dyn_cast<FloatType>(t).getWidth();
    if (width < 64) {
      casted = rewriter.create<arith::ExtFOp>(src.getLoc(), F64, src);
    } else if (width > 64) {
      casted = rewriter.create<arith::TruncFOp>(src.getLoc(), F64, src);
    } else {
      casted = src;
    }
  } else if (llvm::isa<FixedType>(t)) {
    unsigned width = llvm::dyn_cast<FixedType>(t).getWidth();
    unsigned frac = llvm::dyn_cast<FixedType>(t).getFrac();
    Value widthAdjusted;
    if (width < 64) {
      widthAdjusted = rewriter.create<arith::ExtSIOp>(src.getLoc(), I64, src);
    } else if (width > 64) {
      widthAdjusted = rewriter.create<arith::TruncIOp>(src.getLoc(), I64, src);
    } else {
      widthAdjusted = src;
    }
    Value srcF64 =
        rewriter.create<arith::SIToFPOp>(src.getLoc(), F64, widthAdjusted);
    Value const_frac = rewriter.create<arith::ConstantOp>(
        src.getLoc(), F64, rewriter.getFloatAttr(F64, std::pow(2, frac)));
    casted =
        rewriter.create<arith::DivFOp>(src.getLoc(), F64, srcF64, const_frac);
  } else if (llvm::isa<UFixedType>(t)) {
    unsigned width = llvm::dyn_cast<UFixedType>(t).getWidth();
    unsigned frac = llvm::dyn_cast<UFixedType>(t).getFrac();
    Value widthAdjusted;
    if (width < 64) {
      widthAdjusted = rewriter.create<arith::ExtUIOp>(src.getLoc(), I64, src);
    } else if (width > 64) {
      widthAdjusted = rewriter.create<arith::TruncIOp>(src.getLoc(), I64, src);
    } else {
      widthAdjusted = src;
    }
    Value srcF64 =
        rewriter.create<arith::UIToFPOp>(src.getLoc(), F64, widthAdjusted);
    Value const_frac = rewriter.create<arith::ConstantOp>(
        src.getLoc(), F64, rewriter.getFloatAttr(F64, std::pow(2, frac)));
    casted =
        rewriter.create<arith::DivFOp>(src.getLoc(), F64, srcF64, const_frac);
  } else {
    llvm::errs() << src.getLoc() << "could not cast value of type "
                 << src.getType() << " to F64.\n";
  }
  return casted;
}

bool mlir::allo::getEnv(const std::string &key, std::string &value) {
  char *env = std::getenv(key.c_str());
  if (env) {
    value = env;
    return true;
  }
  return false;
}

int mlir::allo::getIndex(SmallVector<Operation *, 4> v, Operation *target) {
  auto it = std::find(v.begin(), v.end(), target);

  // If element was found
  if (it != v.end()) {
    int index = it - v.begin();
    return index;
  } else {
    // If the element is not
    // present in the vector
    return -1;
  }
}

bool chaseAffineApply(Value iv, Value target) {
  for (auto &use : iv.getUses()) {
    auto op = use.getOwner();
    if (dyn_cast<AffineApplyOp>(op)) {
      if (op->getResult(0) == target) {
        return true;
      } else {
        return chaseAffineApply(op->getResult(0), target);
      }
    } else {
      continue;
    }
  }
  return false;
};

// Find the which dimension of affine.store the
// loop induction variable operates on.
// e.g.
// for %i = 0; %i < 10; %i++
//   for %j = 0; %j < 10; %j++
//      %ii = affine.apply(%i) #some_map
//      affine.store %some_value %some_memref[%ii, %j]
// If we want to find the memref axis of %some_memref that
// %i operates on, the return result is 0.
int mlir::allo::findMemRefAxisFromIV(AffineStoreOp store, Value iv) {
  auto memrefRank =
      llvm::dyn_cast<MemRefType>(store.getMemRef().getType()).getRank();
  auto indices = store.getIndices();
  for (int i = 0; i < memrefRank; i++) {
    if (iv == indices[i]) {
      // if it is a direct match
      return i;
    } else {
      // try to chase down the affine.apply op
      // see if any result of the affine.apply op
      // matches indices[i].
      // This essentially is a DFS search.
      if (chaseAffineApply(iv, indices[i])) {
        return i;
      }
    }
  }
  return -1;
}
