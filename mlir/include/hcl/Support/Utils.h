/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modification: ScaleHLS
 * https://github.com/hanchenye/scalehls
 */

#ifndef HCL_ANALYSIS_UTILS_H
#define HCL_ANALYSIS_UTILS_H

#include "hcl/Dialect/HeteroCLDialect.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir::affine;

namespace mlir {
namespace hcl {

//===----------------------------------------------------------------------===//
// HLSCpp attribute parsing utils
//===----------------------------------------------------------------------===//

/// Parse loop directives.
Attribute getLoopDirective(Operation *op, std::string name);

StringRef getLoopName(AffineForOp &forOp);

void setLoopName(AffineForOp &forOp, std::string loop_name);
void setStageName(AffineForOp &forOp, StringRef op_name);

/// Parse other attributes.
SmallVector<int64_t, 8> getIntArrayAttrValue(Operation *op, StringRef name);

bool setLoopNames(SmallVector<AffineForOp, 6> &forOps,
                  const SmallVector<std::string, 6> &nameArr);
bool setIntAttr(SmallVector<AffineForOp, 6> &forOps,
                const SmallVector<int, 6> &attr_arr, std::string attr_name);

std::vector<std::string> split_names(const std::string &arg_names);

//===----------------------------------------------------------------------===//
// Memory and loop analysis utils
//===----------------------------------------------------------------------===//

using AffineLoopBand = SmallVector<AffineForOp, 6>;
using AffineLoopBands = std::vector<AffineLoopBand>;

/// For storing all affine memory access operations (including AffineLoadOp, and
/// AffineStoreOp) indexed by the corresponding memref.
using MemAccessesMap = DenseMap<Value, SmallVector<Operation *, 16>>;

LogicalResult getStage(func::FuncOp &func, AffineForOp &forOp,
                       StringRef stage_name);

int getLoop(AffineForOp &forOp, StringRef loop_name);

void getLoops(AffineForOp &forOp, SmallVector<AffineForOp> &forOpList);

bool findContiguousNestedLoops(const AffineForOp &rootAffineForOp,
                               SmallVector<AffineForOp, 6> &resForOps,
                               SmallVector<StringRef, 6> &nameArr,
                               int depth = -1, bool countReductionLoops = true);

/// Collect all load and store operations in the block and return them in "map".
// void getMemAccessesMap(Block &block, MemAccessesMap &map);

/// Check if the lhsOp and rhsOp are in the same block. If so, return their
/// ancestors that are located at the same block. Note that in this check,
/// AffineIfOp is transparent.
std::optional<std::pair<Operation *, Operation *>> checkSameLevel(Operation *lhsOp,
                                                             Operation *rhsOp);

unsigned getCommonSurroundingLoops(Operation *A, Operation *B,
                                   AffineLoopBand *band);

/// Calculate the upper and lower bound of "bound" if possible.
std::optional<std::pair<int64_t, int64_t>> getBoundOfAffineBound(AffineBound bound);

/// Return the layout map of "memrefType".
AffineMap getLayoutMap(MemRefType memrefType);

/// Calculate partition factors through analyzing the "memrefType" and return
/// them in "factors". Meanwhile, the overall partition number is calculated and
/// returned as well.
int64_t getPartitionFactors(MemRefType memrefType,
                            SmallVector<int64_t, 8> *factors = nullptr);

bool isFullyPartitioned(MemRefType memrefType, int axis = -1);

/// Get the whole loop band given the outermost loop and return it in "band".
/// Meanwhile, the return value is the innermost loop of this loop band.
AffineForOp getLoopBandFromOutermost(AffineForOp forOp, AffineLoopBand &band);

/// Collect all loop bands in the "block" and return them in "bands". If
/// "allowHavingChilds" is true, loop bands containing more than 1 other loop
/// bands are also collected. Otherwise, only loop bands that contains no child
/// loops are collected.
void getLoopBands(Block &block, AffineLoopBands &bands,
                  bool allowHavingChilds = false);

void getArrays(Block &block, SmallVectorImpl<Value> &arrays,
               bool allowArguments = true);

std::optional<unsigned> getAverageTripCount(AffineForOp forOp);

bool checkDependence(Operation *A, Operation *B);

// Returns a string representation of 'sliceUnion'.
std::string getSliceStr(const mlir::affine::ComputationSliceState &sliceUnion);

// Dependency types between two stages
enum Dependency { RAW, RAR, WAR, WAW };

bool analyzeDependency(const AffineForOp &forOpA, const AffineForOp &forOpB,
                       SmallVectorImpl<Dependency> &dependency);

int findMemRefAxisFromIV(AffineStoreOp storeOp, Value iv);

//===----------------------------------------------------------------------===//
// PtrLikeMemRefAccess Struct Declaration
//===----------------------------------------------------------------------===//

/// Encapsulates a memref load or store access information.
struct PtrLikeMemRefAccess {
  Value memref = nullptr;
  AffineValueMap accessMap;

  void *impl = nullptr;

  /// Constructs a MemRefAccess from a load or store operation.
  explicit PtrLikeMemRefAccess(Operation *opInst);

  PtrLikeMemRefAccess(const void *impl) : impl(const_cast<void *>(impl)) {}

  bool operator==(const PtrLikeMemRefAccess &rhs) const;

  llvm::hash_code getHashValue() {
    return llvm::hash_combine(memref, accessMap.getAffineMap(),
                              accessMap.getOperands(), impl);
  }
};

using ReverseOpIteratorsMap =
    DenseMap<PtrLikeMemRefAccess,
             SmallVector<std::reverse_iterator<Operation **>, 16>>;
using OpIteratorsMap =
    DenseMap<PtrLikeMemRefAccess, SmallVector<Operation **, 16>>;

} // namespace hcl
} // namespace mlir

//===----------------------------------------------------------------------===//
// Make PtrLikeMemRefAccess eligible as key of DenseMap
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct DenseMapInfo<mlir::hcl::PtrLikeMemRefAccess> {
  static mlir::hcl::PtrLikeMemRefAccess getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::hcl::PtrLikeMemRefAccess(pointer);
  }
  static mlir::hcl::PtrLikeMemRefAccess getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::hcl::PtrLikeMemRefAccess(pointer);
  }
  static unsigned getHashValue(mlir::hcl::PtrLikeMemRefAccess access) {
    return access.getHashValue();
  }
  static bool isEqual(mlir::hcl::PtrLikeMemRefAccess lhs,
                      mlir::hcl::PtrLikeMemRefAccess rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Cast utilities
//===----------------------------------------------------------------------===//
namespace mlir {
namespace hcl {

Value castInteger(OpBuilder builder, Location loc, Value input, Type srcType,
                  Type tgtType, bool is_signed);
Value castIntMemRef(OpBuilder &builder, Location loc, const Value &oldMemRef,
                    size_t newWidth, bool unsign = false, bool replace = true,
                    const Value &dstMemRef = NULL);
Value castToF64(OpBuilder &rewriter, const Value &src, bool hasUnsignedAttr);

} // namespace hcl
} // namespace mlir

//===----------------------------------------------------------------------===//
// String utils
//===----------------------------------------------------------------------===//
namespace mlir {
namespace hcl {
bool replace(std::string& str, const std::string& from, const std::string& to);
bool getEnv(const std::string &key, std::string &value);

} // namespace hcl
} // namespace mlir


//===----------------------------------------------------------------------===//
// SmallVector utils
//===----------------------------------------------------------------------===//
namespace mlir{
namespace hcl{
// TODO(Niansong): make this template function
int getIndex(SmallVector<Operation*, 4> v, Operation* target);
} // namespace hcl
} // namespace mlir

#endif // HCL_ANALYSIS_UTILS_H
