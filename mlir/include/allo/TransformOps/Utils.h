/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSFORM_OPS_UTILS_H
#define ALLO_TRANSFORM_OPS_UTILS_H

#include "allo/Dialect/AlloOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::allo {
bool affineExprUsesValue(AffineExpr expr, ValueRange mapOperands,
                         unsigned numDims, Value needle);
int findMemRefAxisFromIVs(affine::AffineStoreOp storeOp, Value iv);
Value resolveMemRefValueRoot(Value value);
// strip away index casts, extension/truncation ops,
// which do not affect the value as an affine expression
Value stripCast(Value value);
bool isMemRefCastOrViewLike(Operation *op);

struct AffineValueMapBuilder {
  MLIRContext *ctx;
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> syms;
  SmallVector<Value, 4> results;
  llvm::SmallDenseSet<Value, 4> exprFailureCache;
  SmallVector<AffineExpr, 4> exprs;

  explicit AffineValueMapBuilder(MLIRContext *ctx) : ctx(ctx) {}

  // used to import a single value as an affine expression
  LogicalResult importValue(Value v) {
    auto result = importValueInternal(v);
    if (failed(result))
      return failure();
    exprs.push_back(*result);
    return success();
  }
  // used to import an affine map and its operands
  // if allowMultiResults is false, the map must have exactly one result
  LogicalResult importMapAndOperands(AffineMap map, ValueRange dims,
                                     ValueRange syms,
                                     bool allowMultiResults = false);
  // compose the imported expressions and simplify the resulting map
  affine::AffineValueMap compose() const;
  // reset internal state to reuse the builder for another map
  // it does not clear the failure cache,
  // since it's used to accelerate repeated failed import attempts on the same
  // values.
  void reset();
  // add results to the final value map
  // optional if only cares about how to compose the results.
  void addResults(ArrayRef<Value> results) {
    llvm::append_range(this->results, results);
  }

private:
  LogicalResult cacheFailure(Value v) {
    exprFailureCache.insert(v);
    return failure();
  }
  FailureOr<AffineExpr> importValueInternal(Value v);
  AffineExpr addDim(Value v);
  AffineExpr addSym(Value v);
};
} // namespace mlir::allo

#endif // ALLO_TRANSFORM_OPS_UTILS_H
