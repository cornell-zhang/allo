/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Transforms/Passes.h"

#include "mlir/CAPI/IR.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/InitAllDialects.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

bool compareAffineExprs(AffineExpr lhsExpr, AffineExpr rhsExpr) {
  // Compare the kinds of affine exprs
  if (lhsExpr.getKind() != rhsExpr.getKind()) {
    return false;
  }

  // Compare affine exprs based on kind
  switch (lhsExpr.getKind()) {
  case AffineExprKind::Constant: {
    auto lhsConst = llvm::cast<AffineConstantExpr>(lhsExpr);
    auto rhsConst = llvm::cast<AffineConstantExpr>(rhsExpr);
    return lhsConst.getValue() == rhsConst.getValue();
  }
  case AffineExprKind::DimId: {
    auto lhsDim = llvm::cast<AffineDimExpr>(lhsExpr);
    auto rhsDim = llvm::cast<AffineDimExpr>(rhsExpr);
    return lhsDim.getPosition() == rhsDim.getPosition();
  }
  case AffineExprKind::SymbolId: {
    auto lhsSymbol = llvm::cast<AffineSymbolExpr>(lhsExpr);
    auto rhsSymbol = llvm::cast<AffineSymbolExpr>(rhsExpr);
    return lhsSymbol.getPosition() == rhsSymbol.getPosition();
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::Mod:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    auto lhsBinary = llvm::cast<AffineBinaryOpExpr>(lhsExpr);
    auto rhsBinary = llvm::cast<AffineBinaryOpExpr>(rhsExpr);
    return compareAffineExprs(lhsBinary.getLHS(), rhsBinary.getLHS()) &&
           compareAffineExprs(lhsBinary.getRHS(), rhsBinary.getRHS());
  }
  }
  return false;
}

bool compareAffineMaps(AffineMap lhsMap, AffineMap rhsMap) {
  AffineMap simplifiedLhsMap = simplifyAffineMap(lhsMap);
  AffineMap simplifiedRhsMap = simplifyAffineMap(rhsMap);

  if (simplifiedLhsMap.getNumDims() != simplifiedRhsMap.getNumDims() &&
      simplifiedLhsMap.getNumSymbols() != simplifiedRhsMap.getNumSymbols() &&
      simplifiedLhsMap.getNumResults() != simplifiedRhsMap.getNumResults())
    return false;

  // Compare exprs
  for (unsigned i = 0; i < simplifiedLhsMap.getNumResults(); ++i) {
    if (!compareAffineExprs(simplifiedLhsMap.getResult(i),
                            simplifiedRhsMap.getResult(i))) {
      return false;
    }
  }

  // Todo: Might need to compare operands or use evaluation to compare
  return true;
}

bool compareAffineForOps(affine::AffineForOp &affineForOp1,
                         affine::AffineForOp &affineForOp2) {
  if (affineForOp1 == affineForOp2)
    return true;

  if (affineForOp1.getStep() != affineForOp2.getStep())
    return false;
  if (!compareAffineMaps(affineForOp1.getLowerBoundMap(),
                         affineForOp2.getLowerBoundMap()) ||
      !compareAffineMaps(affineForOp1.getUpperBoundMap(),
                         affineForOp2.getUpperBoundMap()))
    return false;
  return true;
}

void mergeLoop(OpBuilder &builder, affine::AffineForOp &op1,
               affine::AffineForOp &op2, IRMapping &mapping1,
               IRMapping &mapping2, Value conditionArg, bool &foundDifference) {
  auto loc = op1->getLoc();

  // Save insertion point
  OpBuilder::InsertionGuard guard(builder);

  // Create new affine.for with same arguments
  auto lowerBoundMap = op1.getLowerBoundMap();
  auto lowerBoundOperands = llvm::SmallVector<Value, 4>(
      op1.getLowerBoundOperands().begin(), op1.getLowerBoundOperands().end());
  auto upperBoundMap = op1.getUpperBoundMap();
  auto upperBoundOperands = llvm::SmallVector<Value, 4>(
      op1.getUpperBoundOperands().begin(), op1.getUpperBoundOperands().end());
  int64_t step = op1.getStep().getSExtValue();

  auto newAffineForOp = builder.create<mlir::affine::AffineForOp>(
      loc, lowerBoundOperands, lowerBoundMap, upperBoundOperands, upperBoundMap,
      step);

  Block *body1 = op1.getBody();
  Block *body2 = op2.getBody();
  Block *newBody = newAffineForOp.getBody();

  // Set insertion point to current loop body
  builder.setInsertionPointToStart(newBody);

  // Add IRMapping for latter cloning
  for (size_t i = 0; i < body1->getNumArguments(); ++i) {
    mapping1.map(body1->getArgument(i), newBody->getArgument(i));
  }
  for (size_t i = 0; i < body2->getNumArguments(); ++i) {
    mapping2.map(body2->getArgument(i), newBody->getArgument(i));
  }

  auto body1It = body1->begin();
  auto body2It = body2->begin();

  // Iterate over two FuncOps to find branch location
  while (body1It != body1->end() && body2It != body2->end()) {
    if (!foundDifference) {
      if (!(&(*body1It) == &(*body2It))) {
        // If we found an affine.for to merge
        // Todo: Support dynamic loop range
        auto affineForOp1 = dyn_cast<affine::AffineForOp>(&(*body1It));
        auto affineForOp2 = dyn_cast<affine::AffineForOp>(&(*body2It));
        if (affineForOp1 && affineForOp2 &&
            compareAffineForOps(affineForOp1, affineForOp2)) {
          mergeLoop(builder, affineForOp1, affineForOp2, mapping1, mapping2,
                    conditionArg, foundDifference);
        } else {
          foundDifference = true;
          break;
        }
      } else {
        builder.clone(*body1It);
      }
      ++body1It;
      ++body2It;
    } else {
      break;
    }
  }

  // Create branch for the rest after difference is found
  builder.create<scf::IfOp>(
      loc, conditionArg,
      [&](OpBuilder &thenBuilder, Location thenLoc) {
        while (body1It != body1->end()) {
          auto &op = *body1It;
          if (auto yieldOp = dyn_cast<affine::AffineYieldOp>(&op)) {
            break;
          }
          thenBuilder.clone(*body1It, mapping1);
          ++body1It;
        }
        thenBuilder.create<scf::YieldOp>(thenLoc);
      },
      [&](OpBuilder &elseBuilder, Location elseLoc) {
        while (body2It != body2->end()) {
          auto &op = *body2It;
          if (auto yieldOp = dyn_cast<affine::AffineYieldOp>(&op)) {
            break;
          }
          elseBuilder.clone(*body2It, mapping2);
          ++body2It;
        }
        elseBuilder.create<scf::YieldOp>(elseLoc);
      });
}

func::FuncOp unifyKernels(func::FuncOp &func1, func::FuncOp &func2,
                          OpBuilder &builder, int loop_num) {
  std::string newFuncName =
      func1.getName().str() + "_" + func2.getName().str() + "_unified";

  // Todo: Now assuming return types and input types are the same
  // Create new FuncOp with additional parameter
  auto oldFuncType = func1.getFunctionType();
  auto oldInputTypes = oldFuncType.getInputs();
  auto loc = builder.getUnknownLoc();
  SmallVector<Type, 4> newInputTypes(oldInputTypes.begin(),
                                     oldInputTypes.end());
  auto newOutputTypes = oldFuncType.getResults();
  Type i8Type = builder.getI8Type();
  Type memrefType = MemRefType::get({loop_num}, i8Type);
  newInputTypes.push_back(memrefType);
  auto newFuncType = builder.getFunctionType(newInputTypes, newOutputTypes);
  // Todo: Need to clone attribute
  auto newFuncOp = func::FuncOp::create(loc, newFuncName, newFuncType,
                                        ArrayRef<NamedAttribute>{});

  // Create new block for insertion
  Block *entryBlock = newFuncOp.addEntryBlock();
  auto inst = entryBlock->getArgument(entryBlock->getNumArguments() - 1);
  builder.setInsertionPointToStart(entryBlock);

  auto outterLoop =
      builder.create<mlir::affine::AffineForOp>(loc, 0, loop_num, 1);
  mlir::Value loopIndex = outterLoop.getInductionVar();
  builder.setInsertionPointToStart(outterLoop.getBody());
  mlir::Value curInst =
      builder.create<mlir::affine::AffineLoadOp>(loc, inst, loopIndex);
  mlir::Value zeroValue = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getI8Type(), builder.getI8IntegerAttr(0));
  mlir::Value conditionArg = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, curInst, zeroValue);

  auto &block1 = func1.front();
  auto &block2 = func2.front();
  auto block1It = block1.begin();
  auto block2It = block2.begin();
  bool foundDifference = false;

  // Create IRMapping for latter cloning
  IRMapping mapping1;
  for (size_t i = 0; i < block1.getNumArguments(); ++i) {
    mapping1.map(block1.getArgument(i), entryBlock->getArgument(i));
  }
  IRMapping mapping2;
  for (size_t i = 0; i < block2.getNumArguments(); ++i) {
    mapping2.map(block2.getArgument(i), entryBlock->getArgument(i));
  }

  // Iterate over two FuncOps to find branch location
  while (block1It != block1.end() && block2It != block2.end()) {
    if (!foundDifference) {
      if (!(&(*block1It) == &(*block2It))) {
        // If we found an affine.for to merge
        // Todo: Support dynamic loop range
        auto affineForOp1 = dyn_cast<affine::AffineForOp>(&(*block1It));
        auto affineForOp2 = dyn_cast<affine::AffineForOp>(&(*block2It));
        if (affineForOp1 && affineForOp2 &&
            compareAffineForOps(affineForOp1, affineForOp2)) {
          mergeLoop(builder, affineForOp1, affineForOp2, mapping1, mapping2,
                    conditionArg, foundDifference);
        } else {
          foundDifference = true;
          break;
        }
      } else {
        builder.clone(*block1It);
      }
      ++block1It;
      ++block2It;
    } else {
      break;
    }
  }

  auto &op1 = *block1It;
  auto &op2 = *block2It;
  auto returnOp1 = dyn_cast<func::ReturnOp>(&op1);
  auto returnOp2 = dyn_cast<func::ReturnOp>(&op2);
  // Create branch for the rest after difference is found
  if (!returnOp1 || !returnOp2) {
    builder.create<scf::IfOp>(
        loc, conditionArg,
        [&](OpBuilder &thenBuilder, Location thenLoc) {
          while (block1It != block1.end()) {
            auto &op = *block1It;
            if (auto returnOp = dyn_cast<func::ReturnOp>(&op)) {
              break;
            }
            thenBuilder.clone(*block1It, mapping1);
            ++block1It;
          }
          thenBuilder.create<scf::YieldOp>(thenLoc);
        },
        [&](OpBuilder &elseBuilder, Location elseLoc) {
          while (block2It != block2.end()) {
            auto &op = *block2It;
            if (auto returnOp = dyn_cast<func::ReturnOp>(&op)) {
              break;
            }
            elseBuilder.clone(*block2It, mapping2);
            ++block2It;
          }
          elseBuilder.create<scf::YieldOp>(elseLoc);
        });
  }

  // Create returnOp
  // Todo: Now assume the return value is the same
  builder.setInsertionPointToEnd(entryBlock);
  builder.clone(*block1It, mapping1);

  return newFuncOp;
}

/// Pass entry point
ModuleOp applyUnifyKernels(ModuleOp &module1, ModuleOp &module2, int loop_num) {
  auto funcOps1 = module1.getOps<func::FuncOp>();
  auto funcOps2 = module2.getOps<func::FuncOp>();

  auto it1 = funcOps1.begin();
  auto it2 = funcOps2.begin();

  ModuleOp newModule = ModuleOp::create(UnknownLoc::get(module1.getContext()));
  OpBuilder builder(newModule.getContext());

  while (it1 != funcOps1.end() && it2 != funcOps2.end()) {
    func::FuncOp funcOp1 = *it1;
    func::FuncOp funcOp2 = *it2;
    func::FuncOp newFuncOp = unifyKernels(funcOp1, funcOp2, builder, loop_num);
    newModule.push_back(newFuncOp);

    ++it1;
    ++it2;
  }

  return newModule;
}

} // namespace allo
} // namespace mlir