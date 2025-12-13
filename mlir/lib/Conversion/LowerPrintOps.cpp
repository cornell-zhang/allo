/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// LowerPrintOps.cpp defines a pass to lower PrintOp and PrintMemRefOp to
// MLIR's utility printing functions or C printf functions. It also handles
// Fixed-point values/memref casting to float.
// We define our own memref printing and value printing operations to support
// following cases:
// - Multiple values printed with format string.
// - Print memref. Note that memref printing doesn't support formating.
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Support/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <string>

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {
#define GEN_PASS_DEF_LOWERPRINTOPS
#include "allo/Conversion/Passes.h.inc"
} // namespace allo
} // namespace mlir

namespace mlir {
namespace allo {

/// Helper functions to decalare C printf function and format string
/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
LLVM::LLVMFuncOp getOrInsertPrintf(OpBuilder &rewriter, ModuleOp module) {
  auto *context = module.getContext();
  for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
    if (func.getName() == "printf")
      return func;
  }

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI8PtrTy = LLVM::LLVMPointerType::get(context, 8);
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                           llvmFnType);
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name,
                              StringRef value, ModuleOp module) {
  // Create the global at the entry of the module.
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value),
                                            /*alignment=*/32);
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, IntegerType::get(builder.getContext(), 64),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext(), 8),
      global.getType(), globalPtr, ArrayRef<Value>({cst0, cst0}));
}

void lowerPrintOpToPrintf(Operation *op, int idx) {
  OpBuilder builder(op);
  auto loc = op->getLoc();
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();

  // If the PrintOp has string attribute, it is the format string
  std::string format_str = "%.4f \00";
  if (op->hasAttr("format")) {
    format_str = llvm::cast<StringAttr>(op->getAttr("format")).getValue().str();
    bool replaced = true;
    while (replaced) {
      replaced = replace(format_str, "%d", "%.0f");
    }
  }
  std::string sign_str;
  if (op->hasAttr("signedness")) {
    sign_str =
        llvm::cast<StringAttr>(op->getAttr("signedness")).getValue().str();
  }
  // Get a symbol reference to the printf function, inserting it if
  // necessary. Create global strings for format and new line
  auto printfRef = getOrInsertPrintf(builder, parentModule);
  Value formatSpecifierCst =
      getOrCreateGlobalString(loc, builder, "frmt_spec" + std::to_string(idx),
                              StringRef(format_str), parentModule);

  // Create a call to printf with the format string and the values to print.
  SmallVector<Value, 4> operands;
  operands.push_back(formatSpecifierCst);
  for (auto enum_value : llvm::enumerate(op->getOperands())) {
    // Note: llvm.mlir.printf only works with f64 type, so we need to cast the
    // value.
    auto value = enum_value.value();
    auto idx = enum_value.index();
    bool is_unsigned = false;
    if (idx < sign_str.size()) {
      is_unsigned = sign_str[idx] == 'u';
    }
    operands.push_back(castToF64(builder, value, is_unsigned));
  }
  builder.create<LLVM::CallOp>(loc, printfRef, operands);
}

void lowerPrintMemRef(Operation *op) {
  OpBuilder builder(op);
  auto loc = op->getLoc();
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto srcMemRefType = llvm::cast<MemRefType>(op->getOperand(0).getType());
  auto srcElementType = srcMemRefType.getElementType();
  bool unsign = op->hasAttr("unsigned");
  std::string funcName;
  if (llvm::isa<FloatType>(srcElementType) &&
      srcElementType.getIntOrFloatBitWidth() == 32) {
    funcName = "printMemrefF32";
  } else if (llvm::isa<FloatType>(srcElementType) &&
             srcElementType.getIntOrFloatBitWidth() == 64) {
    funcName = "printMemrefF64";
  } else if (llvm::isa<IntegerType>(srcElementType)) {
    funcName = "printMemrefI64";
  } else {
    op->emitError("unsupported type for printMemref");
  }
  // For integer memrefs, cast to I64 memref before printing.
  builder.setInsertionPoint(op);
  Value srcMemRef;
  if (llvm::isa<IntegerType>(srcElementType)) {
    Value operand = op->getOperand(0);
    srcMemRef = castIntMemRef(builder, op->getLoc(), operand, 64, unsign,
                              /*replace*/ false);
    srcElementType = IntegerType::get(builder.getContext(), 64);
  } else {
    srcMemRef = op->getOperand(0);
  }

  // Create print function declaration
  // lookup if the function already exists
  func::FuncOp printFuncDecl;
  auto pointerType =
      UnrankedMemRefType::get(srcElementType, srcMemRefType.getMemorySpace());
  if (!(printFuncDecl = parentModule.lookupSymbol<func::FuncOp>(funcName))) {
    builder.setInsertionPointToStart(parentModule.getBody());
    FunctionType printMemRefType =
        FunctionType::get(builder.getContext(), {pointerType}, {});
    printFuncDecl =
        builder.create<func::FuncOp>(loc, funcName, printMemRefType);
    printFuncDecl.setPrivate();
  }

  // Use memref.cast to remove rank
  builder.setInsertionPoint(op);
  auto castedMemRef =
      builder.create<memref::CastOp>(loc, pointerType, srcMemRef);
  SmallVector<Value, 1> operands{castedMemRef};
  builder.create<func::CallOp>(loc, printFuncDecl, operands);
}

void PrintOpLoweringDispatcher(func::FuncOp &funcOp) {
  SmallVector<Operation *, 4> printOps;
  funcOp.walk([&](Operation *op) {
    if (auto printOp = dyn_cast<PrintOp>(op)) {
      printOps.push_back(printOp);
    }
  });
  for (auto v : llvm::enumerate(printOps)) {
    lowerPrintOpToPrintf(v.value(), v.index());
  }
  std::reverse(printOps.begin(), printOps.end());
  for (auto printOp : printOps) {
    printOp->erase();
  }
}

void PrintMemRefOpLoweringDispatcher(func::FuncOp &funcOp) {
  SmallVector<Operation *, 4> printMemRefOps;
  funcOp.walk([&](Operation *op) {
    if (auto printMemRefOp = dyn_cast<PrintMemRefOp>(op)) {
      printMemRefOps.push_back(printMemRefOp);
    }
  });
  for (auto printMemRefOp : printMemRefOps) {
    lowerPrintMemRef(printMemRefOp);
  }
  std::reverse(printMemRefOps.begin(), printMemRefOps.end());
  for (auto printMemRefOp : printMemRefOps) {
    printMemRefOp->erase();
  }
}

/// Pass entry point
bool applyLowerPrintOps(ModuleOp &module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    PrintOpLoweringDispatcher(func);
    PrintMemRefOpLoweringDispatcher(func);
  }
  return true;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerPrintOpsTransformation
    : public mlir::allo::impl::LowerPrintOpsBase<
          AlloLowerPrintOpsTransformation> {
  void runOnOperation() override {
    auto module = getOperation();
    if (!applyLowerPrintOps(module)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createLowerPrintOpsPass() {
  return std::make_unique<AlloLowerPrintOpsTransformation>();
}
} // namespace allo
} // namespace mlir