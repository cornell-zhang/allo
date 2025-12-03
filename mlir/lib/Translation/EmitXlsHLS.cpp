/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Based on EmitVivadoHLS.cpp for XLS HLS support
 */

#include "allo/Translation/EmitXlsHLS.h"
#include "allo/Translation/EmitVivadoHLS.h"  // Include Vivado emitter
#include "allo/Dialect/Visitor.h"
#include "allo/Support/Utils.h"
#include "allo/Translation/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"

using namespace mlir;
using namespace allo;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// used for determine whether to generate C++ default types or ac_(u)int
static bool BIT_FLAG = false;

// Flag to control whether arrays are emitted as __xls_memory<T, size> (true)
// or as plain C arrays int arr[size] (false - default for registers)
static bool USE_MEMORY_FLAG = false;

static SmallString<16> getXLSTypeName(Type valType) {
  if (auto arrayType = valType.dyn_cast<ShapedType>())
    valType = arrayType.getElementType();

  // Handle integer types.
  else if (valType.isa<IndexType>())
    return SmallString<16>("int");
  else if (auto intType = valType.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 1) {
        return SmallString<16>("ac_int<1, false>");
    } else {
      switch (intType.getWidth()) {
      case 32:
        if (intType.getSignedness() == IntegerType::SignednessSemantics::Unsigned)
          return SmallString<16>("uint32_t");
        else
          return SmallString<16>("int");
        break;
      case 64:
        if (intType.getSignedness() == IntegerType::SignednessSemantics::Unsigned)
          return SmallString<16>("uint64_t");
        else
          return SmallString<16>("long long");
        break;
      default:
        return SmallString<16>("ac_int<" + std::to_string(intType.getWidth()) + ", " +
                                (intType.getSignedness() == IntegerType::SignednessSemantics::Unsigned ? "false" : "true") + ">");
      }
    
    }
  }

  else if (auto streamType = valType.dyn_cast<StreamType>())
    return SmallString<16>(
        "__xls_channel< " +
        std::string(getXLSTypeName(streamType.getBaseType()).c_str()) + ", "
        + "__XLS_IO_PLACEHOLDER__" + ">");

  // Check for unsupported types and provide clear error message
  else if (valType.isa<Float16Type>() || valType.isa<Float32Type>() || 
           valType.isa<Float64Type>()) {
    assert(1 == 0 && "XLS[cc] backend does not currently support floating-point types. Please use integer or fixed-point types instead.");
  }
  else if (auto fixedType = valType.dyn_cast<allo::FixedType>()) {
    assert(1 == 0 && "XLS[cc] backend does not currently support fixed-point types. Please use integer types instead.");
  }
  else if (auto ufixedType = valType.dyn_cast<allo::UFixedType>()) {
    assert(1 == 0 && "XLS[cc] backend does not currently support unsigned fixed-point types. Please use integer types instead.");
  }
  else {
    assert(1 == 0 && "XLS[cc] backend encountered unsupported type. Only integer types and streams are currently supported.");
  }

  return SmallString<16>();
}

// Forward declare the Vivado ModuleEmitter from the Vivado namespace
namespace vhls {
  class ModuleEmitter;
}

namespace {
// AffineExprEmitter class for emitting affine expressions
class AffineExprEmitter : public AlloEmitterBase,
                          public AffineExprVisitor<AffineExprEmitter> {
public:
  using operand_range = Operation::operand_range;
  explicit AffineExprEmitter(AlloEmitterState &state, unsigned numDim,
                             operand_range operands)
      : AlloEmitterBase(state), numDim(numDim), operands(operands) {}

  void visitAddExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "+"); }
  void visitMulExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "*"); }
  void visitModExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "%"); }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) {
    emitAffineBinary(expr, "/");
  }
  void visitCeilDivExpr(AffineBinaryOpExpr expr) {
    // This is super inefficient.
    os << "(";
    visit(expr.getLHS());
    os << " + ";
    visit(expr.getRHS());
    os << " - 1) / ";
    visit(expr.getRHS());
    os << ")";
  }

  void visitConstantExpr(AffineConstantExpr expr) { os << expr.getValue(); }

  void visitDimExpr(AffineDimExpr expr) {
    os << getName(operands[expr.getPosition()]);
  }
  void visitSymbolExpr(AffineSymbolExpr expr) {
    os << getName(operands[numDim + expr.getPosition()]);
  }

  /// Affine expression emitters.
  void emitAffineBinary(AffineBinaryOpExpr expr, const char *syntax) {
    os << "(";
    if (auto constRHS = expr.getRHS().dyn_cast<AffineConstantExpr>()) {
      if ((unsigned)*syntax == (unsigned)*"*" && constRHS.getValue() == -1) {
        os << "-";
        visit(expr.getLHS());
        os << ")";
        return;
      }
      if ((unsigned)*syntax == (unsigned)*"+" && constRHS.getValue() < 0) {
        visit(expr.getLHS());
        os << " - ";
        os << -constRHS.getValue();
        os << ")";
        return;
      }
    }
    if (auto binaryRHS = expr.getRHS().dyn_cast<AffineBinaryOpExpr>()) {
      if (auto constRHS = binaryRHS.getRHS().dyn_cast<AffineConstantExpr>()) {
        if ((unsigned)*syntax == (unsigned)*"+" && constRHS.getValue() == -1 &&
            binaryRHS.getKind() == AffineExprKind::Mul) {
          visit(expr.getLHS());
          os << " - ";
          visit(binaryRHS.getLHS());
          os << ")";
          return;
        }
      }
    }
    visit(expr.getLHS());
    os << " " << syntax << " ";
    visit(expr.getRHS());
    os << ")";
  }

  void emitAffineExpr(AffineExpr expr) { visit(expr); }

private:
  unsigned numDim;
  operand_range operands;
};

// XLS ModuleEmitter that inherits from Vivado ModuleEmitter
class XLSModuleEmitter : public allo::vhls::ModuleEmitter {
public:
  using operand_range = Operation::operand_range;
  explicit XLSModuleEmitter(AlloEmitterState &state) : allo::vhls::ModuleEmitter(state) {}

  // Override methods that need XLS-specific behavior
  void emitModule(ModuleOp module) override;
  void emitFunctionDirectives(func::FuncOp func,
                              ArrayRef<Value> portList) override;
  void emitArrayDecl(Value array, bool isFunc = false,
                     std::string name = "") override;
  void emitLoopDirectives(Operation *op) override;
  void emitStreamConstruct(allo::StreamConstructOp op) override;
  void emitArrayDirectives(Value memref) override;
  void emitFunction(func::FuncOp func) override;
  void emitAffineFor(AffineForOp op) override;
  void emitScfFor(scf::ForOp op) override;
  void emitLoad(memref::LoadOp op) override;
  void emitStore(memref::StoreOp op) override;
  void emitAffineLoad(AffineLoadOp op) override;
  void emitAffineStore(AffineStoreOp op) override;

protected:
  // Helper to emit flattened index for multi-dimensional array access
  void emitFlattenedIndex(MemRefType memrefType, Operation::operand_range indices);
  // Helper to emit flattened affine index
  void emitFlattenedAffineIndex(MemRefType memrefType, AffineMap affineMap, 
                                 AffineExprEmitter &affineEmitter);
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "");
  // Helper method to get XLS-specific type names
  SmallString<16> getTypeName(Type valType) { return getXLSTypeName(valType); }
  SmallString<16> getTypeName(Value val) { return getXLSTypeName(val.getType()); }
  
  // Track array arguments that should become local arrays populated from channels
  bool inArrayFunction = false;
  DenseMap<Value, std::string> arrayArgToLocalName;  // Maps function arg -> local array name (e.g., "v0")
  DenseMap<Value, std::string> arrayArgToChannelName; // Maps function arg -> channel name (e.g., "v0_in")
  Value returnArray = nullptr;
  std::string outputChannelName = "out";
};
} // namespace

//===----------------------------------------------------------------------===//
// XLS-specific implementations
//===----------------------------------------------------------------------===//

void XLSModuleEmitter::emitValue(Value val,
                                 unsigned rank,  
                                 bool isPtr, 
                                 std::string name) {
  (void)rank; (void)isPtr;

  // already assigned name
  if (isDeclared(val)) {
    os << getName(val);
    return;
  }

  os << getXLSTypeName(val.getType()) << " ";

  if (name.empty()) {
    os << addName(val, /*isPtr=*/false);
  } else {
    os << addName(val, /*isPtr=*/false, name);
  }
}

void XLSModuleEmitter::emitFunctionDirectives(func::FuncOp func,
                                                   ArrayRef<Value> portList) {
  // For XLS HLS, emit the hls_design top pragma for top-level functions
  if (func->hasAttr("top")) {
    indent();
    os << "#pragma hls_design top\n";
    os << "\n";
  }

  // Emit array directives for function ports
  for (auto &port : portList)
    if (port.getType().isa<MemRefType>())
      emitArrayDirectives(port);
}

void XLSModuleEmitter::emitArrayDecl(Value array, bool isFunc, std::string name) {
  assert(!isDeclared(array) && "has been declared before.");

  auto shapedType = array.getType().dyn_cast<ShapedType>();
  if (!shapedType || !shapedType.hasStaticShape()) {
    // XLS[cc] only supports fixed-size arrays here.
    assert(1 == 0 && "XLS[cc] backend only supports statically-sized arrays/memories");
  }

  auto elemType = shapedType.getElementType();
  auto shape = shapedType.getShape();

  if (auto memref = array.getType().dyn_cast<MemRefType>()) {
    auto attr = memref.getMemorySpace().dyn_cast_or_null<StringAttr>();
    std::string memspace = attr ? attr.getValue().str() : "";

    // 1) STREAM CASE -> channel, not array/memory
    if (!memspace.empty() && memspace.rfind("stream", 0) == 0) {
      // Delegate to stream emission (ac_channel / __xls_channel)
      os << getXLSTypeName(elemType) << " ";
      os << addName(array, /*isPtr=*/false, name);
      return;
    }

    // 2) MEMORY CASE -> __xls_memory< T, Size >
    if (!memspace.empty() && memspace.rfind("memory", 0) == 0) {
      // For now, flatten all dims into a single size.
      int64_t totalSize = 1;
      for (int64_t dim : shape) totalSize *= dim;

      // You can use __xls_memory directly, or your alias:
      // template<typename T, int Size> using Memory = __xls_memory<T, Size>;
      // If you're using the alias, emit "Memory<...>" instead.
      os << "__xls_memory<"
         << getXLSTypeName(elemType) << ", "
         << totalSize << "> ";

      std::string varName = std::string(addName(array, /*isPtr=*/false, name).str());
      os << varName;

      // Optional: keep original multidim shape as a comment
      os << " /* original shape: ";
      os << getXLSTypeName(elemType) << " " << varName;
      for (int64_t dim : shape)
        os << "[" << dim << "]";
      os << " */";

      return;
    }

    // 3) DEFAULT MEMREF -> either skip (if USE_MEMORY_FLAG, wrapper handles it)
    //    or plain C array: T name[d0][d1]...
    if (USE_MEMORY_FLAG) {
      // Memory mode: Don't emit declaration here - wrapper will add __xls_memory at class level
      // Just register the name so subsequent uses work
      addName(array, /*isPtr=*/false, name);
      // Emit a comment placeholder so the wrapper knows about this array
      int64_t totalSize = 1;
      for (int64_t dim : shape) totalSize *= dim;
      os << "// __xls_memory_decl__: " << getXLSTypeName(elemType) << ", " << totalSize << ", ";
      os << (name.empty() ? std::string(getName(array).c_str()) : name);
      for (int64_t dim : shape)
        os << ", " << dim;
      return;
    }

    // Register mode (default): plain C array
    if (isFunc)
      os << "const ";   // or drop const if you need writes

    os << getXLSTypeName(elemType) << " ";
    std::string varName = std::string(addName(array, /*isPtr=*/false, name).str());
    os << varName;

    for (int64_t dim : shape)
      os << "[" << dim << "]";

    // Debug comment
    os << " /* ";
    os << getXLSTypeName(elemType) << " " << varName;
    for (int64_t dim : shape)
      os << "[" << dim << "]";
    os << " */";

    return;
  }

  // Non-memref shaped types: treat like plain arrays (or error)
  if (isFunc)
    os << "const ";

  os << getXLSTypeName(elemType) << " ";
  std::string varName = std::string(addName(array, /*isPtr=*/false, name).str());
  os << varName;
  for (int64_t dim : shape)
    os << "[" << dim << "]";

  os << " /* tensor-like shaped type */";
}

/// Helper function to check if a region contains channel operations.
/// Returns true if any operation in the region reads from or writes to a channel.
static bool containsChannelOperations(Region &region) {
  bool hasChannelOps = false;
  region.walk([&](Operation *op) {
    // Check for StreamGetOp (channel read)
    if (isa<StreamGetOp>(op)) {
      hasChannelOps = true;
      return WalkResult::interrupt();
    }
    // Check for StreamPutOp (channel write)
    if (isa<StreamPutOp>(op)) {
      hasChannelOps = true;
      return WalkResult::interrupt();
    }
    // Check for memref::LoadOp on stream memory space
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      auto memref = loadOp.getMemRef();
      if (auto memrefType = memref.getType().dyn_cast<MemRefType>()) {
        if (auto attr = memrefType.getMemorySpace()) {
          if (auto strAttr = attr.dyn_cast<StringAttr>()) {
            if (strAttr.getValue().str().rfind("stream", 0) == 0) {
              hasChannelOps = true;
              return WalkResult::interrupt();
            }
          }
        }
      }
    }
    // Check for memref::StoreOp on stream memory space
    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      auto memref = storeOp.getMemRef();
      if (auto memrefType = memref.getType().dyn_cast<MemRefType>()) {
        if (auto attr = memrefType.getMemorySpace()) {
          if (auto strAttr = attr.dyn_cast<StringAttr>()) {
            if (strAttr.getValue().str().rfind("stream", 0) == 0) {
              hasChannelOps = true;
              return WalkResult::interrupt();
            }
          }
        }
      }
    }
    return WalkResult::advance();
  });
  return hasChannelOps;
}

void XLSModuleEmitter::emitLoopDirectives(Operation *op) {
  // Check for pipeline attributes first - if pipeline is present, don't unroll
  // Check for pipeline attributes (could be "allo.pipeline" unit attr or "pipeline_ii" with value)
  if (op->hasAttr("allo.pipeline")) {
    indent();
    os << "#pragma hls_pipeline_init_interval 1\n";
    return;  // Pipeline takes precedence, don't unroll
  } else if (auto ii = getLoopDirective(op, "pipeline_ii")) {
    indent();
    os << "#pragma hls_pipeline_init_interval 1\n";
    return;  // Pipeline takes precedence, don't unroll
  }

  // Check if loop body contains channel operations.
  // If it does, we cannot unroll because multiple unrolled iterations
  // would all read/write from the same channel simultaneously, which is not allowed.
  // In this case, use pipelining with init interval 1 instead.
  bool hasChannelOps = false;
  if (auto affineFor = dyn_cast<AffineForOp>(op)) {
    hasChannelOps = containsChannelOperations(affineFor.getRegion());
  } else if (auto scfFor = dyn_cast<scf::ForOp>(op)) {
    hasChannelOps = containsChannelOperations(scfFor.getRegion());
  }

  if (hasChannelOps) {
    // Cannot unroll loops with channel operations - use pipelining instead
    indent();
    os << "#pragma hls_pipeline_init_interval 1\n";
    return;
  }

  // By default, unroll all loops unless they have a pipeline pragma
  // Check if explicit unroll directive is present (with optional factor)
  if (auto factor = getLoopDirective(op, "unroll")) {
    indent();
    os << "#pragma hls_unroll yes";
    auto val = factor.cast<IntegerAttr>().getValue();
    if (val != 0) {
      os << " factor=" << val;
    }
    os << "\n";
  } else {
    // Default: fully unroll the loop
    indent();
    os << "#pragma hls_unroll yes\n";
  }
}

void XLSModuleEmitter::emitAffineFor(AffineForOp op) {
  auto iterVar = op.getInductionVar();
  std::string loop_name = "";
  if (op->hasAttr("loop_name")) {
    loop_name = op->getAttr("loop_name").cast<StringAttr>().getValue().str();
  }

  // Emit loop directives BEFORE the loop (unroll first, then pipeline)
  emitLoopDirectives(op);

  // Emit the for loop WITHOUT label
  indent();
  os << "for (";
  
  // Emit lower bound - first declare the loop variable
  os << getXLSTypeName(iterVar.getType()) << " ";
  // Add the name to the table without re-emitting type
  os << addName(iterVar, false, loop_name);
  os << " = ";
  auto lowerMap = op.getLowerBoundMap();
  AffineExprEmitter lowerEmitter(state, lowerMap.getNumDims(),
                                 op.getLowerBoundOperands());
  if (lowerMap.getNumResults() == 1)
    lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
  else {
    for (unsigned i = 0, e = lowerMap.getNumResults() - 1; i < e; ++i)
      os << "max(";
    lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
    for (auto &expr : llvm::drop_begin(lowerMap.getResults(), 1)) {
      os << ", ";
      lowerEmitter.emitAffineExpr(expr);
      os << ")";
    }
  }
  os << "; ";

  // Emit upper bound.
  emitValue(iterVar, 0, false, loop_name);
  os << " < ";
  auto upperMap = op.getUpperBoundMap();
  AffineExprEmitter upperEmitter(state, upperMap.getNumDims(),
                                 op.getUpperBoundOperands());
  if (upperMap.getNumResults() == 1)
    upperEmitter.emitAffineExpr(upperMap.getResult(0));
  else {
    for (unsigned i = 0, e = upperMap.getNumResults() - 1; i < e; ++i)
      os << "min(";
    upperEmitter.emitAffineExpr(upperMap.getResult(0));
    for (auto &expr : llvm::drop_begin(upperMap.getResults(), 1)) {
      os << ", ";
      upperEmitter.emitAffineExpr(expr);
      os << ")";
    }
  }
  os << "; ";

  // Emit increase step.
  emitValue(iterVar, 0, false, loop_name);
  if (op.getStep() == 1)
    os << "++) {";
  else
    os << " += " << op.getStep() << ") {";
  emitInfoAndNewLine(op);

  addIndent();

  // Emit loop body
  emitBlock(*op.getBody());
  reduceIndent();

  indent();
  os << "}\n";
}

void XLSModuleEmitter::emitScfFor(scf::ForOp op) {
  // Emit loop directives BEFORE the loop (unroll first, then pipeline)
  emitLoopDirectives(op);

  // Emit the for loop WITHOUT label
  indent();
  os << "for (";
  auto iterVar = op.getInductionVar();

  // Emit lower bound.
  emitValue(iterVar);
  os << " = ";
  emitValue(op.getLowerBound());
  os << "; ";

  // Emit upper bound.
  emitValue(iterVar);
  os << " < ";
  emitValue(op.getUpperBound());
  os << "; ";

  // Emit increase step.
  emitValue(iterVar);
  os << " += ";
  emitValue(op.getStep());
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();

  // Emit loop body (no directives inside - they're already before the loop)
  emitBlock(*op.getBody());
  reduceIndent();

  indent();
  os << "}\n";
}

void XLSModuleEmitter::emitStreamConstruct(allo::StreamConstructOp op) {
  indent();
  Value result = op.getResult();

  // Apply unsigned fixups if requested.
  fixUnsignedType(result, op->hasAttr("unsigned"));

  Type ty = result.getType();

  // Reject arrays of channels.
  if (auto shaped = ty.dyn_cast<ShapedType>()) {
    if (shaped.getElementType().isa<StreamType>()) {
      emitError(op, "XLS[cc] backend does not support arrays-of-streams; flatten or convert to separate streams/memory before EmitXLSHLS.");
      return;
    }
  }

  // We expect plain StreamType.
  if (!ty.isa<StreamType>()) {
    emitError(op, "emitStreamConstruct expected a StreamType result for XLS backend.");
    return;
  }

  // This uses your getXLSTypeName(StreamType) → __xls_channel<..., __XLS_IO_PLACEHOLDER__>
  emitValue(result);
  os << ";\n";

  // No depth pragmas for XLS; buffering is via __xls_channel or synthesis settings.
  emitInfoAndNewLine(op);
}

void XLSModuleEmitter::emitArrayDirectives(Value memref) {
  bool emitPragmaFlag = false;
  auto type = memref.getType().cast<MemRefType>();

  // streaming
  auto attr = type.getMemorySpace();
  if (attr) {
    std::string attr_str = attr.cast<StringAttr>().getValue().str();
    if (attr_str.substr(0, 6) == "stream") {
      // Note: XLS HLS doesn't need explicit stream pragmas like Vivado HLS
      return;
    }
  }

  // For other array directives, delegate to the parent implementation
  // but we need to call the parent method explicitly
  allo::vhls::ModuleEmitter::emitArrayDirectives(memref);
}

void XLSModuleEmitter::emitFlattenedIndex(MemRefType memrefType, Operation::operand_range indices) {
  // For __xls_memory (flattened), compute linear index: i * d1 * d2 + j * d2 + k
  auto shape = memrefType.getShape();
  bool first = true;
  
  for (unsigned i = 0; i < indices.size(); ++i) {
    // Calculate stride (product of all dimensions after this one)
    int64_t stride = 1;
    for (unsigned j = i + 1; j < shape.size(); ++j) {
      stride *= shape[j];
    }
    
    if (!first) {
      os << " + ";
    }
    first = false;
    
    if (stride == 1) {
      emitValue(indices[i]);
    } else {
      emitValue(indices[i]);
      os << " * " << stride;
    }
  }
}

void XLSModuleEmitter::emitLoad(memref::LoadOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  
  auto memref = op.getMemRef();
  auto memrefType = memref.getType().cast<MemRefType>();
  emitValue(memref);
  
  // Check for stream memory space
  auto attr = memrefType.getMemorySpace();
  if (attr && attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    // Stream case - use parent's implementation
    allo::vhls::ModuleEmitter::emitLoad(op);
    return;
  }
  
  // For regular memref, handle memory mode vs register mode
  if (USE_MEMORY_FLAG && memrefType.getRank() > 1) {
    // Memory mode with multi-dimensional array: use flattened index
    os << "[";
    emitFlattenedIndex(memrefType, op.getIndices());
    os << "]";
  } else {
    // Register mode or 1D array: use normal indexing
    for (auto index : op.getIndices()) {
      os << "[";
      emitValue(index);
      os << "]";
    }
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void XLSModuleEmitter::emitStore(memref::StoreOp op) {
  indent();
  
  auto memref = op.getMemRef();
  auto memrefType = memref.getType().cast<MemRefType>();
  emitValue(memref);
  
  // Check for stream memory space
  auto attr = memrefType.getMemorySpace();
  if (attr && attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    // Stream case - use parent's implementation
    allo::vhls::ModuleEmitter::emitStore(op);
    return;
  }
  
  // For regular memref, handle memory mode vs register mode
  if (USE_MEMORY_FLAG && memrefType.getRank() > 1) {
    // Memory mode with multi-dimensional array: use flattened index
    os << "[";
    emitFlattenedIndex(memrefType, op.getIndices());
    os << "]";
  } else {
    // Register mode or 1D array: use normal indexing
    for (auto index : op.getIndices()) {
      os << "[";
      emitValue(index);
      os << "]";
    }
  }
  
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  emitInfoAndNewLine(op);
}

void XLSModuleEmitter::emitFlattenedAffineIndex(MemRefType memrefType, AffineMap affineMap,
                                                 AffineExprEmitter &affineEmitter) {
  // For __xls_memory (flattened), compute linear index from affine expressions
  auto shape = memrefType.getShape();
  auto results = affineMap.getResults();
  bool first = true;
  
  for (unsigned i = 0; i < results.size(); ++i) {
    // Calculate stride (product of all dimensions after this one)
    int64_t stride = 1;
    for (unsigned j = i + 1; j < shape.size(); ++j) {
      stride *= shape[j];
    }
    
    if (!first) {
      os << " + ";
    }
    first = false;
    
    if (stride == 1) {
      os << "(";
      affineEmitter.emitAffineExpr(results[i]);
      os << ")";
    } else {
      os << "(";
      affineEmitter.emitAffineExpr(results[i]);
      os << ") * " << stride;
    }
  }
}

void XLSModuleEmitter::emitAffineLoad(AffineLoadOp op) {
  indent();
  std::string load_from_name = "";
  if (op->hasAttr("from")) {
    load_from_name = op->getAttr("from").cast<StringAttr>().getValue().str();
  }
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  
  auto memref = op.getMemRef();
  emitValue(memref, 0, false, load_from_name);
  
  auto memrefType = memref.getType().cast<MemRefType>();
  auto attr = memrefType.getMemorySpace();
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(), op.getMapOperands());
  
  // Check for stream memory space
  if (attr && attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    // Stream case - delegate to parent
    allo::vhls::ModuleEmitter::emitAffineLoad(op);
    return;
  }
  
  // For regular memref, handle memory mode vs register mode
  if (USE_MEMORY_FLAG && memrefType.getRank() > 1) {
    // Memory mode with multi-dimensional array: use flattened index
    os << "[";
    emitFlattenedAffineIndex(memrefType, affineMap, affineEmitter);
    os << "]";
  } else {
    // Register mode or 1D array: use normal indexing
    for (auto index : affineMap.getResults()) {
      os << "[";
      affineEmitter.emitAffineExpr(index);
      os << "]";
    }
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void XLSModuleEmitter::emitAffineStore(AffineStoreOp op) {
  indent();
  std::string store_to_name = "";
  if (op->hasAttr("to")) {
    store_to_name = op->getAttr("to").cast<StringAttr>().getValue().str();
  }
  
  auto memref = op.getMemRef();
  emitValue(memref, 0, false, store_to_name);
  
  auto memrefType = memref.getType().cast<MemRefType>();
  auto attr = memrefType.getMemorySpace();
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(), op.getMapOperands());
  
  // Check for stream memory space
  if (attr && attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    // Stream case - delegate to parent
    allo::vhls::ModuleEmitter::emitAffineStore(op);
    return;
  }
  
  // For regular memref, handle memory mode vs register mode
  if (USE_MEMORY_FLAG && memrefType.getRank() > 1) {
    // Memory mode with multi-dimensional array: use flattened index
    os << "[";
    emitFlattenedAffineIndex(memrefType, affineMap, affineEmitter);
    os << "]";
  } else {
    // Register mode or 1D array: use normal indexing
    for (auto index : affineMap.getResults()) {
      os << "[";
      affineEmitter.emitAffineExpr(index);
      os << "]";
    }
  }
  
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  emitInfoAndNewLine(op);
}

void XLSModuleEmitter::emitFunction(func::FuncOp func) {
  if (func->hasAttr("bit"))
    BIT_FLAG = true;

  if (func.getBlocks().empty())
    // This is a declaration.
    return;

  if (func.getBlocks().size() > 1)
    emitError(func, "has more than one basic blocks.");

  if (func->hasAttr("top"))
    os << "/// This is top function.\n";

  // Check for return value first to determine function return type
  auto args = func.getArguments();
  Value returnValue = nullptr;
  std::string otypes = "";
  if (func->hasAttr("otypes"))
    otypes = func->getAttr("otypes").cast<StringAttr>().getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      otypes += "x";
  }
  
  if (auto funcReturn =
          dyn_cast<func::ReturnOp>(func.front().getTerminator())) {
    // For XLS HLS, find the first return value that's not an argument
    for (auto result : funcReturn.getOperands()) {
      if (std::find(args.begin(), args.end(), result) == args.end()) {
        returnValue = result;
        break; // XLS HLS typically supports single return value
      }
    }
  }

  // Check if return value is an array/matrix (ShapedType)
  bool returnIsArray = returnValue && returnValue.getType().isa<ShapedType>() &&
                       !returnValue.getType().cast<ShapedType>().getElementType().isa<StreamType>();

  // Detect if function has array arguments (non-stream memrefs/tensors)
  bool hasArrayArgs = false;
  SmallVector<Value, 4> arrayArgs;
  for (auto &arg : func.getArguments()) {
    auto st = arg.getType().dyn_cast<ShapedType>();
    if (st && !st.getElementType().isa<StreamType>()) {
      hasArrayArgs = true;
      arrayArgs.push_back(arg);
    }
  }
  
  // If function has array arguments, convert to void function with local arrays
  if (hasArrayArgs || returnIsArray) {
    // Clear and set up state
    arrayArgToLocalName.clear();
    arrayArgToChannelName.clear();
    returnArray = nullptr;
    inArrayFunction = true;
    
    // Emit void function with no parameters
    os << "void " << func.getName() << "() {";
    emitInfoAndNewLine(func);
    addIndent();
    
    // Emit function directives
    SmallVector<Value, 8> emptyPorts;
    emitFunctionDirectives(func, emptyPorts);
    
    // Declare local arrays for each array argument
    unsigned argIdx = 0;
    for (auto &arg : arrayArgs) {
      auto st = arg.getType().cast<ShapedType>();
      auto elemType = st.getElementType();
      auto shape = st.getShape();
      
      // Generate local array name (v0, v1, etc.)
      std::string localName = "v" + std::to_string(argIdx);
      std::string channelName = localName + "_in";
      
      // Store mappings
      arrayArgToLocalName[arg] = localName;
      arrayArgToChannelName[arg] = channelName;
      
      // Register the local array name for this argument value
      // This ensures emitLoad will use the local array name instead of the argument name
      addName(arg, /*isPtr=*/false, localName);
      
      // Declare local array - emit comment placeholder if USE_MEMORY_FLAG (wrapper handles class-level decl)
      indent();
      if (USE_MEMORY_FLAG) {
        // Memory mode: emit comment placeholder for wrapper to parse
        // Wrapper will add __xls_memory at class level
        int64_t totalSize = 1;
        for (int64_t dim : shape) totalSize *= dim;
        os << "// __xls_memory_decl__: " << getXLSTypeName(elemType) << ", " << totalSize << ", " << localName;
        for (int64_t dim : shape) os << ", " << dim;
        os << "\n";
      } else {
        // Register mode: plain C array
        os << getXLSTypeName(elemType) << " " << localName;
        for (int64_t dim : shape) {
          os << "[" << dim << "]";
        }
        os << ";\n";
      }
      
      argIdx++;
    }
    
    // Populate arrays from channels
    for (auto &arg : arrayArgs) {
      auto st = arg.getType().cast<ShapedType>();
      auto shape = st.getShape();
      std::string localName = arrayArgToLocalName[arg];
      std::string channelName = arrayArgToChannelName[arg];
      
      // Calculate total size
      int64_t totalSize = 1;
      for (int64_t dim : shape) totalSize *= dim;
      
      // Emit loop to populate from channel
      // Use pipelining instead of unrolling since we're reading from a channel
      indent();
      os << "#pragma hls_pipeline_init_interval 1\n";
      indent();
      os << "for (int _idx = 0; _idx < " << totalSize << "; ++_idx) {\n";
      addIndent();
      indent();
      
      // Generate array access with proper indexing
      os << localName;
      if (USE_MEMORY_FLAG || shape.size() == 1) {
        // Memory mode uses flat indexing (already flattened), or 1D array
        os << "[_idx]";
      } else {
        // Register mode with multi-dimensional: compute indices using division and modulo
        // e.g., for [d0][d1]: v0[_idx / d1][_idx % d1]
        // e.g., for [d0][d1][d2]: v0[_idx / (d1*d2)][(_idx / d2) % d1][_idx % d2]
        for (size_t i = 0; i < shape.size(); ++i) {
          os << "[";
          if (i == shape.size() - 1) {
            // Last dimension: just modulo
            os << "_idx % " << shape[i];
          } else {
            // Calculate divisor (product of all dimensions after this one)
            int64_t divisor = 1;
            for (size_t j = i + 1; j < shape.size(); ++j) {
              divisor *= shape[j];
            }
            if (i == 0) {
              os << "_idx / " << divisor;
            } else {
              os << "(_idx / " << divisor << ") % " << shape[i];
            }
          }
          os << "]";
        }
      }
      os << " = " << channelName << ".read();\n";
      
      reduceIndent();
      indent();
      os << "}\n";
    }
    
    // Track return array if it exists
    if (returnIsArray && returnValue) {
      returnArray = returnValue;
      if (func->hasAttr("outputs")) {
        outputChannelName = func->getAttr("outputs").cast<StringAttr>().getValue().str();
      }
    }
    
    // Emit the function body
    Block &block = func.front();
    allo::vhls::ModuleEmitter::emitBlock(block);
    
    // Handle return value: write array to output channel
    if (returnIsArray && returnValue) {
      auto terminator = block.getTerminator();
      if (auto returnOp = dyn_cast<func::ReturnOp>(terminator)) {
        for (auto operand : returnOp.getOperands()) {
          if (operand == returnValue) {
            auto returnType = returnValue.getType().cast<ShapedType>();
            auto shape = returnType.getShape();
            int64_t totalSize = 1;
            for (int64_t dim : shape) totalSize *= dim;
            
            // Emit loop to write to output channel
            // Use pipelining instead of unrolling since we're writing to a channel
            indent();
            os << "#pragma hls_pipeline_init_interval 1\n";
            indent();
            os << "for (int _idx = 0; _idx < " << totalSize << "; ++_idx) {\n";
            addIndent();
            indent();
            
            // Generate array access with proper indexing
            os << outputChannelName << ".write(";
            emitValue(returnValue);
            if (USE_MEMORY_FLAG || shape.size() == 1) {
              // Memory mode uses flat indexing (already flattened), or 1D array
              os << "[_idx]";
            } else {
              // Register mode with multi-dimensional: compute indices using division and modulo
              for (size_t i = 0; i < shape.size(); ++i) {
                os << "[";
                if (i == shape.size() - 1) {
                  os << "_idx % " << shape[i];
                } else {
                  int64_t divisor = 1;
                  for (size_t j = i + 1; j < shape.size(); ++j) {
                    divisor *= shape[j];
                  }
                  if (i == 0) {
                    os << "_idx / " << divisor;
                  } else {
                    os << "(_idx / " << divisor << ") % " << shape[i];
                  }
                }
                os << "]";
              }
            }
            os << ");\n";
            
            reduceIndent();
            indent();
            os << "}\n";
            break;
          }
        }
      }
    }
    
    // Reset state
    inArrayFunction = false;
    arrayArgToLocalName.clear();
    arrayArgToChannelName.clear();
    returnArray = nullptr;
    
    reduceIndent();
    os << "}\n\n";
    return;
  }

  // COMBINATIONAL MODE: Simple function without arrays
  // Add #pragma hls_top for top-level combinational functions
  if (func->hasAttr("top")) {
    os << "#pragma hls_top\n";
  }
  
  // Emit function signature: void if returning array, otherwise use return type
  if (returnIsArray) {
    // For array/matrix returns, make function void
    os << "void " << func.getName() << "(\n";
  } else if (returnValue) {
    fixUnsignedType(returnValue, otypes.size() > 0 && otypes[0] == 'u');
    os << getXLSTypeName(returnValue.getType()) << " " << func.getName() << "(\n";
  } else {
    os << "void " << func.getName() << "(\n";
  }
  addIndent();

  // This vector is to record all ports of the function.
  SmallVector<Value, 8> portList;

  // Emit input arguments.
  unsigned argIdx = 0;
  std::vector<std::string> input_args;
  if (func->hasAttr("inputs")) {
    std::string input_names =
        func->getAttr("inputs").cast<StringAttr>().getValue().str();
    input_args = split_names(input_names);
  }
  std::string output_names;
  if (func->hasAttr("outputs")) {
    output_names = func->getAttr("outputs").cast<StringAttr>().getValue().str();
    // suppose only one output
    input_args.push_back(output_names);
  }
  std::string itypes = "";
  if (func->hasAttr("itypes"))
    itypes = func->getAttr("itypes").cast<StringAttr>().getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      itypes += "x";
  }
  
  bool hasArgs = func.getNumArguments() > 0;
  for (auto &arg : func.getArguments()) {
    indent();
    fixUnsignedType(arg, itypes[argIdx] == 'u');
    if (arg.getType().isa<ShapedType>()) {
      if (arg.getType().cast<ShapedType>().getElementType().isa<StreamType>()) {
        auto shapedType = arg.getType().dyn_cast<ShapedType>();
        // Use XLS-specific stream type name
        os << getXLSTypeName(arg.getType()) << " ";
        os << addName(arg, false);
        for (auto shape : shapedType.getShape())
          os << "[" << shape << "]";
      } else if (input_args.size() == 0) {
        emitArrayDecl(arg, true);
      } else {
        emitArrayDecl(arg, true, input_args[argIdx]);
      }
    } else {
      if (arg.getType().isa<StreamType>()) {
        // need to pass by reference - use XLS-specific stream type
        os << getXLSTypeName(arg.getType()) << "& ";
        os << addName(arg, false);
      } else if (input_args.size() == 0) {
        emitValue(arg);
      } else {
        emitValue(arg, 0, false, input_args[argIdx]);
      }
    }

    portList.push_back(arg);
    // Add comma if there are more arguments or if we need to add output parameter
    if (argIdx + 1 < func.getNumArguments() || returnIsArray)
      os << ",\n";
    argIdx++;
  }

  // For XLS HLS, if returning an array, add it as output parameter (third parameter)
  std::string outputParamName = "";
  if (returnIsArray && returnValue) {
    indent();
    fixUnsignedType(returnValue, otypes.size() > 0 && otypes[0] == 'u');
    // Generate a name for the output parameter
    if (func->hasAttr("outputs")) {
      outputParamName = func->getAttr("outputs").cast<StringAttr>().getValue().str();
    } else {
      // Use a simple default name - try to get from return value or use "out"
      outputParamName = "out";  // Simple default name
    }
    // Emit output parameter declaration
    auto returnType = returnValue.getType().cast<ShapedType>();
    auto elemType = returnType.getElementType();
    auto shape = returnType.getShape();
    os << getXLSTypeName(elemType) << " " << outputParamName;
    for (int64_t dim : shape)
      os << "[" << dim << "]";
    os << " /* output */";
    
    // Store the output param name in a way we can access it later
    // We'll use a class member or pass it through state - for now use a simple approach
  }

  reduceIndent();
  os << "\n) {";
  emitInfoAndNewLine(func);

  // Emit function body.
  addIndent();

  emitFunctionDirectives(func, portList);
  
  // Emit the function body using base class
  // ReturnOp visitor in base class just returns true without emitting
  Block &block = func.front();
  allo::vhls::ModuleEmitter::emitBlock(block);
  
  // For XLS HLS, handle return value
  if (returnValue) {
    auto terminator = block.getTerminator();
    if (auto returnOp = dyn_cast<func::ReturnOp>(terminator)) {
      if (returnIsArray) {
        // For array returns, copy the return value to the output parameter
        // Compute output parameter name (same logic as above)
        std::string outParamName = "";
        if (func->hasAttr("outputs")) {
          outParamName = func->getAttr("outputs").cast<StringAttr>().getValue().str();
        } else {
          outParamName = "out";  // Simple default name
        }
        
        // Find the return operand that matches returnValue
        for (auto operand : returnOp.getOperands()) {
          if (operand == returnValue) {
            indent();
            os << "// Copy result to output parameter\n";
            // Emit a loop to copy each element
            auto returnType = returnValue.getType().cast<ShapedType>();
            auto shape = returnType.getShape();
            int64_t totalSize = 1;
            for (int64_t dim : shape) totalSize *= dim;
            
            // Emit a simple loop to copy all elements (unrolled by default)
            indent();
            os << "#pragma hls_unroll yes\n";
            indent();
            os << "for (int i = 0; i < " << totalSize << "; ++i) {\n";
            addIndent();
            indent();
            os << outParamName << "[i] = ";
            emitValue(returnValue);
            os << "[i];\n";
            reduceIndent();
            indent();
            os << "}\n";
            break;
          }
        }
      } else if (!returnIsArray) {
        // For non-array returns, emit return statement
        indent();
        os << "return ";
        // Find the operand that matches returnValue, or use first operand
        bool found = false;
        for (auto operand : returnOp.getOperands()) {
          if (operand == returnValue) {
            emitValue(operand);
            found = true;
            break;
          }
        }
        if (!found && returnOp.getNumOperands() > 0) {
          emitValue(returnOp.getOperand(0));
        }
        os << ";\n";
      }
    }
  }

  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

void XLSModuleEmitter::emitModule(ModuleOp module) {
  std::string header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically Generated by Allo (XLS [CC] Backend)
//
//===----------------------------------------------------------------------===//
#include <cstdint>
#include "/xls_builtin.h"
#include "xls_int.h"

template <int Width, bool Signed = true>
using ac_int = XlsInt<Width, Signed>;

)XXX";
  
  os << header;
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<func::FuncOp>(op))
      emitFunction(func);
    else if (auto cst = dyn_cast<memref::GlobalOp>(op))
      emitGlobal(cst);
    else
      emitError(&op, "is unsupported operation.");
  }
}

//===----------------------------------------------------------------------===//
// Entry of allo-translate
//===----------------------------------------------------------------------===//

LogicalResult allo::emitXlsHLS(ModuleOp module, llvm::raw_ostream &os, bool useMemory) {
  USE_MEMORY_FLAG = useMemory;
  AlloEmitterState state(os);
  XLSModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

// Wrapper function for TranslateFromMLIRRegistration (uses default useMemory=false)
static LogicalResult emitXlsHLSDefault(ModuleOp module, llvm::raw_ostream &os) {
  return allo::emitXlsHLS(module, os, /*useMemory=*/false);
}

void allo::registerEmitXlsHLSTranslation() {
  static TranslateFromMLIRRegistration toXLSHLS(
      "emit-XLS-hls", "Emit XLS HLS", emitXlsHLSDefault,
      [&](DialectRegistry &registry) {
        // clang-format off
        registry.insert<
          mlir::allo::AlloDialect,
          mlir::func::FuncDialect,
          mlir::arith::ArithDialect,
          mlir::tensor::TensorDialect,
          mlir::scf::SCFDialect,
          mlir::affine::AffineDialect,
          mlir::math::MathDialect,
          mlir::memref::MemRefDialect,
          mlir::linalg::LinalgDialect
        >();
        // clang-format on
      });
} 