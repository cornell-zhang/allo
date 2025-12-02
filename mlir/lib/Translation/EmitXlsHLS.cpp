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
#include "llvm/Support/raw_ostream.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"

using namespace mlir;
using namespace allo;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// used for determine whether to generate C++ default types or ac_(u)int
static bool BIT_FLAG = false;

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
        break;x
      case 64:
        if (intType.getSignedness() == IntegerType::SignednessSemantics::Unsigned)
          return SmallString<16>("uint64_t");
        else
          return SmallString<16>("int64_t");
        break;
      default:
        return SmallString<16>("ac_int<" + std::to_string(intType.getWidth()) + ", " +
                                (intType.getSignedness() == IntegerType::SignednessSemantics::Unsigned ? "false" : "true") + ">");
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

protected:
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "");
  // Helper method to get XLS-specific type names
  SmallString<16> getTypeName(Type valType) { return getXLSTypeName(valType); }
  SmallString<16> getTypeName(Value val) { return getXLSTypeName(val.getType()); }
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

    // 3) DEFAULT MEMREF -> plain C array: T name[d0][d1]...
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
  
  // Emit lower bound.
  os << getXLSTypeName(iterVar.getType()) << " ";
  emitValue(iterVar, 0, false, loop_name);
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
// Standard C/C++ headers
#include <cstdint>

// XLS [CC] headers
#include "/xls_builtin.h"  // NOLINT
#include "xls_int.h"       // NOLINT

// Templated Types for XLS [CC]
template<typename T>
using InputChannel = __xls_channel<T, __xls_channel_dir_In>;
template<typename T>
using OutputChannel = __xls_channel<T, __xls_channel_dir_Out>;
template<typename T, int Size>
using Memory = __xls_memory<T, Size>;
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

LogicalResult allo::emitXlsHLS(ModuleOp module, llvm::raw_ostream &os) {
  AlloEmitterState state(os);
  XLSModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void allo::registerEmitXlsHLSTranslation() {
  static TranslateFromMLIRRegistration toXLSHLS(
      "emit-XLS-hls", "Emit XLS HLS", emitXlsHLS,
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