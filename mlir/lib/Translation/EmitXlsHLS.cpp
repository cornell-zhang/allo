/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Based on EmitVivadoHLS.cpp for XLS HLS support
 */

#include <stdexcept>

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
      std::string signedness = "";
      bool is_signed = (intType.getSignedness() != IntegerType::SignednessSemantics::Unsigned);
      return SmallString<16>("ac_int<" +
                              std::to_string(intType.getWidth()) + ", " +
                              (is_signed ? "true" : "false") + ">");
    }
  }

  else if (auto streamType = valType.dyn_cast<StreamType>())
    return SmallString<16>(
        "__xls_channel< " +
        std::string(getCatapultTypeName(streamType.getBaseType()).c_str()) + ", "
        + "__XLS_IO_PLACEHOLDER__" + ">");

  // floating and fixed point
  else
    assert(1 == 0 && "Got unsupported type.");

  return SmallString<16>();
}

// Forward declare the Vivado ModuleEmitter from the Vivado namespace
namespace vhls {
  class ModuleEmitter;
}

namespace {
// XLS ModuleEmitter that inherits from Vivado ModuleEmitter
class XLSModuleEmitter : public allo::vhls::ModuleEmitter {
public:
  using operand_range = Operation::operand_range;
  explicit XLSModuleEmitter(AlloEmitterState &state) : allo::vhls::ModuleEmitter(state) {}

  // Override methods that need XLS-specific behavior
  void emitModule(ModuleOp module) override;
  void emitFunctionDirectives(func::FuncOp func, ArrayRef<Value> portList) override;
  void emitArrayDecl(Value array, bool isFunc = false, std::string name = "") override;
  void emitLoopDirectives(Operation *op);
  void emitStreamConstruct(allo::StreamConstructOp op);
  void emitArrayDirectives(Value memref);
  void emitFunction(func::FuncOp func);

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

void XLSModuleEmitter::void XLSModuleEmitter::emitValue(Value val,
                                 unsigned rank,
                                 bool isPtr,
                                 std::string name) {
  (void)rank; // unused
  assert(!isPtr && "XLS backend does not support raw pointers in emitValue.");
  assert(rank == 0 && "XLS backend does not use rank-based VLAs in emitValue.");

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
    throw std::invalid_argument(
        "XLS[cc] backend only supports statically-sized arrays/memories");
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

      std::string varName = addName(array, /*isPtr=*/false, name);
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
    std::string varName = addName(array, /*isPtr=*/false, name);
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
  std::string varName = addName(array, /*isPtr=*/false, name);
  os << varName;
  for (int64_t dim : shape)
    os << "[" << dim << "]";

  os << " /* tensor-like shaped type */";
}

void XLSModuleEmitter::emitLoopDirectives(Operation *op) {
  if (auto ii = getLoopDirective(op, "pipeline_ii")) {
    reduceIndent();
    indent();
    os << "#pragma hls_pipeline_init_interval " << ii.cast<IntegerAttr>().getValue();
    os << "\n";
    addIndent();
  }

  if (auto factor = getLoopDirective(op, "unroll")) {
    reduceIndent();
    indent();
    os << "#pragma hls_unroll"
        << "\n";
    addIndent();
  }
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
      throw std::invalid_argument(
          "XLS[cc] backend does not support arrays-of-streams; "
          "flatten or convert to separate streams/memory before EmitXLSHLS.");
    }
  }

  // We expect plain StreamType.
  if (!ty.isa<StreamType>()) {
    throw std::invalid_argument(
        "emitStreamConstruct expected a StreamType result for XLS backend.");
  }

  // This uses your getXLSTypeName(StreamType) → __xls_channel<..., __XLS_IO_PLACEHOLDER__>
  emitValue(result);
  os << ";\n";

  // No depth pragmas for XLS; buffering is via __xls_channel or synthesis settings.
  emitInfoAndNewLine(op);
}

// TODO STARTING FROM HERE

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

  // Emit function signature.
  os << "void " << func.getName() << "(\n";
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
    if (argIdx++ != func.getNumArguments() - 1)
      os << ",\n";
  }

  // Emit results.
  auto args = func.getArguments();
  std::string otypes = "";
  if (func->hasAttr("otypes"))
    otypes = func->getAttr("otypes").cast<StringAttr>().getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      otypes += "x";
  }
  if (auto funcReturn =
          dyn_cast<func::ReturnOp>(func.front().getTerminator())) {
    unsigned idx = 0;
    for (auto result : funcReturn.getOperands()) {
      if (std::find(args.begin(), args.end(), result) == args.end()) {
        if (func.getArguments().size() > 0)
          os << ",\n";
        indent();

        // TODO: a known bug, cannot return a value twice, e.g. return %0, %0
        // : index, index. However, typically this should not happen.
        fixUnsignedType(result, otypes[idx] == 'u');
        if (result.getType().isa<ShapedType>()) {
          if (output_names != "")
            emitArrayDecl(result, true);
          else
            emitArrayDecl(result, true, output_names);
        } else {
          // In XLS HLS, pointer indicates the value is an output.
          if (output_names != "")
            emitValue(result, /*rank=*/0, /*isPtr=*/true);
          else
            emitValue(result, /*rank=*/0, /*isPtr=*/true, output_names);
        }

        portList.push_back(result);
      }
      idx += 1;
    }
  } else
    emitError(func, "doesn't have a return operation as terminator.");

  reduceIndent();
  os << "\n) {";
  emitInfoAndNewLine(func);

  // Emit function body.
  addIndent();

  emitFunctionDirectives(func, portList);

  if (func->hasAttr("systolic")) {
    os << "#pragma scop\n";
  }
  emitBlock(func.front());
  if (func->hasAttr("systolic")) {
    os << "#pragma endscop\n";
  }

  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

void XLSModuleEmitter::emitModule(ModuleOp module) {
  std::string device_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for XLS High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_channel.h>
#include <math.h>
#include <stdint.h>
using namespace std;
)XXX";

  std::string host_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for host
//
//===----------------------------------------------------------------------===//
// standard C/C++ headers
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <time.h>

// XLS hls headers
#include "kernel.h"
#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_channel.h>
#include <math.h>
#include <stdint.h>

)XXX";

  if (module.getName().has_value() && module.getName().value() == "host") {
    os << host_header;
    for (auto op : module.getOps<func::FuncOp>()) {
      if (op.getName() == "main")
        emitHostFunction(op);
      else
        emitFunction(op);
    }
  } else {
    os << device_header;
    for (auto &op : *module.getBody()) {
      if (auto func = dyn_cast<func::FuncOp>(op))
        emitFunction(func);
      else if (auto cst = dyn_cast<memref::GlobalOp>(op))
        emitGlobal(cst);
      else
        emitError(&op, "is unsupported operation.");
    }
  }
}

//===----------------------------------------------------------------------===//
// Entry of allo-translate
//===----------------------------------------------------------------------===//

LogicalResult allo::emitXLSHLS(ModuleOp module, llvm::raw_ostream &os) {
  AlloEmitterState state(os);
  XLSModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void allo::registerEmitXLSHLSTranslation() {
  static TranslateFromMLIRRegistration toXLSHLS(
      "emit-XLS-hls", "Emit XLS HLS", emitXLSHLS,
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