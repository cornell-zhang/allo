/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Based on EmitVivadoHLS.cpp for Catapult HLS support
 */

#include "allo/Translation/EmitCatapultHLS.h"
#include "allo/Dialect/Visitor.h"
#include "allo/Support/Utils.h"
#include "allo/Translation/EmitVivadoHLS.h" // Include Vivado emitter
#include "allo/Translation/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

static SmallString<16> getCatapultTypeName(Type valType) {
  if (auto arrayType = llvm::dyn_cast<ShapedType>(valType))
    valType = arrayType.getElementType();

  // Handle float types.
  if (llvm::isa<Float16Type>(valType))
    return SmallString<16>("half");
  else if (llvm::isa<Float32Type>(valType))
    return SmallString<16>("float");
  else if (llvm::isa<Float64Type>(valType))
    return SmallString<16>("double");

  // Handle integer types.
  else if (llvm::isa<IndexType>(valType))
    return SmallString<16>("int");
  else if (auto intType = llvm::dyn_cast<IntegerType>(valType)) {
    if (intType.getWidth() == 1) {
      if (!BIT_FLAG)
        return SmallString<16>("bool");
      else
        return SmallString<16>("ac_int<1, false>");
    } else {
      std::string signedness = "";
      bool is_signed = (intType.getSignedness() !=
                        IntegerType::SignednessSemantics::Unsigned);
      if (!BIT_FLAG) {
        switch (intType.getWidth()) {
        case 8:
        case 16:
        case 32:
        case 64:
          if (!is_signed)
            signedness = "u";
          return SmallString<16>(signedness + "int" +
                                 std::to_string(intType.getWidth()) + "_t");
        default:
          return SmallString<16>("ac_int<" +
                                 std::to_string(intType.getWidth()) + ", " +
                                 (is_signed ? "true" : "false") + ">");
        }
      } else {
        return SmallString<16>("ac_int<" + std::to_string(intType.getWidth()) +
                               ", " + (is_signed ? "true" : "false") + ">");
      }
    }
  }

  // Handle (custom) fixed point types.
  else if (auto fixedType = llvm::dyn_cast<allo::FixedType>(valType))
    return SmallString<16>(
        "ac_fixed<" + std::to_string(fixedType.getWidth()) + ", " +
        std::to_string(fixedType.getWidth() - fixedType.getFrac()) + ", true>");

  else if (auto ufixedType = llvm::dyn_cast<allo::UFixedType>(valType))
    return SmallString<16>(
        "ac_fixed<" + std::to_string(ufixedType.getWidth()) + ", " +
        std::to_string(ufixedType.getWidth() - ufixedType.getFrac()) +
        ", false>");

  else if (auto streamType = llvm::dyn_cast<StreamType>(valType))
    return SmallString<16>(
        "ac_channel< " +
        std::string(getCatapultTypeName(streamType.getBaseType()).c_str()) +
        " >");

  else
    assert(1 == 0 && "Got unsupported type.");

  return SmallString<16>();
}

namespace {
// Catapult ModuleEmitter that inherits from Vivado HLS ModuleEmitter
class CatapultModuleEmitter : public allo::hls::VhlsModuleEmitter {
public:
  using operand_range = Operation::operand_range;
  explicit CatapultModuleEmitter(AlloEmitterState &state)
      : allo::hls::VhlsModuleEmitter(state) {}

  // Override methods that need Catapult-specific behavior
  void emitModule(ModuleOp module) override;
  void emitFunctionDirectives(func::FuncOp func,
                              ArrayRef<Value> portList) override;
  void emitArrayDecl(Value array, bool isFunc = false,
                     std::string name = "") override;
  void emitLoopDirectives(Operation *op) override;
  void emitStreamConstruct(allo::StreamConstructOp op) override;
  void emitArrayDirectives(Value memref) override;
  void emitFunction(func::FuncOp func) override;

protected:
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "") override;
  // Helper method to get Catapult-specific type names
  SmallString<16> getTypeName(Type valType) {
    return getCatapultTypeName(valType);
  }
  SmallString<16> getTypeName(Value val) {
    return getCatapultTypeName(val.getType());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Catapult-specific implementations
//===----------------------------------------------------------------------===//

void CatapultModuleEmitter::emitValue(Value val, unsigned rank, bool isPtr,
                                      std::string name) {

  assert(!(rank && isPtr) && "should be either an array or a pointer.");

  // Value has been declared before or is a constant number.
  if (isDeclared(val)) {
    os << getName(val);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
    return;
  }

  os << getCatapultTypeName(val.getType()) << " ";

  if (name == "") {
    // Add the new value to nameTable and emit its name.
    os << addName(val, isPtr);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
  } else {
    os << addName(val, isPtr, name);
  }
}

void CatapultModuleEmitter::emitFunctionDirectives(func::FuncOp func,
                                                   ArrayRef<Value> portList) {
  // For Catapult HLS, emit the hls_design top pragma for top-level functions
  if (func->hasAttr("top")) {
    indent();
    os << "#pragma hls_design top\n";
    os << "\n";
  }

  // Emit other function-level directives as needed
  if (func->hasAttr("dataflow")) {
    indent();
    os << "#pragma hls_design dataflow\n";
  }

  if (func->hasAttr("inline")) {
    indent();
    os << "#pragma hls_design inline\n";
  }

  // Emit array directives for function ports
  for (auto &port : portList)
    if (llvm::isa<MemRefType>(port.getType()))
      emitArrayDirectives(port);
}

void CatapultModuleEmitter::emitArrayDecl(Value array, bool isFunc,
                                          std::string name) {
  assert(!isDeclared(array) && "has been declared before.");

  auto arrayType = llvm::cast<ShapedType>(array.getType());
  if (arrayType.hasStaticShape()) {
    auto memref = llvm::dyn_cast<MemRefType>(array.getType());
    if (memref) {
      auto attr = memref.getMemorySpace();
      // Use dyn_cast to safely check if attr is a StringAttr (it could be
      // IntegerAttr)
      auto strAttr = attr ? llvm::dyn_cast<StringAttr>(attr) : nullptr;
      if (strAttr && strAttr.getValue().str().substr(0, 6) == "stream") {
        // Value has been declared before or is a constant number.
        if (isDeclared(array)) {
          os << getName(array);
          return;
        }

        // print stream type using ac_channel instead of hls::stream
        os << "ac_channel< " << getCatapultTypeName(arrayType.getElementType())
           << " > ";

        auto attr_str = strAttr.getValue().str();
        int S_index = attr_str.find("S"); // spatial
        int T_index = attr_str.find("T"); // temporal
        if (isFunc &&
            !(((int)(arrayType.getShape().size()) > T_index - S_index) &&
              (T_index > S_index))) {
          os << "&"; // pass by reference, only non-array needs reference
        }

        // Add the new value to nameTable and emit its name.
        os << addName(array, /*isPtr=*/false, name);
        if ((int)(arrayType.getShape().size()) > T_index - S_index) {
          for (int i = 0; i < T_index - S_index; ++i)
            os << "[" << arrayType.getShape()[i] << "]";
        }
        // Add original array declaration as comment
        os << " /* ";
        emitValue(array, 0, false, name);
        for (auto &shape : arrayType.getShape())
          os << "[" << shape << "]";
        os << " */";
      } else {
        emitValue(array, 0, false, name);
        for (auto &shape : arrayType.getShape())
          os << "[" << shape << "]";
      }
    } else { // tensor
      emitValue(array, 0, false, name);
    }
  } else
    emitValue(array, /*rank=*/0, /*isPtr=*/true, name);
}

void CatapultModuleEmitter::emitLoopDirectives(Operation *op) {
  if (auto ii = getLoopDirective(op, "pipeline_ii")) {
    reduceIndent();
    indent();
    os << "#pragma hls_pipeline_init_interval "
       << llvm::cast<IntegerAttr>(ii).getValue();
    os << "\n";
    addIndent();
  }

  if (auto factor = getLoopDirective(op, "unroll")) {
    reduceIndent();
    indent();
    auto val = llvm::cast<IntegerAttr>(factor).getValue();
    if (val == 0)
      os << "#pragma hls_unroll"
         << "\n";
    else
      os << "#pragma hls_unroll " << val << "\n";
    addIndent();
  }

  if (auto parallel = getLoopDirective(op, "parallel")) {
    reduceIndent();
    indent();
    // parallel implies full unroll
    os << "#pragma hls_unroll"
       << "\n";
    addIndent();
  }

  if (auto dataflow = getLoopDirective(op, "dataflow")) {
    reduceIndent();
    indent();
    os << "#pragma hls_design dataflow\n";
    addIndent();
  }
}

void CatapultModuleEmitter::emitStreamConstruct(allo::StreamConstructOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  if (auto shapedType = llvm::dyn_cast<ShapedType>(result.getType())) {
    for (auto shape : shapedType.getShape()) {
      os << "[" << shape << "]";
    }
  }
  os << ";\n";
  // Note: Catapult HLS doesn't need explicit stream depth pragmas like Vivado
  // HLS The depth is handled through the ac_channel template parameter or
  // synthesis settings
  emitInfoAndNewLine(op);
}

void CatapultModuleEmitter::emitArrayDirectives(Value memref) {
  bool emitPragmaFlag = false;
  auto type = llvm::cast<MemRefType>(memref.getType());

  // streaming
  auto attr = type.getMemorySpace();
  if (attr) {
    // Use dyn_cast to safely check if attr is a StringAttr (it could be
    // IntegerAttr)
    auto strAttr = llvm::dyn_cast<StringAttr>(attr);
    if (strAttr) {
      std::string attr_str = strAttr.getValue().str();
      if (attr_str.substr(0, 6) == "stream") {
        // Note: Catapult HLS doesn't need explicit stream pragmas like Vivado
        // HLS The streaming behavior is handled through ac_channel type
        return;
      }
    }
  }

  // For other array directives, delegate to the parent implementation
  // but we need to call the parent method explicitly
  // allo::hls::VhlsModuleEmitter::emitArrayDirectives(memref);
  // Catapult ignores #pragma HLS array_partition.
  // TODO: Implement Catapult-specific memory directives (e.g. via TCL or other
  // pragmas)
}

void CatapultModuleEmitter::emitFunction(func::FuncOp func) {
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
        llvm::cast<StringAttr>(func->getAttr("inputs")).getValue().str();
    input_args = split_names(input_names);
  }
  std::string output_names;
  if (func->hasAttr("outputs")) {
    output_names =
        llvm::cast<StringAttr>(func->getAttr("outputs")).getValue().str();
    // suppose only one output
    input_args.push_back(output_names);
  }
  std::string itypes = "";
  if (func->hasAttr("itypes"))
    itypes = llvm::cast<StringAttr>(func->getAttr("itypes")).getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      itypes += "x";
  }
  for (auto &arg : func.getArguments()) {
    indent();
    fixUnsignedType(arg, itypes[argIdx] == 'u');
    if (llvm::isa<ShapedType>(arg.getType())) {
      if (llvm::isa<StreamType>(
              llvm::cast<ShapedType>(arg.getType()).getElementType())) {
        auto shapedType = llvm::dyn_cast<ShapedType>(arg.getType());
        // Use Catapult-specific stream type name
        os << getCatapultTypeName(arg.getType()) << " ";
        os << addName(arg, false);
        for (auto shape : shapedType.getShape())
          os << "[" << shape << "]";
      } else if (input_args.size() == 0) {
        emitArrayDecl(arg, true);
      } else {
        emitArrayDecl(arg, true, input_args[argIdx]);
      }
    } else {
      if (llvm::isa<StreamType>(arg.getType())) {
        // need to pass by reference - use Catapult-specific stream type
        os << getCatapultTypeName(arg.getType()) << "& ";
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
    otypes = llvm::cast<StringAttr>(func->getAttr("otypes")).getValue().str();
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
        if (llvm::isa<ShapedType>(result.getType())) {
          if (output_names != "")
            emitArrayDecl(result, true);
          else
            emitArrayDecl(result, true, output_names);
        } else {
          // In Catapult HLS, pointer indicates the value is an output.
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

void CatapultModuleEmitter::emitModule(ModuleOp module) {
  std::string device_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for Catapult High-level Synthesis (HLS).
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

// catapult hls headers
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

LogicalResult allo::emitCatapultHLS(ModuleOp module, llvm::raw_ostream &os) {
  AlloEmitterState state(os);
  CatapultModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void allo::registerEmitCatapultHLSTranslation() {
  static TranslateFromMLIRRegistration toCatapultHLS(
      "emit-catapult-hls", "Emit Catapult HLS", emitCatapultHLS,
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