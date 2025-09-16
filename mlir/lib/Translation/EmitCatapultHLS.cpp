/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Based on EmitVivadoHLS.cpp for Catapult HLS support
 */

#include "allo/Translation/EmitCatapultHLS.h"
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

static SmallString<16> getCatapultTypeName(Type valType) {
  if (auto arrayType = valType.dyn_cast<ShapedType>())
    valType = arrayType.getElementType();

  // Handle float types.
  if (valType.isa<Float16Type>())
    return SmallString<16>("half");
  else if (valType.isa<Float32Type>())
    return SmallString<16>("float");
  else if (valType.isa<Float64Type>())
    return SmallString<16>("double");

  // Handle integer types.
  else if (valType.isa<IndexType>())
    return SmallString<16>("int");
  else if (auto intType = valType.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 1) {
      if (!BIT_FLAG)
        return SmallString<16>("bool");
      else
        return SmallString<16>("ac_int<1, false>");
    } else {
      std::string signedness = "";
      bool is_signed = (intType.getSignedness() != IntegerType::SignednessSemantics::Unsigned);
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
        return SmallString<16>("ac_int<" +
                               std::to_string(intType.getWidth()) + ", " +
                               (is_signed ? "true" : "false") + ">");
      }
    }
  }

  // Handle (custom) fixed point types.
  else if (auto fixedType = valType.dyn_cast<allo::FixedType>())
    return SmallString<16>(
        "ac_fixed<" + std::to_string(fixedType.getWidth()) + ", " +
        std::to_string(fixedType.getWidth() - fixedType.getFrac()) + ", true>");

  else if (auto ufixedType = valType.dyn_cast<allo::UFixedType>())
    return SmallString<16>(
        "ac_fixed<" + std::to_string(ufixedType.getWidth()) + ", " +
        std::to_string(ufixedType.getWidth() - ufixedType.getFrac()) + ", false>");

  else if (auto streamType = valType.dyn_cast<StreamType>())
    return SmallString<16>(
        "ac_channel< " +
        std::string(getCatapultTypeName(streamType.getBaseType()).c_str()) + " >");

  else
    assert(1 == 0 && "Got unsupported type.");

  return SmallString<16>();
}

// Forward declare the Vivado ModuleEmitter from the Vivado namespace
namespace vhls {
  class ModuleEmitter;
}

namespace {
// Catapult ModuleEmitter that inherits from Vivado ModuleEmitter
class CatapultModuleEmitter : public allo::vhls::ModuleEmitter {
public:
  using operand_range = Operation::operand_range;
  explicit CatapultModuleEmitter(AlloEmitterState &state) : allo::vhls::ModuleEmitter(state) {}

  // Override methods that need Catapult-specific behavior
  void emitModule(ModuleOp module) override;
  void emitFunctionDirectives(func::FuncOp func, ArrayRef<Value> portList) override;
  void emitArrayDecl(Value array, bool isFunc = false, std::string name = "") override;

protected:
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "");
  // Helper method to get Catapult-specific type names
  SmallString<16> getTypeName(Type valType) { return getCatapultTypeName(valType); }
  SmallString<16> getTypeName(Value val) { return getCatapultTypeName(val.getType()); }
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
    if (port.getType().isa<MemRefType>())
      emitArrayDirectives(port);
}

void CatapultModuleEmitter::emitArrayDecl(Value array, bool isFunc, std::string name) {
  if (auto shapedType = array.getType().dyn_cast<ShapedType>()) {
    if (name.empty()) {
      os << getCatapultTypeName(array.getType().cast<ShapedType>().getElementType());
      // For Catapult, use shaped arrays instead of pointers for function parameters
      os << " " << addName(array, false);
      for (auto dim : shapedType.getShape()) {
        os << "[" << dim << "]";
      }
    } else {
      os << getCatapultTypeName(array.getType().cast<ShapedType>().getElementType());
      // For Catapult, use shaped arrays instead of pointers for function parameters  
      os << " " << name;
      for (auto dim : shapedType.getShape()) {
        os << "[" << dim << "]";
      }
    }
  }
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