/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_PYTHON_IR_H
#define ALLO_PYTHON_IR_H

#include "nanobind/nanobind.h"
#include "nanobind/stl/function.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "nanobind/stl/unique_ptr.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Format.h"

#include "allo/Dialect/AlloOps.h"
#include "allo/InitAllDialects.h"
#include "allo/InitAllExtensions.h"
#include "allo/TransformOps/AlloTransformOps.h"

namespace nb = nanobind;

/* Used to dispatch the correct wrapper type for a given mlir::Type or mlir::Attribute. The
 * creator function is expected to take the base mlir::Type or mlir::Attribute and
 * return the appropriate wrapper type.
 */
class PyTypeRegistry {
public:
  using CreatorFunc = nb::object (*)(mlir::Type);

  template <typename ConcreteType> static void registerType() {
    registerType(mlir::TypeID::get<ConcreteType>(),
                 [](mlir::Type t) -> nb::object {
                   return nb::cast(mlir::cast<ConcreteType>(t));
                 });
  }

  static void registerType(mlir::TypeID id, CreatorFunc &&creator) {
    getMap()[id] = creator;
  }

  static nb::object create(mlir::Type t) {
    if (!t)
      return nb::none();
    auto &map = getMap();
    auto id = t.getTypeID();
    auto it = map.find(id);
    if (it != map.end()) {
      return it->second(t);
    }
    return nb::cast(t);
  }

private:
  static llvm::DenseMap<mlir::TypeID, CreatorFunc> &getMap() {
    static llvm::DenseMap<mlir::TypeID, CreatorFunc> instance;
    return instance;
  }
};

class PyAttributeRegistry {
public:
  using CreatorFunc = nb::object (*)(mlir::Attribute);
  template <typename ConcreteAttr> static void registerAttr() {
    registerAttr(mlir::TypeID::get<ConcreteAttr>(),
                 [](mlir::Attribute a) -> nb::object {
                   return nb::cast(mlir::cast<ConcreteAttr>(a));
                 });
  }
  static void registerAttr(mlir::TypeID id, CreatorFunc &&creator) {
    getMap()[id] = creator;
  }
  static nb::object create(mlir::Attribute a) {
    if (!a)
      return nb::none();
    auto &map = getMap();
    auto id = a.getTypeID();
    auto it = map.find(id);
    if (it != map.end()) {
      return it->second(a);
    }
    return nb::cast(a);
  }

private:
  static llvm::DenseMap<mlir::TypeID, CreatorFunc> &getMap() {
    static llvm::DenseMap<mlir::TypeID, CreatorFunc> instance;
    return instance;
  }
};

class AlloOpBuilder : public mlir::OpBuilder {
public:
  using OpBuilder::OpBuilder;
  mlir::Location get_loc() const { return loc; }
  void set_loc(mlir::Location newLoc) { loc = newLoc; }
  void set_unknown_loc() { loc = getUnknownLoc(); }
  std::pair<OpBuilder::InsertPoint, mlir::Location>
  get_insertion_point_and_loc() const {
    return {saveInsertionPoint(), loc};
  }
  void set_insertion_point_and_loc(const OpBuilder::InsertPoint &ip,
                                   mlir::Location newLoc) {
    restoreInsertionPoint(ip);
    loc = newLoc;
  }

private:
  // default init to unknown
  mlir::Location loc = getUnknownLoc();
};

void init_ir(nb::module_ &m);
void init_math_ops(nb::module_ &m);
void init_arith_ops(nb::module_ &m);
void init_scf_ops(nb::module_ &m);
void init_cf_ops(nb::module_ &m);
void init_ub_ops(nb::module_ &m);
void init_func_ops(nb::module_ &m);
void init_affine_ops(nb::module_ &m);
void init_tensor_ops(nb::module_ &m);
void init_memref_ops(nb::module_ &m);
void init_linalg_ops(nb::module_ &m);
void init_allo_ir(nb::module_ &m);
void init_transform(nb::module_ &m);
void init_allo_transforms(nb::module_ &m);

#endif // ALLO_PYTHON_IR_H
