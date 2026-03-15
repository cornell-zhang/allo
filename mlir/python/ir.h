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
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/unique_ptr.h"
#include "nanobind/stl/vector.h"

#include <tuple>
#include <type_traits>
#include <utility>

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
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/MemOpInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"

#include "allo/Dialect/AlloOps.h"
#include "allo/InitAllDialects.h"
#include "allo/InitAllExtensions.h"
#include "allo/TransformOps/AlloTransformOps.h"

namespace nb = nanobind;

/* Used to dispatch the correct wrapper type for a given mlir::Type or
 * mlir::Attribute. The creator function is expected to take the base mlir::Type
 * or mlir::Attribute and return the appropriate wrapper type.
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

class PyOpRegistry {
public:
  using CreatorFunc = nb::object (*)(mlir::Operation *);

  template <typename ConcreteOp> static void registerOp() {
    registerOp(ConcreteOp::getOperationName(),
               [](mlir::Operation *op) -> nb::object {
                 auto concrete = mlir::dyn_cast<ConcreteOp>(op);
                 if (!concrete)
                   return nb::none();
                 return nb::cast(concrete);
               });
  }

  static void registerOp(std::string_view name, CreatorFunc &&creator) {
    getMap()[llvm::StringRef(name)].push_back(creator);
  }

  static nb::object create(mlir::Operation *op) {
    if (!op)
      return nb::none();
    auto &map = getMap();
    auto name = op->getName().getStringRef();
    auto it = map.find(name);
    if (it != map.end()) {
      for (auto creator = it->second.rbegin(); creator != it->second.rend();
           ++creator) {
        nb::object wrapped = (*creator)(op);
        if (!wrapped.is_none())
          return wrapped;
      }
    }
    return nb::cast(op, nb::rv_policy::reference);
  }

private:
  using CreatorList = llvm::SmallVector<CreatorFunc, 2>;

  static llvm::StringMap<CreatorList> &getMap() {
    static llvm::StringMap<CreatorList> instance;
    return instance;
  }
};

class AlloOpBuilder : public mlir::OpBuilder {
public:
  using OpBuilder::OpBuilder;
  mlir::Location getLocation() const { return loc; }
  void setLocation(mlir::Location newLoc) { loc = newLoc; }
  void setUnknownLoc() { loc = getUnknownLoc(); }
  std::pair<OpBuilder::InsertPoint, mlir::Location>
  getInsertionPointAndLoc() const {
    return {saveInsertionPoint(), loc};
  }
  void setInsertionPointAndLoc(const OpBuilder::InsertPoint &ip,
                               mlir::Location newLoc) {
    restoreInsertionPoint(ip);
    loc = newLoc;
  }

private:
  // default init to unknown
  mlir::Location loc = getUnknownLoc();
};

template <typename Fn>
struct FunctionTraits : FunctionTraits<decltype(&Fn::operator())> {};

template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...) const> {
  using return_type = ReturnType;
  using args_tuple = std::tuple<Args...>;
  static constexpr std::size_t arity = sizeof...(Args);
};

template <typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (*)(Args...)> {
  using return_type = ReturnType;
  using args_tuple = std::tuple<Args...>;
  static constexpr std::size_t arity = sizeof...(Args);
};

template <typename ConcreteOp, typename Base = mlir::OpState>
using OpClass = nb::class_<ConcreteOp, Base>;

template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base> bindOp(nb::module_ &m, const char *pyName) {
  PyOpRegistry::registerOp<ConcreteOp>();
  return nb::class_<ConcreteOp, Base>(m, pyName);
}

template <typename ConcreteOp, typename Class, typename BuilderFn,
          std::size_t... I, typename... NbArgs>
inline Class &bindConstructorImpl(Class &cls, BuilderFn &&builderFn,
                                  std::index_sequence<I...>,
                                  NbArgs &&...nbArgs) {
  using FnTraits = FunctionTraits<std::decay_t<BuilderFn>>;
  using ArgsTuple = typename FnTraits::args_tuple;
  using ReturnType = typename FnTraits::return_type;
  static_assert(
      std::is_same_v<std::remove_cv_t<std::remove_reference_t<ReturnType>>,
                     ConcreteOp>,
      "builder init lambda must return the concrete op type");
  static_assert(
      std::is_same_v<std::remove_cv_t<std::remove_reference_t<
                         std::tuple_element_t<0, ArgsTuple>>>,
                     AlloOpBuilder>,
      "builder init lambda must take AlloOpBuilder as its first argument");

  return cls.def(
      "__init__",
      [builderFn = std::forward<BuilderFn>(builderFn)](
          ConcreteOp &self, AlloOpBuilder &builder,
          std::tuple_element_t<I + 1, ArgsTuple>... args) {
        self = builderFn(builder, args...);
      },
      nb::arg("builder"), std::forward<NbArgs>(nbArgs)...);
}

template <typename ConcreteOp, typename Class, typename BuilderFn,
          typename... NbArgs>
inline Class &bindConstructor(Class &cls, BuilderFn &&builderFn,
                              NbArgs &&...nbArgs) {
  using FnTraits = FunctionTraits<std::decay_t<BuilderFn>>;
  static_assert(FnTraits::arity >= 1,
                "builder init lambda must take AlloOpBuilder");
  return bindConstructorImpl<ConcreteOp>(
      cls, std::forward<BuilderFn>(builderFn),
      std::make_index_sequence<FnTraits::arity - 1>{},
      std::forward<NbArgs>(nbArgs)...);
}

template <typename ConcreteOp, typename Base, typename BuilderFn,
          typename... NbArgs>
inline OpClass<ConcreteOp, Base> &
bindConstructor(OpClass<ConcreteOp, Base> &cls, BuilderFn &&builderFn,
                NbArgs &&...nbArgs) {
  using FnTraits = FunctionTraits<std::decay_t<BuilderFn>>;
  static_assert(FnTraits::arity >= 1,
                "builder init lambda must take AlloOpBuilder");
  return bindConstructorImpl<ConcreteOp>(
      cls, std::forward<BuilderFn>(builderFn),
      std::make_index_sequence<FnTraits::arity - 1>{},
      std::forward<NbArgs>(nbArgs)...);
}

template <typename Class, typename Getter>
inline Class &bindGetter(Class &cls, const char *name, Getter &&getter) {
  return cls.def(name, std::forward<Getter>(getter), nb::rv_policy::reference);
}

///===--------------------------------------------------------------------===//
/// Common op builder patterns
///===--------------------------------------------------------------------===//

// Bind a unary op that takes a single mlir::Value operand
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base> bindUnaryValueOp(nb::module_ &m,
                                                  const char *pyName,
                                                  const char *argName = "val") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &value) {
        return ConcreteOp::create(builder, builder.getLocation(), value);
      },
      nb::arg(argName));
  return cls;
}

// Bind a binary op that takes two mlir::Value operands
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindBinaryValueOp(nb::module_ &m, const char *pyName,
                  const char *lhsName = "lhs", const char *rhsName = "rhs") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs) {
        return ConcreteOp::create(builder, builder.getLocation(), lhs, rhs);
      },
      nb::arg(lhsName), nb::arg(rhsName));
  return cls;
}

// Bind an op that takes a mlir::Value operand and a mlir::ValueRange operand
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindValueRangeOp(nb::module_ &m, const char *pyName,
                 const char *valueName = "value",
                 const char *rangeName = "values") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &value,
         const std::vector<mlir::Value> &range) {
        return ConcreteOp::create(builder, builder.getLocation(), value, range);
      },
      nb::arg(valueName), nb::arg(rangeName));
  return cls;
}

// Bind an op that takes two mlir::Value operands and a mlir::ValueRange operand
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base> bindTwoValueRangeOp(
    nb::module_ &m, const char *pyName, const char *firstName = "first",
    const char *secondName = "second", const char *rangeName = "values") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &first, mlir::Value &second,
         const std::vector<mlir::Value> &range) {
        return ConcreteOp::create(builder, builder.getLocation(), first, second,
                                  range);
      },
      nb::arg(firstName), nb::arg(secondName), nb::arg(rangeName));
  return cls;
}

// Bind an op that takes a mlir::Value operand
// and a mlir::Value operand as init value
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindUnaryInitOp(nb::module_ &m, const char *pyName,
                const char *inputName = "input",
                const char *initName = "init") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &input, mlir::Value &init) {
        return ConcreteOp::create(builder, builder.getLocation(), input, init);
      },
      nb::arg(inputName), nb::arg(initName));
  return cls;
}

// Bind an op that takes two mlir::Value operands
// and a mlir::Value operand as init value
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindBinaryInputsInitOp(nb::module_ &m, const char *pyName,
                       const char *lhsName = "lhs", const char *rhsName = "rhs",
                       const char *initName = "init") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
         mlir::Value &init) {
        return ConcreteOp::create(builder, builder.getLocation(),
                                  std::initializer_list<mlir::Value>{lhs, rhs},
                                  init);
      },
      nb::arg(lhsName), nb::arg(rhsName), nb::arg(initName));
  return cls;
}

// Bind an op that takes a mlir::Value operand and a mlir::Type operand,
// where the value is the source and the type is the destination type (e.g. a
// cast op)
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindSourceToTypeOp(nb::module_ &m, const char *pyName,
                   const char *srcName = "src",
                   const char *dstTypeName = "dst_type") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &src, mlir::Type &dstType) {
        return ConcreteOp::create(builder, builder.getLocation(), dstType, src);
      },
      nb::arg(srcName), nb::arg(dstTypeName));
  return cls;
}

// Bind an op that takes a mlir::Type operand and a mlir::Value operand,
// where the type is the destination type and the value is the source (e.g. a
// cast op)
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindTypeToSourceOp(nb::module_ &m, const char *pyName,
                   const char *dstTypeName = "dst_type",
                   const char *srcName = "src") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Type &dstType, mlir::Value &src) {
        return ConcreteOp::create(builder, builder.getLocation(), dstType, src);
      },
      nb::arg(dstTypeName), nb::arg(srcName));
  return cls;
}

void bindIR(nb::module_ &m);
void bindMathOps(nb::module_ &m);
void bindArithOps(nb::module_ &m);
void bindSCFOps(nb::module_ &m);
void bindCFOps(nb::module_ &m);
void bindFuncOps(nb::module_ &m);
void bindAffineOps(nb::module_ &m);
void bindTensorOps(nb::module_ &m);
void bindMemRefOps(nb::module_ &m);
void bindLinalgOps(nb::module_ &m);
void bindTransform(nb::module_ &m);
void bindUBOps(nb::module_ &m);

inline mlir::OpPrintingFlags getOpPrintingFlags(bool debug = false) {
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.enableDebugInfo(debug);
  printingFlags.printNameLocAsPrefix(true);
  printingFlags.printGenericOpForm(false);
  return printingFlags;
}

#endif // ALLO_PYTHON_IR_H
