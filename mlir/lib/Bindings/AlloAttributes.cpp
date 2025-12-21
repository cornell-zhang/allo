/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"

#include "allo-c/Dialect/AlloAttributes.h"
#include "allo/Bindings/AlloModule.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

using namespace mlir;
using namespace mlir::python;

namespace nanobind {
namespace detail {

/// Casts object <-> MlirIntegerSet.
template <> struct type_caster<MlirIntegerSet> {
  NB_TYPE_CASTER(MlirIntegerSet, const_name("ir.IntegerSet"));
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (auto capsule = mlirApiObjectToCapsule(src)) {
      value = mlirPythonCapsuleToIntegerSet(capsule->ptr());
      return !mlirIntegerSetIsNull(value);
    }
    return false;
  }
  static handle from_cpp(MlirIntegerSet v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(mlirPythonIntegerSetToCapsule(v));
    return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("IntegerSet")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

} // namespace detail
} // namespace nanobind

void mlir::python::populateAlloAttributes(nb::module_ &m) {
  mlir_attribute_subclass(m, "IntegerSetAttr", mlirAttributeIsAIntegerSet)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirIntegerSet IntegerSet, MlirContext ctx) {
            return cls(mlirIntegerSetAttrGet(IntegerSet));
          },
          nb::arg("cls"), nb::arg("integer_set"),
          nb::arg("context") = nb::none(),
          "Gets an attribute wrapping an IntegerSet.");

  mlir_attribute_subclass(m, "PartitionKindEnum", mlirAttributeIsAPartitionKind)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirAttribute kind, MlirContext ctx) {
            return cls(mlirPartitionKindGet(ctx, kind));
          },
          nb::arg("cls"), nb::arg("kind"), nb::arg("context") = nb::none(),
          "Gets an attribute wrapping a partition kind.");

  mlir_attribute_subclass(m, "NDRangeDimKindEnum",
                          mlirAttributeIsANDRangeDimKind)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirAttribute kind, MlirContext ctx) {
            return cls(mlirNDRangeDimKindGet(ctx, kind));
          },
          nb::arg("cls"), nb::arg("kind"), nb::arg("context") = nb::none(),
          "Gets an attribute wrapping a NDRange dimension kind.");
}
