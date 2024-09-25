/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"

#include "hcl-c/Dialect/HCLAttributes.h"
#include "hcl/Bindings/Python/HCLModule.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mlir::python::adaptors;

using namespace mlir;
using namespace mlir::python;

namespace pybind11 {
namespace detail {

/// Casts object <-> MlirIntegerSet.
template <> struct type_caster<MlirIntegerSet> {
  PYBIND11_TYPE_CASTER(MlirIntegerSet, _("MlirIntegerSet"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToIntegerSet(capsule.ptr());
    if (mlirIntegerSetIsNull(value)) {
      return false;
    }
    return true;
  }
  static handle cast(MlirIntegerSet v, return_value_policy, handle) {
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonIntegerSetToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("IntegerSet")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

} // namespace detail
} // namespace pybind11

void mlir::python::populateHCLAttributes(py::module &m) {
  mlir_attribute_subclass(m, "IntegerSetAttr", mlirAttributeIsAIntegerSet)
      .def_classmethod(
          "get",
          [](py::object cls, MlirIntegerSet IntegerSet, MlirContext ctx) {
            return cls(mlirIntegerSetAttrGet(IntegerSet));
          },
          py::arg("cls"), py::arg("integer_set"),
          py::arg("context") = py::none(),
          "Gets an attribute wrapping an IntegerSet.");

  mlir_attribute_subclass(m, "PartitionKindEnum", mlirAttributeIsAPartitionKind)
      .def_classmethod(
          "get",
          [](py::object cls, MlirAttribute kind, MlirContext ctx) {
            return cls(mlirPartitionKindGet(ctx, kind));
          },
          py::arg("cls"), py::arg("kind"), py::arg("context") = py::none(),
          "Gets an attribute wrapping a partition kind.");

  mlir_attribute_subclass(m, "NDRangeDimKindEnum",
                          mlirAttributeIsANDRangeDimKind)
      .def_classmethod(
          "get",
          [](py::object cls, MlirAttribute kind, MlirContext ctx) {
            return cls(mlirNDRangeDimKindGet(ctx, kind));
          },
          py::arg("cls"), py::arg("kind"), py::arg("context") = py::none(),
          "Gets an attribute wrapping a NDRange dimension kind.");
}
