/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"

#include "hcl-c/Dialect/HCLTypes.h"
#include "hcl/Bindings/Python/HCLModule.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mlir::python::adaptors;

using namespace mlir;
using namespace mlir::python;

void mlir::python::populateHCLIRTypes(py::module &m) {
  mlir_type_subclass(m, "LoopHandleType", hclMlirTypeIsALoopHandle)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx) {
            return cls(hclMlirLoopHandleTypeGet(ctx));
          },
          "Get an instance of LoopHandleType in given context.", py::arg("cls"),
          py::arg("context") = py::none());

  mlir_type_subclass(m, "OpHandleType", hclMlirTypeIsAOpHandle)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx) {
            return cls(hclMlirOpHandleTypeGet(ctx));
          },
          "Get an instance of OpHandleType in given context.", py::arg("cls"),
          py::arg("context") = py::none());

  mlir_type_subclass(m, "FixedType", hclMlirTypeIsAFixedType)
      .def_classmethod(
          "get",
          [](py::object cls, size_t width, size_t frac, MlirContext ctx) {
            return cls(hclMlirFixedTypeGet(ctx, width, frac));
          },
          "Get an instance of FixedType in given context.", py::arg("cls"),
          py::arg("width"), py::arg("frac"), py::arg("context") = py::none())
      .def_property_readonly(
          "width", [](MlirType type) { return hclMlirFixedTypeGetWidth(type); },
          "Returns the width of the fixed point type")
      .def_property_readonly(
          "frac", [](MlirType type) { return hclMlirFixedTypeGetFrac(type); },
          "Returns the fraction of the fixed point type");

  mlir_type_subclass(m, "UFixedType", hclMlirTypeIsAUFixedType)
      .def_classmethod(
          "get",
          [](py::object cls, size_t width, size_t frac, MlirContext ctx) {
            return cls(hclMlirUFixedTypeGet(ctx, width, frac));
          },
          "Get an instance of FixedType in given context.", py::arg("cls"),
          py::arg("width"), py::arg("frac"), py::arg("context") = py::none())
      .def_property_readonly(
          "width",
          [](MlirType type) { return hclMlirUFixedTypeGetWidth(type); },
          "Returns the width of the fixed point type")
      .def_property_readonly(
          "frac", [](MlirType type) { return hclMlirUFixedTypeGetFrac(type); },
          "Returns the fraction of the fixed point type");
  mlir_type_subclass(m, "StructType", hclMlirTypeIsAStructType)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<MlirType> &members,
             MlirContext ctx) {
            return cls(
                hclMlirStructTypeGet(ctx, members.size(), members.data()));
          },
          "Get an instance of StructType in given context.", py::arg("cls"),
          py::arg("members"), py::arg("context") = py::none())
      .def_property_readonly(
          "field_types",
          [](MlirType type) {
            py::list types;
            unsigned num_fields = hclMlirStructTypeGetNumFields(type);
            for (size_t i = 0; i < num_fields; i++) {
              types.append(hclMlirStructGetEleType(type, i));
            }
            return types;
          },
          "Get a field type of a struct type by index.");
}