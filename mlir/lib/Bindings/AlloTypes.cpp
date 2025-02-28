/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"

#include "allo-c/Dialect/AlloTypes.h"
#include "allo/Bindings/AlloModule.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mlir::python::adaptors;

using namespace mlir;
using namespace mlir::python;

void mlir::python::populateAlloIRTypes(py::module &m) {
  mlir_type_subclass(m, "LoopHandleType", alloMlirTypeIsALoopHandle)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx) {
            return cls(alloMlirLoopHandleTypeGet(ctx));
          },
          "Get an instance of LoopHandleType in given context.", py::arg("cls"),
          py::arg("context") = py::none());

  mlir_type_subclass(m, "OpHandleType", alloMlirTypeIsAOpHandle)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx) {
            return cls(alloMlirOpHandleTypeGet(ctx));
          },
          "Get an instance of OpHandleType in given context.", py::arg("cls"),
          py::arg("context") = py::none());

  mlir_type_subclass(m, "FixedType", alloMlirTypeIsAFixedType)
      .def_classmethod(
          "get",
          [](py::object cls, size_t width, size_t frac, MlirContext ctx) {
            return cls(alloMlirFixedTypeGet(ctx, width, frac));
          },
          "Get an instance of FixedType in given context.", py::arg("cls"),
          py::arg("width"), py::arg("frac"), py::arg("context") = py::none())
      .def_property_readonly(
          "width",
          [](MlirType type) { return alloMlirFixedTypeGetWidth(type); },
          "Returns the width of the fixed point type")
      .def_property_readonly(
          "frac", [](MlirType type) { return alloMlirFixedTypeGetFrac(type); },
          "Returns the fraction of the fixed point type");

  mlir_type_subclass(m, "UFixedType", alloMlirTypeIsAUFixedType)
      .def_classmethod(
          "get",
          [](py::object cls, size_t width, size_t frac, MlirContext ctx) {
            return cls(alloMlirUFixedTypeGet(ctx, width, frac));
          },
          "Get an instance of FixedType in given context.", py::arg("cls"),
          py::arg("width"), py::arg("frac"), py::arg("context") = py::none())
      .def_property_readonly(
          "width",
          [](MlirType type) { return alloMlirUFixedTypeGetWidth(type); },
          "Returns the width of the fixed point type")
      .def_property_readonly(
          "frac", [](MlirType type) { return alloMlirUFixedTypeGetFrac(type); },
          "Returns the fraction of the fixed point type");

  mlir_type_subclass(m, "StructType", alloMlirTypeIsAStructType)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<MlirType> &members,
             MlirContext ctx) {
            return cls(
                alloMlirStructTypeGet(ctx, members.size(), members.data()));
          },
          "Get an instance of StructType in given context.", py::arg("cls"),
          py::arg("members"), py::arg("context") = py::none())
      .def_property_readonly(
          "field_types",
          [](MlirType type) {
            py::list types;
            unsigned num_fields = alloMlirStructTypeGetNumFields(type);
            for (size_t i = 0; i < num_fields; i++) {
              types.append(alloMlirStructGetEleType(type, i));
            }
            return types;
          },
          "Get a field type of a struct type by index.");

  mlir_type_subclass(m, "StreamType", alloMlirTypeIsAStreamType)
      .def_classmethod(
          "get",
          [](py::object cls, MlirType &baseType, size_t depth,
             MlirContext ctx) {
            return cls(alloMlirStreamTypeGet(ctx, baseType, depth));
          },
          "Get an instance of StreamType in given context.", py::arg("cls"),
          py::arg("base_type"), py::arg("depth"),
          py::arg("context") = py::none())
      .def_property_readonly(
          "base_type",
          [](MlirType type) { return alloMlirStreamTypeGetBaseType(type); },
          "Returns the base type of the stream object")
      .def_property_readonly(
          "depth",
          [](MlirType type) { return alloMlirStreamTypeGetDepth(type); },
          "Returns the depth of the stream");
}