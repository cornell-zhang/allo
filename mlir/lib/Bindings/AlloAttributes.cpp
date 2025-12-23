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
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

void mlir::python::populateAlloAttributes(nb::module_ &m) {
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
