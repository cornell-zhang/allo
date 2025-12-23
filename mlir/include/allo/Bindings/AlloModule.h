/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_BINDINGS_PYTHON_IRMODULES_H
#define ALLO_BINDINGS_PYTHON_IRMODULES_H

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace mlir {
namespace python {

void populateAlloIRTypes(nanobind::module_ &m);
void populateAlloAttributes(nanobind::module_ &m);

} // namespace python
} // namespace mlir

#endif // ALLO_BINDINGS_PYTHON_IRMODULES_H