/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_BINDINGS_PYTHON_IRMODULES_H
#define ALLO_BINDINGS_PYTHON_IRMODULES_H

// #include "PybindUtils.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace mlir {
namespace python {

void populateAlloIRTypes(pybind11::module &m);
void populateAlloAttributes(pybind11::module &m);

} // namespace python
} // namespace mlir

#endif // ALLO_BINDINGS_PYTHON_IRMODULES_H