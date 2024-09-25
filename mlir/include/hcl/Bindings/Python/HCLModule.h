/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_BINDINGS_PYTHON_IRMODULES_H
#define HCL_BINDINGS_PYTHON_IRMODULES_H

// #include "PybindUtils.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace mlir {
namespace python {

void populateHCLIRTypes(pybind11::module &m);
void populateHCLAttributes(pybind11::module &m);

} // namespace python
} // namespace mlir

#endif // HCL_BINDINGS_PYTHON_IRMODULES_H