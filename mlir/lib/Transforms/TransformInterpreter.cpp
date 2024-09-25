/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"
#include "hcl/Transforms/Passes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

using namespace mlir;
using namespace hcl;

namespace {
struct TransformInterpreter
    : public hcl::TransformInterpreterBase<TransformInterpreter> {
  void runOnOperation() override;
};
} // namespace

void TransformInterpreter::runOnOperation() {
  ModuleOp module = getOperation();
  // transform::TransformState state(
  //     module.getBodyRegion(), module,
  //     transform::TransformOptions().enableExpensiveChecks());
  // for (auto op : module.getBody()->getOps<transform::TransformOpInterface>())
  // {
  //   if (failed(state.applyTransform(op).checkAndReport()))
  //     return signalPassFailure();
  // }
}

std::unique_ptr<OperationPass<ModuleOp>> hcl::createTransformInterpreterPass() {
  return std::make_unique<TransformInterpreter>();
}
