/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "allo/Bindings/AlloModule.h"
#include "allo-c/Dialect/AlloAttributes.h"
#include "allo-c/Dialect/AlloTypes.h"
#include "allo-c/Dialect/Dialects.h"
#include "allo-c/Dialect/Registration.h"
#include "allo-c/Translation/EmitCatapultHLS.h"
#include "allo-c/Translation/EmitIntelHLS.h"
#include "allo-c/Translation/EmitTapaHLS.h"
#include "allo-c/Translation/EmitVivadoHLS.h"
#include "allo-c/Translation/EmitXlsHLS.h"
#include "allo/Conversion/Passes.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Support/Liveness.h"
#include "allo/Transforms/Passes.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

using namespace mlir;
using namespace mlir::python;
using namespace allo;

//===----------------------------------------------------------------------===//
// Customized Python classes
//===----------------------------------------------------------------------===//

// NanobindUtils.h
class PyFileAccumulator {
public:
  PyFileAccumulator(const nanobind::object &fileOrStringObject, bool binary)
      : binary(binary) {
    std::string filePath;
    if (nanobind::try_cast<std::string>(fileOrStringObject, filePath)) {
      std::error_code ec;
      writeTarget.emplace<llvm::raw_fd_ostream>(filePath, ec);
      if (ec) {
        throw nanobind::value_error(
            (std::string("Unable to open file for writing: ") + ec.message())
                .c_str());
      }
    } else {
      writeTarget.emplace<nanobind::object>(fileOrStringObject.attr("write"));
    }
  }

  MlirStringCallback getCallback() {
    return writeTarget.index() == 0 ? getPyWriteCallback()
                                    : getOstreamCallback();
  }

  void *getUserData() { return this; }

private:
  MlirStringCallback getPyWriteCallback() {
    return [](MlirStringRef part, void *userData) {
      nanobind::gil_scoped_acquire acquire;
      PyFileAccumulator *accum = static_cast<PyFileAccumulator *>(userData);
      if (accum->binary) {
        // Note: Still has to copy and not avoidable with this API.
        nanobind::bytes pyBytes(part.data, part.length);
        std::get<nanobind::object>(accum->writeTarget)(pyBytes);
      } else {
        nanobind::str pyStr(part.data,
                            part.length); // Decodes as UTF-8 by default.
        std::get<nanobind::object>(accum->writeTarget)(pyStr);
      }
    };
  }

  MlirStringCallback getOstreamCallback() {
    return [](MlirStringRef part, void *userData) {
      PyFileAccumulator *accum = static_cast<PyFileAccumulator *>(userData);
      std::get<llvm::raw_fd_ostream>(accum->writeTarget)
          .write(part.data, part.length);
    };
  }

  std::variant<nanobind::object, llvm::raw_fd_ostream> writeTarget;
  bool binary;
};

//===----------------------------------------------------------------------===//
// Loop transform APIs
//===----------------------------------------------------------------------===//

static bool loopTransformation(MlirModule &mlir_mod) {
  nb::gil_scoped_release release;
  auto mod = unwrap(mlir_mod);
  return applyLoopTransformation(mod);
}

//===----------------------------------------------------------------------===//
// Emission APIs
//===----------------------------------------------------------------------===//

static bool emitVivadoHls(MlirModule &mod, nb::object fileObject,
                          bool flatten = false) {
  PyFileAccumulator accum(fileObject, false);
  nb::gil_scoped_release release;
  return mlirLogicalResultIsSuccess(mlirEmitVivadoHls(
      mod, accum.getCallback(), accum.getUserData(), flatten));
}

static bool emitXlsHls(MlirModule &mod, nb::object fileObject,
                       bool useMemory = false) {
  PyFileAccumulator accum(fileObject, false);
  nb::gil_scoped_release release;
  return mlirLogicalResultIsSuccess(
      mlirEmitXlsHls(mod, accum.getCallback(), accum.getUserData(), useMemory));
}

static bool emitIntelHls(MlirModule &mod, nb::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  nb::gil_scoped_release release;
  return mlirLogicalResultIsSuccess(
      mlirEmitIntelHls(mod, accum.getCallback(), accum.getUserData()));
}

static bool emitTapaHls(MlirModule &mod, nb::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  nb::gil_scoped_release release;
  return mlirLogicalResultIsSuccess(
      mlirEmitTapaHls(mod, accum.getCallback(), accum.getUserData()));
}

static bool emitCatapultHls(MlirModule &mod, nb::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  nb::gil_scoped_release release;
  return mlirLogicalResultIsSuccess(
      mlirEmitCatapultHls(mod, accum.getCallback(), accum.getUserData()));
}

//===----------------------------------------------------------------------===//
// Lowering APIs
//===----------------------------------------------------------------------===//

static bool lowerAlloToLLVM(MlirModule &mlir_mod, MlirContext &mlir_ctx) {
  auto mod = unwrap(mlir_mod);
  auto ctx = unwrap(mlir_ctx);
  return applyAlloToLLVMLoweringPass(mod, *ctx);
}

static bool lowerFixedPointToInteger(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyFixedPointToInteger(mod);
}

static bool lowerAnyWidthInteger(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyAnyWidthInteger(mod);
}

static bool moveReturnToInput(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyMoveReturnToInput(mod);
}

static bool lowerCompositeType(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLowerCompositeType(mod);
}

static bool lowerBitOps(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLowerBitOps(mod);
}

static bool lowerTransformLayoutOps(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLowerTransformLayoutOps(mod);
}

static bool legalizeCast(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLegalizeCast(mod);
}

static bool removeStrideMap(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyRemoveStrideMap(mod);
}

static bool lowerPrintOps(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyLowerPrintOps(mod);
}

//===----------------------------------------------------------------------===//
// Utility pass APIs
//===----------------------------------------------------------------------===//
static bool memRefDCE(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyMemRefDCE(mod);
}

static bool copyOnWrite(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return applyCopyOnWrite(mod);
}

static void copyOnWriteOnFunction(MlirOperation &func) {
  applyCopyOnWriteOnFunction(*unwrap(func));
}

static MlirModule UnifyKernels(MlirModule &mlir_mod1, MlirModule &mlir_mod2,
                               int loop_num) {
  auto mod1 = unwrap(mlir_mod1);
  auto mod2 = unwrap(mlir_mod2);
  return wrap(applyUnifyKernels(mod1, mod2, loop_num));
}

//===----------------------------------------------------------------------===//
// Utility APIs
//===----------------------------------------------------------------------===//
static MlirOperation getFirstUseInFunction(MlirValue value,
                                           MlirOperation &func) {
  Operation *result = getFirstUse(unwrap(value), *unwrap(func));
  return wrap(result);
}

static MlirOperation getLastUseInFunction(MlirValue value,
                                          MlirOperation &func) {
  Operation *result = getLastUse(unwrap(value), *unwrap(func));
  return wrap(result);
}

static MlirOperation getNextUseInFunction(MlirValue value, MlirOperation curUse,
                                          MlirOperation &func) {
  Operation *result = getNextUse(unwrap(value), unwrap(curUse), *unwrap(func));
  return wrap(result);
}

//===----------------------------------------------------------------------===//
// Allo Python module definition
//===----------------------------------------------------------------------===//

NB_MODULE(_allo, m) {
  m.doc() = "Allo Python Native Extension";
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  // register passes
  alloMlirRegisterAllPasses();

  auto allo_m = m.def_submodule("allo");

  // register dialects
  allo_m.def(
      "register_dialect",
      [](MlirContext context) {
        MlirDialectHandle allo = mlirGetDialectHandle__allo__();
        mlir::DialectRegistry registry;
        unwrap(context)->appendDialectRegistry(registry);
        mlirDialectHandleRegisterDialect(allo, context);
        mlirDialectHandleLoadDialect(allo, context);
      },
      nb::arg("context") = nb::none());

  // Apply transform to a design.
  allo_m.def("apply_transform", [](MlirModule &mlir_mod) {
    ModuleOp module = unwrap(mlir_mod);

    // Apply Transform patterns.
    // FIXME: Transform dialect
    // is private within this context.
    /*
    const RaggedArray<mlir::transform::MappedValue> extraMapping = {};
    transform::TransformState state(
        &module.getBodyRegion(), (mlir::Operation*)&module, extraMapping,
        transform::TransformOptions().enableExpensiveChecks());
    for (auto op : llvm::make_early_inc_range(
             module.getBody()->getOps<transform::TransformOpInterface>())) {
      if (failed(state.applyTransform(op).checkAndReport()))
        throw nb::value_error("failed to apply the transform");
      op.erase();
    }

    // Collect PDL patterns to a temporary module.
    OpBuilder b(module);
    auto pdlModule = ModuleOp::create(b.getUnknownLoc());
    b.setInsertionPointToStart(pdlModule.getBody());
    for (auto op : llvm::make_early_inc_range(
             module.getBody()->getOps<pdl::PatternOp>())) {
      op->remove();
      b.insert(op);
    }

    // Apply PDL patterns.
    if (!pdlModule.getBody()->empty()) {
      PDLPatternModule pdlPattern(pdlModule);
      RewritePatternSet patternList(module->getContext());
      patternList.add(std::move(pdlPattern));
      if (failed(applyPatternsAndFoldGreedily(module.getBodyRegion(),
                                              std::move(patternList))))
        throw nb::value_error("failed to apply the PDL pattern");
    }
    */

    // Simplify the loop structure after the transform.
    PassManager pm(module.getContext());
    pm.addNestedPass<func::FuncOp>(
        mlir::affine::createSimplifyAffineStructuresPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(module)))
      throw nb::value_error("failed to apply the post-transform optimization");
  });

  // Declare customized types and attributes
  populateAlloIRTypes(allo_m);
  populateAlloAttributes(allo_m);

  // Loop transform APIs.
  allo_m.def("loop_transformation", &loopTransformation);

  // Codegen APIs.
  allo_m.def("emit_vhls", &emitVivadoHls, nb::arg("module"),
             nb::arg("file_object"), nb::arg("flatten") = false);
  allo_m.def("emit_ihls", &emitIntelHls);
  allo_m.def("emit_thls", &emitTapaHls);
  allo_m.def("emit_xhls", &emitXlsHls, nb::arg("module"),
             nb::arg("file_object"), nb::arg("use_memory") = false);
  allo_m.def("emit_catapult", &emitCatapultHls);

  // LLVM backend APIs.
  allo_m.def("lower_allo_to_llvm", &lowerAlloToLLVM);
  allo_m.def("lower_fixed_to_int", &lowerFixedPointToInteger);
  allo_m.def("lower_anywidth_int", &lowerAnyWidthInteger);
  allo_m.def("move_return_to_input", &moveReturnToInput);

  // Lowering APIs.
  allo_m.def("lower_composite_type", &lowerCompositeType);
  allo_m.def("lower_bit_ops", &lowerBitOps);
  allo_m.def("lower_transform_layout_ops", &lowerTransformLayoutOps);
  allo_m.def("legalize_cast", &legalizeCast);
  allo_m.def("remove_stride_map", &removeStrideMap);
  allo_m.def("lower_print_ops", &lowerPrintOps);

  // Utility pass APIs.
  allo_m.def("memref_dce", &memRefDCE);
  allo_m.def("copy_on_write", &copyOnWrite);
  allo_m.def("unify_kernels", &UnifyKernels);

  allo_m.def("copy_on_write_on_function", &copyOnWriteOnFunction);

  // Utility APIs
  allo_m.def("get_first_use_in_function", &getFirstUseInFunction);
  allo_m.def("get_last_use_in_function", &getLastUseInFunction);
  allo_m.def("get_next_use_in_function", &getNextUseInFunction);
}
