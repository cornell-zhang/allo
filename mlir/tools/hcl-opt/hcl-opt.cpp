/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/TransformOps/HCLTransformOps.h"

#include "hcl/Conversion/Passes.h"
#include "hcl/Support/Utils.h"
#include "hcl/Transforms/Passes.h"

#include <iostream>

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    preloadDialectsInContext("preload-dialects-in-context",
                             llvm::cl::desc("Preloads dialects in context"),
                             llvm::cl::init(false));

static llvm::cl::opt<bool> enableOpt("opt",
                                     llvm::cl::desc("Enable HCL schedules"),
                                     llvm::cl::init(false));

static llvm::cl::opt<bool> lowerToLLVM("lower-to-llvm",
                                       llvm::cl::desc("Lower to LLVM Dialect"),
                                       llvm::cl::init(false));

static llvm::cl::opt<bool>
    lowerComposite("lower-composite", llvm::cl::desc("Lower composite types"),
                   llvm::cl::init(false));

static llvm::cl::opt<bool> lowerBitOps("lower-bitops",
                                       llvm::cl::desc("Lower bitops"),
                                       llvm::cl::init(false));

static llvm::cl::opt<bool> legalizeCast("legalize-cast",
                                        llvm::cl::desc("Legalize cast"),
                                        llvm::cl::init(false));

static llvm::cl::opt<bool> removeStrideMap("remove-stride-map",
                                           llvm::cl::desc("Remove stride map"),
                                           llvm::cl::init(false));

static llvm::cl::opt<bool> lowerPrintOps("lower-print-ops",
                                         llvm::cl::desc("Lower print ops"),
                                         llvm::cl::init(false));

static llvm::cl::opt<bool> bufferization("bufferization",
                                         llvm::cl::desc("Bufferization"),
                                         llvm::cl::init(false));

static llvm::cl::opt<bool> linalgConversion("linalg-to-affine",
                                            llvm::cl::desc("Linalg to affine"),
                                            llvm::cl::init(false));

static llvm::cl::opt<bool> dataPlacement("data-placement",
                                         llvm::cl::desc("Data placement"),
                                         llvm::cl::init(false));

static llvm::cl::opt<bool>
    enableNormalize("normalize",
                    llvm::cl::desc("Enable other common optimizations"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool> runJiT("jit", llvm::cl::desc("Run JiT compiler"),
                                  llvm::cl::init(false));

static llvm::cl::opt<bool> fixedPointToInteger(
    "fixed-to-integer",
    llvm::cl::desc("Lower fixed-point operations to integer"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    anyWidthInteger("lower-anywidth-integer",
                    llvm::cl::desc("Lower anywidth integer to 64-bit integer"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool> moveReturnToInput(
    "return-to-input",
    llvm::cl::desc("Move return values to input argument list"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    memRefDCE("memref-dce",
              llvm::cl::desc("Remove memrefs that are never loaded from"),
              llvm::cl::init(false));

static llvm::cl::opt<bool>
    applyTransform("apply-transform",
                   llvm::cl::desc("Apply pattern-based transformations"),
                   llvm::cl::init(false));

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  module = parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int runJiTCompiler(mlir::ModuleOp module) {

  std::string LLVM_BUILD_DIR;
  bool found = mlir::hcl::getEnv("LLVM_BUILD_DIR", LLVM_BUILD_DIR);
  if (!found) {
    llvm::errs() << "Error: LLVM_BUILD_DIR not found\n";
  }
  std::string HCL_DIALECT_BUILD_DIR;
  found = mlir::hcl::getEnv("HCL_DIALECT_BUILD_DIR", HCL_DIALECT_BUILD_DIR);
  if (!found) {
    llvm::errs() << "Error: HCL_DIALECT_BUILD_DIR not found\n";
  }
  std::string runner_utils = LLVM_BUILD_DIR + "/lib/libmlir_runner_utils.so";
  std::string c_runner_utils =
      LLVM_BUILD_DIR + "/lib/libmlir_c_runner_utils.so";
  std::string hcl_runtime_lib =
      HCL_DIALECT_BUILD_DIR + "/lib/libhcl_runtime_utils.so";
  llvm::SmallVector<std::string, 4> shared_libs = {runner_utils, c_runner_utils,
                                                   hcl_runtime_lib};
  llvm::SmallVector<llvm::SmallString<256>, 4> libPaths;
  // Use absolute library path so that gdb can find the symbol table.
  transform(shared_libs, std::back_inserter(libPaths), [](std::string libPath) {
    llvm::SmallString<256> absPath(libPath.begin(), libPath.end());
    cantFail(llvm::errorCodeToError(llvm::sys::fs::make_absolute(absPath)));
    return absPath;
  });

  // Libraries that we'll pass to the ExecutionEngine for loading.
  llvm::SmallVector<llvm::StringRef, 4> executionEngineLibs;

  using MlirRunnerInitFn = void (*)(llvm::StringMap<void *> &);
  using MlirRunnerDestroyFn = void (*)();

  llvm::StringMap<void *> exportSymbols;
  llvm::SmallVector<MlirRunnerDestroyFn> destroyFns;

  // Handle libraries that do support mlir-runner init/destroy callbacks.
  for (auto &libPath : libPaths) {
    auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(libPath.c_str());
    void *initSym = lib.getAddressOfSymbol("__mlir_runner_init");
    void *destroySim = lib.getAddressOfSymbol("__mlir_runner_destroy");

    // Library does not support mlir runner, load it with ExecutionEngine.
    if (!initSym || !destroySim) {
      executionEngineLibs.push_back(libPath);
      continue;
    }

    auto initFn = reinterpret_cast<MlirRunnerInitFn>(initSym);
    initFn(exportSymbols);

    auto destroyFn = reinterpret_cast<MlirRunnerDestroyFn>(destroySim);
    destroyFns.push_back(destroyFn);
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/1, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.sharedLibPaths = executionEngineLibs;
  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("top");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int main(int argc, char **argv) {
  // Register dialects and passes in current context
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::hcl::HeteroCLDialect>();
  mlir::hcl::registerTransformDialectExtension(registry);

  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);
  context.allowUnregisteredDialects(true);
  context.printOpOnDiagnostic(true);
  context.loadAllAvailableDialects();

  mlir::registerAllPasses();
  mlir::hcl::registerHCLPasses();
  mlir::hcl::registerHCLConversionPasses();

  // Parse pass names in main to ensure static initialization completed
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR modular optimizer driver\n");

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadMLIR(context, module))
    return error;

  // Initialize a pass manager
  // https://mlir.llvm.org/docs/PassManagement/
  // Operation agnostic passes
  mlir::PassManager pm(&context);
  // Operation specific passes
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  if (enableOpt) {
    pm.addPass(mlir::hcl::createLoopTransformationPass());
  }

  if (dataPlacement) {
    pm.addPass(mlir::hcl::createDataPlacementPass());
  }

  if (memRefDCE) {
    pm.addPass(mlir::hcl::createMemRefDCEPass());
  }

  if (lowerComposite) {
    pm.addPass(mlir::hcl::createLowerCompositeTypePass());
  }

  if (fixedPointToInteger) {
    pm.addPass(mlir::hcl::createFixedPointToIntegerPass());
  }

  // lowerPrintOps should be run after lowering fixed point to integer
  if (lowerPrintOps) {
    pm.addPass(mlir::hcl::createLowerPrintOpsPass());
  }

  if (anyWidthInteger) {
    pm.addPass(mlir::hcl::createAnyWidthIntegerPass());
  }

  if (moveReturnToInput) {
    pm.addPass(mlir::hcl::createMoveReturnToInputPass());
  }

  if (lowerBitOps) {
    pm.addPass(mlir::hcl::createLowerBitOpsPass());
  }

  if (legalizeCast) {
    pm.addPass(mlir::hcl::createLegalizeCastPass());
  }

  if (removeStrideMap) {
    pm.addPass(mlir::hcl::createRemoveStrideMapPass());
  }

  if (bufferization) {
    pm.addPass(mlir::bufferization::createOneShotBufferizePass());
  }

  if (linalgConversion) {
    optPM.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  }

  if (enableNormalize) {
    // To make all loop steps to 1.
    optPM.addPass(mlir::affine::createAffineLoopNormalizePass());

    // Sparse Conditional Constant Propagation (SCCP)
    pm.addPass(mlir::createSCCPPass());

    // To factor out the redundant AffineApply/AffineIf operations.
    // optPM.addPass(mlir::createCanonicalizerPass());
    // optPM.addPass(mlir::createSimplifyAffineStructuresPass());

    // To simplify the memory accessing.
    pm.addPass(mlir::memref::createNormalizeMemRefsPass());

    // Generic common sub expression elimination.
    // pm.addPass(mlir::createCSEPass());
  }

  if (applyTransform)
    pm.addPass(mlir::hcl::createTransformInterpreterPass());

  if (runJiT || lowerToLLVM) {
    if (!removeStrideMap) {
      pm.addPass(mlir::hcl::createRemoveStrideMapPass());
    }
    pm.addPass(mlir::hcl::createHCLToLLVMLoweringPass());
  }

  // Run the pass pipeline
  if (mlir::failed(pm.run(*module))) {
    return 4;
  }

  // print output
  std::string errorMessage;
  auto outfile = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!outfile) {
    llvm::errs() << errorMessage << "\n";
    return 2;
  }
  module->print(outfile->os());
  outfile->os() << "\n";

  // run JiT
  if (runJiT)
    return runJiTCompiler(*module);

  return 0;
}