/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hcl/Dialect/TransformOps/HCLTransformOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

namespace {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// HCLParentLoopOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HCLParentLoopOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  SetVector<Operation *> parents;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    affine::AffineForOp loop;
    Operation *current = target;
    for (unsigned i = 0, e = getNumLoops(); i < e; ++i) {
      loop = current->getParentOfType<affine::AffineForOp>();
      if (!loop) {
        DiagnosedSilenceableFailure diag =
            emitSilenceableError()
            << "could not find an '" << affine::AffineForOp::getOperationName()
            << "' parent";
        diag.attachNote(target->getLoc()) << "target op";
        return diag;
      }
      current = loop;
    }
    parents.insert(loop);
  }
  results.set(getResult().cast<OpResult>(), parents.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// HCLUnrollOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HCLUnrollOp::applyToOne(transform::TransformRewriter &rewriter,
                                   affine::AffineForOp target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  if (failed(loopUnrollByFactor(target, getFactor()))) {
    Diagnostic diag(target->getLoc(), DiagnosticSeverity::Note);
    diag << "op failed to unroll";
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// HCLSplitOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HCLSplitOp::applyToOne(transform::TransformRewriter &rewriter,
                                  affine::AffineForOp target,
                                  transform::ApplyToEachResultList &results,
                                  transform::TransformState &state) {
  SmallVector<affine::AffineForOp, 2> splittedLoop;
  if (failed(tilePerfectlyNested({target}, {(unsigned)getFactor()},
                                 &splittedLoop))) {
    Diagnostic diag(target->getLoc(), DiagnosticSeverity::Note);
    diag << "op failed to split";
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }
  results.push_back(splittedLoop.front());
  results.push_back(splittedLoop.back());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// HCLPipelineOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HCLPipelineOp::applyToOne(transform::TransformRewriter &rewriter,
                                     affine::AffineForOp target,
                                     transform::ApplyToEachResultList &results,
                                     transform::TransformState &state) {
  Builder b(target.getContext());
  target->setAttr("pipeline_ii", b.getI32IntegerAttr(getInitialInterval()));
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class HCLTransformDialectExtension
    : public transform::TransformDialectExtension<
          HCLTransformDialectExtension> {
public:
  HCLTransformDialectExtension() {
    declareDependentDialect<affine::AffineDialect>();
    declareDependentDialect<func::FuncDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "hcl/Dialect/TransformOps/HCLTransformOps.cpp.inc"
        >();
  }
};
} // namespace

// mlir/lib/Dialect/Transform/IR/TransformOps.cpp
static ParseResult parseSequenceOpOperands(
    OpAsmParser &parser, std::optional<OpAsmParser::UnresolvedOperand> &root,
    Type &rootType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &extraBindings,
    SmallVectorImpl<Type> &extraBindingTypes) {
  OpAsmParser::UnresolvedOperand rootOperand;
  OptionalParseResult hasRoot = parser.parseOptionalOperand(rootOperand);
  if (!hasRoot.has_value()) {
    root = std::nullopt;
    return success();
  }
  if (failed(hasRoot.value()))
    return failure();
  root = rootOperand;

  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseOperandList(extraBindings)))
      return failure();
  }
  if (failed(parser.parseColon()))
    return failure();

  // The paren is truly optional.
  (void)parser.parseOptionalLParen();

  if (failed(parser.parseType(rootType))) {
    return failure();
  }

  if (!extraBindings.empty()) {
    if (parser.parseComma() || parser.parseTypeList(extraBindingTypes))
      return failure();
  }

  if (extraBindingTypes.size() != extraBindings.size()) {
    return parser.emitError(parser.getNameLoc(),
                            "expected types to be provided for all operands");
  }

  // The paren is truly optional.
  (void)parser.parseOptionalRParen();
  return success();
}

static void printSequenceOpOperands(OpAsmPrinter &printer, Operation *op,
                                    Value root, Type rootType,
                                    ValueRange extraBindings,
                                    TypeRange extraBindingTypes) {
  if (!root)
    return;

  printer << root;
  bool hasExtras = !extraBindings.empty();
  if (hasExtras) {
    printer << ", ";
    printer.printOperands(extraBindings);
  }

  printer << " : ";
  if (hasExtras)
    printer << "(";

  printer << rootType;
  if (hasExtras) {
    printer << ", ";
    llvm::interleaveComma(extraBindingTypes, printer.getStream());
    printer << ")";
  }
}

static ParseResult parseForeachMatchSymbols(OpAsmParser &parser,
                                            ArrayAttr &matchers,
                                            ArrayAttr &actions) {
  StringAttr matcher;
  StringAttr action;
  SmallVector<Attribute> matcherList;
  SmallVector<Attribute> actionList;
  do {
    if (parser.parseSymbolName(matcher) || parser.parseArrow() ||
        parser.parseSymbolName(action)) {
      return failure();
    }
    matcherList.push_back(SymbolRefAttr::get(matcher));
    actionList.push_back(SymbolRefAttr::get(action));
  } while (parser.parseOptionalComma().succeeded());

  matchers = parser.getBuilder().getArrayAttr(matcherList);
  actions = parser.getBuilder().getArrayAttr(actionList);
  return success();
}

/// Prints the comma-separated list of symbol reference pairs of the format
/// `@matcher -> @action`.
static void printForeachMatchSymbols(OpAsmPrinter &printer, Operation *op,
                                     ArrayAttr matchers, ArrayAttr actions) {
  printer.increaseIndent();
  printer.increaseIndent();
  for (auto &&[matcher, action, idx] : llvm::zip_equal(
           matchers, actions, llvm::seq<unsigned>(0, matchers.size()))) {
    printer.printNewline();
    printer << cast<SymbolRefAttr>(matcher) << " -> "
            << cast<SymbolRefAttr>(action);
    if (idx != matchers.size() - 1)
      printer << ", ";
  }
  printer.decreaseIndent();
  printer.decreaseIndent();
}

#define GET_OP_CLASSES
#include "hcl/Dialect/TransformOps/HCLTransformOps.cpp.inc"

void mlir::hcl::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<HCLTransformDialectExtension>();
}
