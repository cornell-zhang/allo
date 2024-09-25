/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_TRANSFORM_OPS
#define HCL_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def HCLParentLoopOp : Op<Transform_Dialect, "hcl.parent_loop",
    [NavigationTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Gets a handle to the parent 'for' loop of the given operation";
  let description = [{
    Produces a handle to the n-th (default 1) parent `affine.for` loop for each
    Payload IR operation associated with the operand. Fails if such a loop
    cannot be found. The list of operations associated with the handle contains
    parent operations in the same order as the list associated with the operand,
    except for operations that are parents to more than one input which are only
    present once.
  }];

  let arguments = (ins PDL_Operation:$target, DefaultValuedAttr<
                       ConfinedAttr<I64Attr, [IntPositive]>, "1">:$num_loops);
  let results = (outs PDL_Operation:$parent);

  let assemblyFormat = "$target attr-dict";
}

def HCLUnrollOp : Op<Transform_Dialect, "hcl.unroll",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Unrolls the given loop with the given unroll factor";
  let description = [{
    Unrolls each loop associated with the given handle to have up to the given
    number of loop body copies per iteration. If the unroll factor is larger
    than the loop trip count, the latter is used as the unroll factor instead.
    Does not produce a new handle as the operation may result in the loop being
    removed after a full unrolling.
  }];

  let arguments = (ins PDL_Operation:$target,
                       ConfinedAttr<I64Attr, [IntPositive]>:$factor);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter, 
        ::mlir::affine::AffineForOp target, 
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def HCLSplitOp : Op<Transform_Dialect, "hcl.split",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Split the given loop with the given split factor";
  let description = [{
    Split each loop associated with the given handle to two loops. Produce two
    handles cooresponding to the two loops generated.
  }];

  let arguments = (ins PDL_Operation:$target,
                       ConfinedAttr<I64Attr, [IntPositive]>:$factor);
  let results = (outs PDL_Operation:$outer, PDL_Operation:$inner);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter, 
        ::mlir::affine::AffineForOp target, 
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def HCLPipelineOp : Op<Transform_Dialect, "hcl.pipeline",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Pipeline the given loop with the given initial interval";
  let description = [{
    Pipeline each loop associated with the given handle. Produce the pipelined
    loop.
  }];

  let arguments = (ins PDL_Operation:$target,
                       ConfinedAttr<I64Attr, [IntPositive]>:$initialInterval);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter, 
        ::mlir::affine::AffineForOp target, 
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // HCL_TRANSFORM_OPS
