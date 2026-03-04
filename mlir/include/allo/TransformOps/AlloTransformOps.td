/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSFORM_OPS
#define ALLO_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "allo/Dialect/AlloAttrs.td"

def Allo_RaiseToAffineOp : Op<Transform_Dialect, "allo.raise_to_affine",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   TransformOpInterface, TransformEachOpTrait,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Raise loop ops to affine form";
  let description = [{
    Input:
      - `$target` is a handle to one or more loop ops.
      - Each payload op must be one of:
        `scf.for`, `scf.parallel`, `affine.for`, or `affine.parallel`.
      - `scf.parallel` with reductions is rejected.

    Behavior:
      - `scf.for` is raised to `affine.for` when bounds/step match affine
        patterns.
      - `scf.parallel` is raised to `affine.parallel` when each dimension's
        bounds/step matches affine patterns.
      - For already-affine loop ops, the loop itself is kept and memref access
        ops inside are raised/canonicalized to affine access ops when possible.

    Output:
      - `$result` contains one payload op per input payload op (same cardinality
        as `$target`).
  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = [{
    $target attr-dict `:` type($target) `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value":$target), [{
      auto anyType = transform::AnyOpType::get(target.getContext());
      build($_builder, $_state, anyType, target);
    }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def Allo_OutlineOp : Op<Transform_Dialect, "allo.outline",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Outline payload ops into `allo.kernel` + `allo.call`";
  let description = [{
    Input:
      - `$target` is a handle to one or more payload operations.
      - Each payload operation must have at most one region.
      - `$kernel_name` is the symbol name used for the outlined kernel.

    Behavior:
      - Each target is outlined independently.
      - The outlined function is converted to `allo.kernel`.
      - The original target site is replaced by an `allo.call` to that kernel.

    Output:
      - `$kernel` contains created `allo.kernel` ops.
      - `$call` contains created `allo.call` ops.
      - Both result handles have one payload op per input payload op.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target, StrAttr:$kernel_name);
  let results = (outs TransformHandleTypeInterface:$kernel, TransformHandleTypeInterface:$call);
  let assemblyFormat = [{
    $target `to` $kernel_name attr-dict `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$target, "StringRef":$kernel_name), [{
      auto kernelType = $_builder.getType<transform::OperationType>("allo.kernel");
      auto callType = $_builder.getType<transform::OperationType>("allo.call");
      build($_builder, $_state, {kernelType, callType}, target, kernel_name);
    }]>
  ];
}

def Allo_LoopReorderOp : Op<Transform_Dialect, "allo.reorder",
  [NavigationTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Reorder selected loops in an affine perfect band";
  let description = [{
    Input:
      - `$loops` must resolve to at least two `affine.for` ops.
      - The selected loops may be unordered and non-contiguous in the handle.
      - All selected loops must belong to one perfectly nested affine loop band.
      - `$permutation` must be a valid permutation whose length equals the
        number of selected loops.

    Behavior:
      - Reorder is applied to the selected subset.
      - Unselected loops in the same perfect band keep their relative positions.
      - Legality is checked with affine loop interchange constraints.

    Output:
      - No result handle; this op mutates payload IR in place.
  }];
  let arguments = (ins TransformHandleTypeInterface:$loops, I64ArrayAttr:$permutation);
  let results = (outs);
  let assemblyFormat = [{
    $loops `with` $permutation attr-dict `:` type($loops)
  }];
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "Value":$loops, "ArrayRef<int64_t>":$permutation), [{
      auto attr = $_builder.getI64ArrayAttr(permutation);
      build($_builder, $_state, loops, attr);
    }]>
  ];
}

def Allo_LoopSplitOp : Op<Transform_Dialect, "allo.split",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   TransformOpInterface, TransformEachOpTrait,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Split each target loop into outer/inner loops";
  let description = [{
    Input:
      - `$loop` is a handle to one or more loop ops.
      - Each payload op must be `affine.for` or `scf.for`.
      - `$factor` must be a positive integer.
      - If loop range is statically known, `$factor` must be strictly smaller
        than the loop trip count.

    Behavior:
      - Each payload loop is split independently (`TransformEachOpTrait`).
      - Input `sym_name` is preserved with suffixes `.outer`/`.inner`.

    Output:
      - `$outer` contains outer loops, one per input payload loop.
      - `$inner` contains inner loops, one per input payload loop.
  }];
  let arguments = (ins TransformHandleTypeInterface:$loop, I64Attr:$factor);
  let results = (outs TransformHandleTypeInterface:$outer, TransformHandleTypeInterface:$inner);
  let assemblyFormat = [{
    $loop `with` $factor attr-dict `:` functional-type(operands, results)
  }];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
  let builders = [
    OpBuilder<(ins "Value":$loop, "int64_t":$factor), [{
      auto anyType = transform::AnyOpType::get($_builder.getContext());
      build($_builder, $_state, anyType, anyType, loop, static_cast<uint64_t>(factor));
    }]>
  ];
}

def Allo_LoopTileOp : Op<Transform_Dialect, "allo.tile",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Tile a loop nest and return tile/point loops";
  let description = [{
    Input:
      - `$loop` is a handle to one or more loops, typically created by
        `transform.merge_handles`.
      - Payload loops must be all `affine.for` or all `scf.for` (mixing
        dialects is rejected).
      - `$factors` must be positive, and its length must match the number of
        payload loops.
      - Payload loops may be unordered; factors are first bound by handle
        iteration order, then loop/factor pairs are sorted by loop depth.
      - Loops must be unique and form one ancestor chain (same loop nest).

    Behavior:
      - Uses perfect or imperfect tiling based on selected-loop contiguity.
      - If input loops carry `sym_name`, outputs are annotated with `.tile` and
        `.point` suffixes.

    Output:
      - `$tile_loops` and `$point_loops` each contain one loop per selected
        input loop.
      - Both result handles are reported in depth order.
  }];
  let arguments = (ins TransformHandleTypeInterface:$loops, I64ArrayAttr:$factors);
  let results = (outs TransformHandleTypeInterface:$tile_loops, TransformHandleTypeInterface:$point_loops);
  let assemblyFormat = [{
    $loops `with` $factors attr-dict `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$loops, "ArrayRef<int64_t>":$factors), [{
      auto anyType = transform::AnyOpType::get($_builder.getContext());
      auto factorsAttr = $_builder.getI64ArrayAttr(factors);
      build($_builder, $_state, anyType, anyType, loops, factorsAttr);
    }]>
  ];
}

def Allo_LoopFlattenOp : Op<Transform_Dialect, "allo.flatten",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Flatten the given loop nest";
  let description = [{
    Input:
      - `$loops` must resolve to `affine.for` ops only.
      - Selected loops may be unordered and non-contiguous in the handle.
      - If two or more loops are selected, they must be in one perfectly nested
        affine loop band.

    Behavior:
      - Selected loops are sorted by depth.
      - The op flattens the contiguous perfect band from selected outermost loop
        to selected innermost loop (intermediate loops are included).
      - Flattening requires normalized loops:
        step = 1, constant lower bound = 0, constant positive upper bound.
      - The multiplied trip count must fit in int64.
      - If fewer than two loops are selected, this op is a no-op.

    Output:
      - `$result` contains the flattened outermost loop.
      - For no-op cases (`<2` selected loops), `$result` forwards input payload
        loops unchanged.
  }];
  let arguments = (ins TransformHandleTypeInterface:$loops);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = [{
    $loops attr-dict `:` type($loops) `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value":$loops), [{
      auto anyType = transform::AnyOpType::get($_builder.getContext());
      build($_builder, $_state, anyType, loops);
    }]>
  ];
}

def Allo_ComputeAtOp : Op<Transform_Dialect, "allo.compute_at",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Attach a producer loop nest under a consumer affine loop";
  let description = [{
    Input:
      - `$producer` resolves to exactly one payload op.
      - `$consumer_loop` resolves to exactly one `affine.for`.
      - `$producer` must be enclosed by an affine loop nest.

    Behavior:
      - If RAW dependence exists, rely on affine producer-consumer fusion.
      - If no dependence exists, move the producer subtree under
        `$consumer_loop`, with optional `affine.if` guarding for subset bounds.
      - Loop-IV remapping is applied for the common prefix depth.

    Handle effects:
      - Consumes only `$producer`.
      - `$consumer_loop` is read-only and remains reusable.
  }];
  let arguments = (ins TransformHandleTypeInterface:$producer, TransformHandleTypeInterface:$consumer_loop);
  let results = (outs);
  let assemblyFormat = [{
    $producer `at` $consumer_loop attr-dict `:` functional-type(operands, results)
  }];
}

def Allo_ReuseAtOp : Op<Transform_Dialect, "allo.reuse_at",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Create a reuse buffer at a given affine loop axis";
  let description = [{
    Input:
      - `$target` resolves to exactly one memref value, or one op with a
        single memref result.
      - `$axis` resolves to exactly one `affine.for`.

    Behavior:
      - Analyze accesses of `$target` under the loop stage containing `$axis`.
      - Create a reuse buffer and rewrite eligible loads to read from it.

    Output:
      - `$result` is the created reuse buffer handle.
  }];
  let arguments = (ins TransformValueHandleTypeInterface:$target, TransformHandleTypeInterface:$axis);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = [{
    $target `at` $axis attr-dict `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$target, "Value":$axis), [{
      auto anyType = transform::AnyOpType::get($_builder.getContext());
      build($_builder, $_state, anyType, target, axis);
    }]>
  ];
}

def Allo_TagPipelineOp : Op<Transform_Dialect, "allo.pipeline",
  [NavigationTransformOpTrait, MemoryEffectsOpInterface,
   TransformOpInterface, TransformEachOpTrait,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Pipeline the given loop nest";
  let description = [{
    Input:
      - `$loops` must resolve to loops only.
      - `$ii` must be a positive integer.

    Behavior:
      - Tag the selected loops with the given initiation interval (II) for pipelining.

    Output:
      - No result handle; this op mutates payload IR in place.
  }];
  let arguments = (ins TransformHandleTypeInterface:$loop, I64Attr:$ii);
  let results = (outs);
  let assemblyFormat = [{
    $loop `with` $ii attr-dict `:` type($loop)
  }];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def Allo_TagUnrollOp : Op<Transform_Dialect, "allo.unroll",
  [NavigationTransformOpTrait, MemoryEffectsOpInterface,
   TransformOpInterface, TransformEachOpTrait,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Unroll the given loops by a constant factor";
  let description = [{
    Input:
      - `$loops` must resolve to loops only.
      - `$factor` must be a positive integer.
      - The op does not check whether loops are actually unrollable with the given factor.

    Behavior:
      - Tag the selected loops with the given factor for unrolling.

    Output:
      - No result handle; this op mutates payload IR in place.
  }];
  let arguments = (ins TransformHandleTypeInterface:$loops, I64Attr:$factor);
  let results = (outs);
  let assemblyFormat = [{
    $loops `with` $factor attr-dict `:` type($loops)
  }];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def Allo_RenameOp : Op<Transform_Dialect, "allo.rename",
  [NavigationTransformOpTrait, MemoryEffectsOpInterface,
   TransformOpInterface, TransformEachOpTrait,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Rename the given operations";
  let description = [{
    Input:
      - `$target` must resolve to operations only.
      - `$name` is the new name for the target operations.

    Behavior:
      - Rename all target operations to the given name.

    Output:
      - No result handle; this op mutates payload IR in place.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target, StrAttr:$name);
  let results = (outs);
  let assemblyFormat = [{
    $target `to` $name attr-dict `:` type($target)
  }];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def Allo_MatchValueOp : Op<Transform_Dialect, "allo.match_value",
  [NavigationTransformOpTrait, MemoryEffectsOpInterface,
   TransformOpInterface, TransformEachOpTrait,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Match target values";
  let description = [{
    Match target values by the name of their defining operation.
    If the value is introduced as a block argument, match by the name of the parent
    operation and the argument index
    (e.g., `foo:0` for the first argument of an op named `foo`).

    The argument index will be ignored if the value has a defining operation.
  }];
  let arguments = (ins
    TransformHandleTypeInterface:$target,
    I64Attr:$number);
  let results = (outs TransformValueHandleTypeInterface:$result);
  let assemblyFormat = [{
    $number `of` $target attr-dict `:` type($target) `->` type($result)
  }];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
  let builders = [
    OpBuilder<(ins "Value":$target, "int64_t":$number), [{
      auto anyType = transform::AnyValueType::get($_builder.getContext());
      build($_builder, $_state, anyType, target, static_cast<uint64_t>(number));
    }]>
  ];
}

def Allo_PartitionOp : Op<Transform_Dialect, "allo.partition", [
  NavigationTransformOpTrait, MemoryEffectsOpInterface,
  DeclareOpInterfaceMethods<TransformOpInterface>,
  ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Partition the given array";
  let description = [{
    Input:
      - `$target` must resolve to exactly one memref value, or one op with a
        single memref result.

    Behavior:
      - Attach the given partitioning attribute to the target value
  }];
  let arguments = (ins TransformValueHandleTypeInterface:$target, Partition:$partition);
  let results = (outs);
  let assemblyFormat = [{
    $target `with` $partition attr-dict `:` type($target)
  }];
}

#endif // ALLO_TRANSFORM_OPS