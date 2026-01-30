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

def LoopSplitOp : Op<Transform_Dialect, "allo.split", [
                    FunctionalStyleTransformOpTrait,
                    MemoryEffectsOpInterface,
                    TransformOpInterface,
                    TransformEachOpTrait,
                    ReportTrackingListenerFailuresOpTrait
                    ]> {
    let summary = "Split the given loop with a specified factor";
    let description = [{
        allo.split(parent, factor=None, nparts=None, mode='transform')

        Split the stage either by factor providing outer scope, or both

        Parameters
        * parent (IterVar) - The parent iter var.
        * factor (Expr, optional) - The splitting factor
        * nparts (Expr, optional) - The number of outer parts.
        * mode (str, "transform" or "annotate") - “transform” mode changes the IR structure, “annotate” mode adds attributes.

        Returns
        * outer (IterVar) - The outer variable of iteration.
        * inner (IterVar) - The inner variable of iteration.
    }];
    let arguments = (ins TransformHandleTypeInterface:$loop, I32Attr:$factor);
    let results = (outs TransformHandleTypeInterface:$outer, TransformHandleTypeInterface:$inner);
    let assemblyFormat = [{
        $loop `,` $factor attr-dict `:` functional-type(operands, results)
    }];
    let extraClassDeclaration = [{
        ::mlir::DiagnosedSilenceableFailure applyToOne(
            ::mlir::transform::TransformRewriter &rewriter,
            ::mlir::Operation *target,
            ::mlir::transform::ApplyToEachResultList &results,
            ::mlir::transform::TransformState &state);
    }];
}

def LoopTileOp : Op<Transform_Dialect, "allo.tile", [
                    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                    DeclareOpInterfaceMethods<TransformOpInterface>,
                    ReportTrackingListenerFailuresOpTrait,
                    ]> {
    let summary = "Tile the given loop with specified tile sizes";
    let description = [{
        allo.tile(x_parent, y_parent, x_factor, y_factor)

        Perform tiling on two dimensions

        The final loop order from outmost to inner most are [x_outer, y_outer, x_inner, y_inner]

        Parameters
        x_parent (IterVar) - The original x dimension
        y_parent (IterVar) - The original y dimension
        x_factor (Expr) - The stride factor on x axis
        y_factor (Expr) - The stride factor on y axis

        Returns
        x_outer (IterVar) - Outer axis of x dimension
        y_outer (IterVar) - Outer axis of y dimension
        x_inner (IterVar) - Inner axis of x dimension
        p_y_inner (IterVar) - Inner axis of y dimension
    }];
    let arguments = (ins Variadic<TransformHandleTypeInterface>:$loops, DenseI32ArrayAttr:$tile_sizes);
    let results = (outs Variadic<TransformHandleTypeInterface>:$tiled_loops);
    let assemblyFormat = [{
        $loops `,` $tile_sizes attr-dict `:` functional-type(operands, results)
    }];
    let hasVerifier = 1;
}

def LoopReorderOp : Op<Transform_Dialect, "allo.reorder", [
                       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                       DeclareOpInterfaceMethods<TransformOpInterface>,
                       ReportTrackingListenerFailuresOpTrait
                       ]> {
    let summary = "Reorder the given loops with a specified permutation map";
    let description = [{
        allo.reorder(*args)

        Reorder the arguments in the specified order.

        Parameters
        args (list of IterVar) - The order to be ordered
    }];
    let arguments = (ins Variadic<TransformHandleTypeInterface>:$loops, DenseI32ArrayAttr:$permutation);
    let results = (outs Variadic<TransformHandleTypeInterface>:$reordered_loops);
    let assemblyFormat = [{
        $loops `,` $permutation attr-dict `:` functional-type(operands, results)
    }];
    let hasVerifier = 1;
}

def LoopUnrollOp : Op<Transform_Dialect, "allo.unroll", [
                         FunctionalStyleTransformOpTrait,
                         MemoryEffectsOpInterface,
                         TransformOpInterface,
                         TransformEachOpTrait,
                         ReportTrackingListenerFailuresOpTrait
                        ]> {
    let summary = "Tag the given unroll factor to the loop";
    let description = [{
        allo.unroll(var, factor=0)

        Unroll the iteration.

        Parameters
        * var (IterVar) - The iteration to be unrolled.
        * factor (Expr) - The unroll factor. Default value 0 means full unroll.
    }];
    let arguments = (ins TransformHandleTypeInterface:$loop, OptionalAttr<I32Attr>:$factor);
    let results = (outs TransformHandleTypeInterface:$tagged_loop);
    let assemblyFormat = [{
        $loop `,` $factor attr-dict `:` functional-type(operands, results)
    }];
    let extraClassDeclaration = [{
        ::mlir::DiagnosedSilenceableFailure applyToOne(
            ::mlir::transform::TransformRewriter &rewriter,
            ::mlir::Operation *target,
            ::mlir::transform::ApplyToEachResultList &results,
            ::mlir::transform::TransformState &state);
    }];
}

def LoopParallelOp : Op<Transform_Dialect, "allo.parallel", [
                           FunctionalStyleTransformOpTrait,
                           MemoryEffectsOpInterface,
                           TransformOpInterface,
                           TransformEachOpTrait,
                           ReportTrackingListenerFailuresOpTrait
                          ]> {
    let summary = "Tag the given loop as parallel";
    let description = [{
        allo.parallel(var)

        Parallelize the iteration.

        Parameters
        var (IterVar) - The iteration to be parallelized.
    }];
    let arguments = (ins TransformHandleTypeInterface:$loop);
    let results = (outs TransformHandleTypeInterface:$tagged_loop);
    let assemblyFormat = [{
        $loop attr-dict `:` functional-type(operands, results)
    }];
    let extraClassDeclaration = [{
        ::mlir::DiagnosedSilenceableFailure applyToOne(
            ::mlir::transform::TransformRewriter &rewriter,
            ::mlir::Operation *target,
            ::mlir::transform::ApplyToEachResultList &results,
            ::mlir::transform::TransformState &state);
    }];
}


def LoopPipelineOp : Op<Transform_Dialect, "allo.pipeline", [
                            FunctionalStyleTransformOpTrait,
                            MemoryEffectsOpInterface,
                            TransformOpInterface,
                            TransformEachOpTrait,
                            ReportTrackingListenerFailuresOpTrait
                           ]> {
    let summary = "Tag the given loop for pipelining with a specified initiation interval";
    let description = [{
        allo.pipeline(var, initiation_interval=1)

        Pipeline the iteration.

        Parameters
        * var (IterVar) - The iteration to be pipelined.
        * initiation_interval (Expr) - The initiation interval in pipeline schedule. Default value is 1.
    }];
    let arguments = (ins TransformHandleTypeInterface:$loop, OptionalAttr<I32Attr>:$interval);
    let results = (outs TransformHandleTypeInterface:$tagged_loop);
    let assemblyFormat = [{
        $loop `,` $interval attr-dict `:` functional-type(operands, results)
    }];
    let extraClassDeclaration = [{
        ::mlir::DiagnosedSilenceableFailure applyToOne(
            ::mlir::transform::TransformRewriter &rewriter,
            ::mlir::Operation *target,
            ::mlir::transform::ApplyToEachResultList &results,
            ::mlir::transform::TransformState &state);
    }];
}

def LoopFuseOp : Op<Transform_Dialect, "allo.fuse", [
                       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                       DeclareOpInterfaceMethods<TransformOpInterface>,
                       ReportTrackingListenerFailuresOpTrait
                       ]> {
    let summary = "";
    let description = [{
        allo.fuse(*args)

        Fuse multiple consecutive iteration variables into a single iteration variable.

        fused = fuse(...fuse(fuse(args[0], args[1]), args[2]),..., args[-1]) The order is from outer to inner.

        Parameters
        args (list of IterVars) - Itervars that proceeds each other

        Returns
        fused - The fused variable of iteration.
    }];
    let arguments = (ins Variadic<TransformHandleTypeInterface>:$loops);
    let results = (outs TransformHandleTypeInterface:$flattened_loop);
    let assemblyFormat = [{
        $loops attr-dict `:` functional-type(operands, results)
    }];
}

def LoopUnfoldOp : Op<Transform_Dialect, "allo.unfold", [
                         FunctionalStyleTransformOpTrait,
                         MemoryEffectsOpInterface,
                         TransformOpInterface,
                         TransformEachOpTrait,
                         ReportTrackingListenerFailuresOpTrait
                        ]> {
    let summary = "Tag the given unfold factor to the loop";
    let description = [{
        allo.unfold(var, factor=0)

        Unfold the iteration into PE array.

        Parameters
        * var (IterVar) - The iteration to be unrolled.
        * factor (Expr) - The unfold factor. Default value 1 means skip unfolding.
    }];
    let arguments = (ins TransformHandleTypeInterface:$loop, OptionalAttr<I32Attr>:$factor);
    let results = (outs TransformHandleTypeInterface:$tagged_loop);
    let assemblyFormat = [{
        $loop `,` $factor attr-dict `:` functional-type(operands, results)
    }];
    let extraClassDeclaration = [{
        ::mlir::DiagnosedSilenceableFailure applyToOne(
            ::mlir::transform::TransformRewriter &rewriter,
            ::mlir::Operation *target,
            ::mlir::transform::ApplyToEachResultList &results,
            ::mlir::transform::TransformState &state);
    }];
}

def ComputeAtOp : Op<Transform_Dialect, "allo.compute_at", [
                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                     DeclareOpInterfaceMethods<TransformOpInterface>,
                     ReportTrackingListenerFailuresOpTrait
                     ]> {
    let summary = "";
    let description = [{
        allo.compute_at(parent, scope)

        Attach the stage at parent's scope

        Parameters
        * parent (_Stage) - The parent stage
        * scope (IterVar) - The loop scope t be attached to.
    }];
    let arguments = (ins
        TransformHandleTypeInterface:$compute_stage,
        TransformHandleTypeInterface:$target_loop
    );
    let results = (outs TransformHandleTypeInterface:$attached_stage);
    let assemblyFormat = [{
        $compute_stage `at` $target_loop attr-dict `:` functional-type(operands, results)
    }];
}

def PartitionOp : Op<Transform_Dialect, "allo.partition", [
                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                     DeclareOpInterfaceMethods<TransformOpInterface>,
                     ReportTrackingListenerFailuresOpTrait
                     ]> {
    let summary = "Partition the given buffer with specified partitioning scheme";
    let description = [{
        allo.partition(target, partition_type=0, dim=0, factor=0)

        Partition a Tensor into smaller Tensors or even registers

        Users can specify the partition type, which includes Complete, Block, and Cyclic. The default type is Complete, which means we completely partition the specified dimension. If Block is specified, the tensor is partitioned into N blocks with equal size. The number N is specified by the factor. Otherwise, if Cyclic is specified, the elements of the tensor is partition in a cyclic manner. For example, if the factor is three, the 1st element will be assigned to the 1st partitioned tensor; the 2nd element will be assigned to the 2nd one; and so on. Finally, if Complete is specified, the factor will be ignored. If dim is set to 0, it means we partition all dimensions.

        Parameters
        * target (Tensor) - The tensor to be partitioned
        * partition_type ({Complete, Block, Cyclic}, optional) - The partition type
        * dim (int, optional) - The dimension to be partitioned
        * factor (int, optional) - The partition factor
    }];

    let arguments = (ins TransformHandleTypeInterface:$buffer,
                     OptionalAttr<PartitionKindEnum>:$partition_type,
                     OptionalAttr<UI32Attr>:$dim,
                     OptionalAttr<UI32Attr>:$factor);
    let results = (outs TransformHandleTypeInterface:$partitioned);
    let assemblyFormat = [{
        $buffer `,` $partition_type `,` $dim `,` $factor attr-dict `:` functional-type(operands, results)
    }];
}

def ReuseAtOp : Op<Transform_Dialect, "allo.reuse_at", [
                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                     DeclareOpInterfaceMethods<TransformOpInterface>,
                     ReportTrackingListenerFailuresOpTrait
                     ]> {
    let summary = "";
    let description = [{
        allo.reuse_at(target, parent, axis, name=None)

        Create a reuse buffer reusing the output of current stage

        This returns a new tensor representing the reuse buffer. A stage is also built correspondingly. The new stage will be a sub-stage of the parent stage under the specified axis. Thus, the axis must be inside the axis list of the parent stage.

        Parameters
        * target (Tensor) - The tensor whose values will be reused
        * parent (Stage) - The stage that reuses the output of the current stage
        * axis (IterVar) - The axis that generates the reuse values
        * name (string, optional) - The name of the reuse buffer
    }];
    let arguments = (ins TransformHandleTypeInterface:$target, TransformHandleTypeInterface:$axis);
    let results = (outs TransformHandleTypeInterface:$result);
    let assemblyFormat = [{
        $target `at` $axis attr-dict `:` functional-type(operands, results)
    }];
}

def ReshapeOp: Op<Transform_Dialect, "allo.reshape", [
                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                     DeclareOpInterfaceMethods<TransformOpInterface>,
                     ReportTrackingListenerFailuresOpTrait
                     ]> {
    let summary = "Reshape the given tensor to the specified shape";
    let description = [{
        allo.reshape(target)

        Reshape the tensor by modifying the memref
    }];

    let arguments = (ins TransformHandleTypeInterface:$tensor,
                     DenseI64ArrayAttr:$newshape);
    let results = (outs TransformHandleTypeInterface:$reshaped);
    let assemblyFormat = [{
        $tensor `,` $newshape attr-dict `:` functional-type(operands, results)
    }];
}

#endif // ALLO_TRANSFORM_OPS