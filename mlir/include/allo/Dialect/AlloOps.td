/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_OPS
#define ALLO_OPS

include "allo/Dialect/AlloDialect.td"
include "allo/Dialect/AlloTypes.td"
include "allo/Dialect/AlloAttrs.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Handle definition
//===----------------------------------------------------------------------===//

def Allo_CreateOpHandleOp : Allo_Op<"create_op_handle"> 
{
    let summary = "op handle";
    let description = [{
        Create a op handle

        Parameters
        op_name (String) - The name of a op
    }];

    let arguments = (ins StrAttr:$op_name);
    let results = (outs OpHandle:$result);
    let assemblyFormat = [{
       $op_name attr-dict
    }];
}

def Allo_CreateLoopHandleOp : Allo_Op<"create_loop_handle"> 
{
    let summary = "loop handle";
    let description = [{
        Create a loop handle

        Parameters
        loop_name (String) - The name of a loop
    }];

    let arguments = (ins OpHandle:$op, StrAttr:$loop_name);
    let results = (outs LoopHandle:$result);
    let assemblyFormat = [{
       $op `,` $loop_name attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// Parameterized customization template
//===----------------------------------------------------------------------===//

def CustomizationOp : Allo_Op<"customization", [
    AffineScope, AutomaticAllocationScope, CallableOpInterface,
    FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
  ]> {
  let summary = "user defined customization";
  let description = [{
    The "allo.customization" operation represents a user defined customization.
    These are callable SSA-region operations that contain optimization primitives.

    Example:

    ```mlir
    allo.customization @opt() {
        // optimizations
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return nullptr; /// getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return nullptr; /// getResAttrs().value_or(nullptr);
    }

    ::mlir::ArrayAttr getArgAttrsAttr() {
      return nullptr;
    }

    ::mlir::ArrayAttr getResAttrsAttr() {
      return nullptr;
    }

    void setArgAttrsAttr(::mlir::ArrayAttr attrs) {
    }

    void setResAttrsAttr(::mlir::ArrayAttr attrs) {
    }

    ::mlir::Attribute removeArgAttrsAttr() {
        return nullptr;
    }

    ::mlir::Attribute removeResAttrsAttr() {
        return nullptr;
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "allo"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  bit hasCustomAssemblyFormat = 1;
  bit hasVerifier = 1;
}

def EndOp : Allo_Op<"end", [NoMemoryEffect, HasParent<"CustomizationOp">, Terminator]> {
  let summary = "end operation";
  let description = [{
    The "end" operation represents a return operation within a customization.
  }];

  // The end operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<AnyType>:$input);

  // The end operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
}

def ApplyOp : Allo_Op<"apply", [CallOpInterface,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "apply customizations";
  let description = [{
    Apply predefined customizations to a function.
  }];

  // The apply operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);

  // The apply operation has no results, just a placeholder below.
  let results = (outs Variadic<AnyType>);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];

  // Add custom build methods for the apply operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$operands)>
  ];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }
    
    /// ArgAndResultAttrsOpInterface
    ::mlir::ArrayAttr getArgAttrsAttr() {
      return nullptr;
    }

    ::mlir::ArrayAttr getResAttrsAttr() {
      return nullptr;
    }
    
    void setArgAttrsAttr(::mlir::ArrayAttr attrs) {
    }

    void setResAttrsAttr(::mlir::ArrayAttr attrs) {
    }

    ::mlir::Attribute removeArgAttrsAttr() {
        return nullptr;
    }

    ::mlir::Attribute removeResAttrsAttr() {
        return nullptr;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Customization primitives
//===----------------------------------------------------------------------===//

def Allo_ReorderOp : Allo_Op<"reorder">
{
    let summary = "reorder";
    let description = [{
        allo.reorder(*args)

        Reorder the arguments in the specified order.

        Parameters
        args (list of IterVar) - The order to be ordered
    }];

    let arguments = (ins Variadic<LoopHandle>:$loops);
    let results = (outs );
    let assemblyFormat = [{
        `(` ($loops^)? `)` attr-dict
    }];
}

def Allo_SplitOp : Allo_Op<"split"> 
{
    let summary = "split";
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

    let arguments = (ins LoopHandle:$loop, UI32Attr:$factor);
    let results = (outs LoopHandle:$outer, LoopHandle:$inner);
    let assemblyFormat = [{
        `(` $loop `,` $factor `)` attr-dict
    }];
    // https://mlir.llvm.org/docs/OpDefinitions/#custom-builder-methods
    // let builders = [];
}

def Allo_TileOp : Allo_Op<"tile"> 
{
    let summary = "tile";
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

    let arguments = (ins LoopHandle:$x_loop, LoopHandle:$y_loop, UI32Attr:$x_factor, UI32Attr:$y_factor);
    let results = (outs LoopHandle:$x_outer, LoopHandle:$x_inner, LoopHandle:$y_outer, LoopHandle:$y_inner);
    let assemblyFormat = [{
        `(` $x_loop `,` $y_loop `,` $x_factor `,` $y_factor `)` attr-dict
    }];
}

def Allo_FuseOp : Allo_Op<"fuse"> 
{
    let summary = "fuse";
    let description = [{
        allo.fuse(*args)

        Fuse multiple consecutive iteration variables into a single iteration variable.

        fused = fuse(...fuse(fuse(args[0], args[1]), args[2]),..., args[-1]) The order is from outer to inner.

        Parameters
        args (list of IterVars) - Itervars that proceeds each other

        Returns
        fused - The fused variable of iteration.
    }];

    let arguments = (ins Variadic<LoopHandle>:$loops);
    let results = (outs LoopHandle:$fused_loop);
    let assemblyFormat = [{
        `(` ($loops^)? `)` attr-dict
    }];
}

def Allo_ComputeAtOp : Allo_Op<"compute_at"> 
{
    let summary = "compute_at";
    let description = [{
        allo.compute_at(parent, scope)

        Attach the stage at parent's scope

        Parameters
        * parent (_Stage) - The parent stage
        * scope (IterVar) - The loop scope t be attached to.
    }];

    let arguments = (ins OpHandle:$producer, OpHandle:$consumer, LoopHandle:$axis);
    let results = (outs );
    let assemblyFormat = [{
        `(` $producer `,` $consumer `,` $axis `)` attr-dict
    }];
}

def Allo_UnrollOp : Allo_Op<"unroll"> 
{
    let summary = "unroll";
    let description = [{
        allo.unroll(var, factor=0)

        Unroll the iteration.

        Parameters
        * var (IterVar) - The iteration to be unrolled.
        * factor (Expr) - The unroll factor. Default value 0 means full unroll.
    }];

    let arguments = (ins LoopHandle:$loop, OptionalAttr<DefaultValuedAttr<UI32Attr,"0">>:$factor);
    let results = (outs );
    let assemblyFormat = [{
        `(` $loop (`,` $factor^)? `)` attr-dict
    }];
}

def Allo_UnfoldOp : Allo_Op<"unfold"> 
{
    let summary = "unfold";
    let description = [{
        allo.unfold(var, factor=0)

        Unfold the iteration into PE array.

        Parameters
        * var (IterVar) - The iteration to be unrolled.
        * factor (Expr) - The unfold factor. Default value  means skip unfolding.
    }];

    let arguments = (ins LoopHandle:$loop, OptionalAttr<DefaultValuedAttr<UI32Attr,"1">>:$factor);
    let results = (outs LoopHandle:$result);
    let assemblyFormat = [{
        `(` $loop (`,` $factor^)? `)` attr-dict
    }];
}

def Allo_PipelineOp : Allo_Op<"pipeline"> 
{
    let summary = "pipeline";
    let description = [{
        allo.pipeline(var, initiation_interval=1)

        Pipeline the iteration.

        Parameters
        * var (IterVar) - The iteration to be pipelined.
        * initiation_interval (Expr) - The initiation interval in pipeline schedule. Default value is 1.
    }];

    let arguments = (ins LoopHandle:$loop, OptionalAttr<DefaultValuedAttr<UI32Attr, "1">>:$ii);
    let results = (outs LoopHandle:$result);
    let assemblyFormat = [{
        `(` $loop (`,` $ii^)? `)` attr-dict
    }];
    let builders = [
        OpBuilder<(ins "allo::LoopHandleType":$loop,
                   CArg<"uint64_t", "1">:$ii)>
    ];
}

def Allo_ParallelOp : Allo_Op<"parallel"> 
{
    let summary = "parallel";
    let description = [{
        allo.parallel(var)

        Parallelize the iteration.

        Parameters
        var (IterVar) - The iteration to be parallelized.
    }];

    let arguments = (ins LoopHandle:$loop);
    let results = (outs );
    let assemblyFormat = [{
        `(` $loop `)` attr-dict
    }];
}

def Allo_ThreadBindOp : Allo_Op<"bind">
{
    let summary = "bind";
    let description = [{
        allo.bind(var, dim=1)

        Bind the iteration to a specific NDRange dimension.

        Parameters
        * var (IterVar) - The iteration to be mapped.
        * dim (Expr) - The NDRange dimension to be binded to the iteration.
    }];

    let arguments = (ins LoopHandle:$loop, NDRangeDimKindEnum:$dim);
    let results = (outs );
    let assemblyFormat = [{
        `(` $loop `,` $dim `)` attr-dict
    }];
    let builders = [
        OpBuilder<(ins "allo::LoopHandleType":$loop,
                   CArg<"uint64_t", "1">:$dim)>
    ];
}

def Allo_PartitionOp : Allo_Op<"partition"> 
{
    let summary = "partition";
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

    let arguments = (ins AnyMemRef:$target,
                     DefaultValuedAttr<PartitionKindEnum, "PartitionKindEnum::CompletePartition">:$partition_kind,
                     DefaultValuedAttr<UI32Attr, "0">:$dim,
                     OptionalAttr<DefaultValuedAttr<UI32Attr, "0">>:$factor);
    let results = (outs );
    let assemblyFormat = [{
        `(` $target `:` type($target) `,` $partition_kind `,` $dim (`,` $factor^)? `)` attr-dict
    }];
}

def Allo_ReuseAtOp : Allo_Op<"reuse_at"> 
{
    let summary = "reuse_at";
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

    let arguments = (ins AnyMemRef:$target, LoopHandle:$axis);
    let results = (outs AnyMemRef:$result);
    let assemblyFormat = [{
        `(` $target `:` type($target) `,` $axis `)` attr-dict `->` type($result)
    }];
}

def Allo_BufferAtOp : Allo_Op<"buffer_at"> 
{
    let summary = "buffer_at";
    let description = [{
        allo.buffer_at(target, parent, axis)

        Create a write buffer at the axis

        This returns a new tensor representing the write buffer. A stage is also built correspondingly. The new stage will be a sub-stage of the parent stage under the specified axis. Thus, the axis must be inside the axis list of the parent stage.

        Parameters
        * target (Tensor) - The tensor whose values will be buffered
        * parent (Stage) - The stage that includes the target tensor
        * axis (IterVar) - The axis that generates the buffer
    }];

    let arguments = (ins AnyMemRef:$target, LoopHandle:$axis);
    let results = (outs AnyMemRef:$result);
    let assemblyFormat = [{
        `(` $target `:` type($target) `,` $axis `)` attr-dict `->` type($result)
    }];
}

def Allo_ReshapeOp : Allo_Op<"reshape"> 
{
    let summary = "reshape";
    let description = [{
        allo.reshape(target)

        Reshape the tensor by modifying the memref
    }];

    let arguments = (ins AnyMemRef:$target);
    let results = (outs AnyMemRef:$output);
    let assemblyFormat = [{
        $target `:` type($target) `to` type($output) attr-dict
    }];
}

def Allo_InterKernelToOp : Allo_Op<"inter_kernel_to"> 
{
    let summary = "inter-kernel data placement";
    let description = [{
        allo.to(tensor, dst=None, src=None, axis=0,
               mode=IO.DMA, fifo_depth=1, burst_len=-1)

        Stream a list of Tensors to dst devices

        Parameters
        ----------
        tensor (Tensor) - The tensor to be moved
        dst (Device or Stage) - The destination of data movement
        src (Device or Stage) - The source of data movement
        axis (str or IterVar) - Move axis-th loop body to xcel scope
        mode (str) - The modes of data movement (Stream, DMA, MMIO)
            For inter-kernel data movemnet, only Stream is supported
        fifo_depth (int) - The streaming channel depth
            We leave an interface here to specify the FIFO depth
            in the future we should be able to infer automatically
    }];

    let arguments = (ins AnyMemRef:$target, OpHandle:$dst,
                         OptionalAttr<DefaultValuedAttr<I32Attr,"-1">>:$fifo_depth);
    let results = (outs );
    let assemblyFormat = [{
        `(` $target `:` type($target) `,` $dst (`,` $fifo_depth^)? `)` attr-dict
    }];
}

def Allo_HostXcelToOp : Allo_Op<"host_xcel_to">
{
    let summary = "host-xcel data placement";
    let description = [{
        allo.to(tensor, axis, device)
    }];
    let arguments = (ins AnyMemRef:$target,
                        DefaultValuedAttr<DeviceEnum, "DeviceEnum::CPUDevice">:$device,
                        Optional<LoopHandle>:$axis);
    let results = (outs );
    let assemblyFormat = [{
        `(` $target `:` type($target) `,` $device (`,` $axis^)? `)` attr-dict
    }];
}

def Allo_IntraKernelToOp : Allo_Op<"to">
{
    let summary = "intra kernel data placement";
    let description = [{
        allo.to(tensor, PEs)
        ---
        IntraKernelToOp takes a target memory as input, i.e., $target, and 
        move the data to a list of PEs, i.e., $pe_array 
    }];
    let arguments = (ins AnyMemRef:$target, LoopHandle:$pe_array);
    let results = (outs AnyMemRef:$result);
    let assemblyFormat = [{
        `(` $target `:` type($target) `,` $pe_array `)` attr-dict `->` type($result)
    }];
}

def Allo_OutlineOp : Allo_Op<"outline"> 
{
    let summary = "outline";
    let description = [{
        allo.outline(stage)

        Outline a stage as a function
    }];

    let arguments = (ins Variadic<OpHandle>:$stages);
    let results = (outs );
    let assemblyFormat = [{
        `(` ($stages^)? `)` attr-dict
    }];
}

def Allo_ReformOp : Allo_Op<"reform"> 
{
    let summary = "reform";
    let description = [{
        allo.reform(target)

        Change the layout of the target tensor
        Need to pass in AffineMapAttr as layout
    }];

    let arguments = (ins AnyMemRef:$target);
    let results = (outs AnyMemRef:$result);
    let assemblyFormat = [{
        `(` $target `:` type($target) `)` attr-dict `->` type($result)
    }];
}

def Allo_CloneOp : Allo_Op<"clone">
{
    let summary = "clone";
    let description = [{
        allo.clone(target, op_handle_0, op_handle_1, ...)
        Clone a function and replace the call operation's callee with the new functions
    }];
    let arguments = (ins FlatSymbolRefAttr:$target, Variadic<OpHandle>:$op_handles);
    let results = (outs );
    let assemblyFormat = [{
        `(` $target `,` $op_handles `)` attr-dict
    }];
}

def Allo_ReplaceOp : Allo_Op<"replace"> 
{
    let summary = "replace";
    let description = [{
        allo.replace(src, dst)
    }];

    let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
    let results = (outs );
    let assemblyFormat = [{
        `(` $src `:` type($src) `,` $dst `:` type($dst) `)` attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// Fixed-point operations
//===----------------------------------------------------------------------===//

def AnyFixed : Type<CPred<"::llvm::isa<::mlir::allo::FixedType>($_self) || ::llvm::isa<::mlir::allo::UFixedType>($_self)">, "fixed-point",
                    "allo::AnyFixedType">;
def FixedLike: TypeConstraint<Or<[AnyFixed.predicate,
        VectorOfAnyRankOf<[AnyFixed]>.predicate, TensorOf<[AnyFixed]>.predicate]>,
    "fixed-point-like">;

class FixedBinaryOp<string mnemonic, list<Trait> traits = []> :
    Op<Allo_Dialect, mnemonic, traits # [SameOperandsAndResultType]>,
    Arguments<(ins FixedLike:$lhs, FixedLike:$rhs)>,
    Results<(outs FixedLike:$result)>;

def AddFixedOp : FixedBinaryOp<"add_fixed"> {
  let summary = "fixed point addition operation";
}

def SubFixedOp : FixedBinaryOp<"sub_fixed"> {
  let summary = "fixed point subtraction operation";
}

def MulFixedOp : FixedBinaryOp<"mul_fixed"> {
  let summary = "fixed point mulplication operation";
}

def DivFixedOp : FixedBinaryOp<"div_fixed"> {
  let summary = "fixed point division operation";
}

def ShLFixedOp : Allo_Op<"shl_fixed"> {
  let summary = "fixed point shift left operation";
  let arguments = (ins
      FixedLike:$lhs,
      SignlessIntegerLike:$rhs
  );
  let results = (outs FixedLike:$result);
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` `:` attr-dict `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

def ShRFixedOp : Allo_Op<"shr_fixed"> {
  let summary = "fixed point shift right operation";
  let arguments = (ins
      FixedLike:$lhs,
      SignlessIntegerLike:$rhs
  );
  let results = (outs FixedLike:$result);
  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` `:` attr-dict `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

def CmpFixedOp : Allo_Op<"cmp_fixed", [NoMemoryEffect, SameTypeOperands, TypesMatchWith<
    "result type has i1 element type and same shape as operands",
    "lhs", "result", "getI1SameShape($_self)">] # ElementwiseMappable.traits> {
  let summary = "fixed-point comparison operation";

  let arguments = (ins
      CmpFixedPredicateAttr:$predicate,
      FixedLike:$lhs,
      FixedLike:$rhs
  );
  let results = (outs BoolLike:$result);

//   let builders = [
//     OpBuilder<(ins "CmpFixedPredicate":$predicate, "Value":$lhs,
//                  "Value":$rhs), [{
//       mlir::allo::buildCmpFixedOp($_builder, $_state, predicate, lhs, rhs);
//     }]>];

  let extraClassDeclaration = [{
    static CmpFixedPredicate getPredicateByName(StringRef name);
  }];

  bit hasVerifier = 0;

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def MinFixedOp : FixedBinaryOp<"min_fixed"> {
  let summary = "fixed point minimum operation";
}

def MaxFixedOp : FixedBinaryOp<"max_fixed"> {
  let summary = "fixed point maximum operation";
}

def GetGlobalFixedOp : Allo_Op<"get_global_fixed", [NoMemoryEffect]> {
    let summary = "generates a fixed point memref from a int global memref";
    let arguments = (ins FlatSymbolRefAttr:$name);
    let results = (outs AnyStaticShapeMemRef:$result);
    let assemblyFormat = [{
        $name `:` type($result) attr-dict
    }];
}

def FixedToFloatOp : Allo_Op<"fixed_to_float"> {
  let summary = "fixed to float cast operation";
  let arguments = (ins FixedLike:$input);
  let results = (outs FloatLike:$res);
  let assemblyFormat = [{
      `(` $input `)` attr-dict `:` type($input) `->` type($res)
  }];
}

def FloatToFixedOp : Allo_Op<"float_to_fixed"> {
  let summary = "float to fixed cast operation";
  let arguments = (ins FloatLike:$input);
  let results = (outs FixedLike:$res);
  let assemblyFormat = [{
      `(` $input `)` attr-dict `:` type($input) `->` type($res)
  }];
}

def IntToFixedOp : Allo_Op<"int_to_fixed"> {
  let summary = "int to fixed cast operation";
  let arguments = (ins SignlessIntegerLike:$input);
  let results = (outs FixedLike:$res);
  let assemblyFormat = [{
      `(` $input `)` attr-dict `:` type($input) `->` type($res)
  }];
}

def FixedToIntOp : Allo_Op<"fixed_to_int"> {
  let summary = "fixed to int cast operation";
  let arguments = (ins FixedLike:$input);
  let results = (outs SignlessIntegerLike:$res);
  let assemblyFormat = [{
      `(` $input `)` attr-dict `:` type($input) `->` type($res)
  }];
}

def FixedToFixedOp : Allo_Op<"fixed_to_fixed"> {
  let summary = "Cast operation from one fixed point type to another";
  let arguments = (ins FixedLike:$input);
  let results = (outs FixedLike:$res);
  let assemblyFormat = [{
      `(` $input `)` attr-dict `:` type($input) `->` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Bitwise operations
//===----------------------------------------------------------------------===//

def GetIntBitOp : Allo_Op<"get_bit"> {
    let summary = "get bit of an integer";
    let arguments = (ins SignlessIntegerLike:$num, Builtin_Index:$index);
    let results = (outs BoolLike:$result);
    let assemblyFormat = [{
        `(` $num `:` type($num) `,` $index `)` attr-dict `->` type($result)
    }];
}

def SetIntBitOp : Allo_Op<"set_bit"> {
    let summary = "set bit of an integer";
    let arguments = (ins SignlessIntegerLike:$num, Builtin_Index:$index, BoolLike:$val);
    let results = (outs SignlessIntegerLike:$result);
    let assemblyFormat = [{
        `(` $num `:` type($num) `,` $index `,` $val `:` type($val) `)` attr-dict `->` type($result)
    }];
}

def GetIntSliceOp : Allo_Op<"get_slice"> {
    let summary = "get slice of an integer";
    let arguments = (ins SignlessIntegerLike:$num, Builtin_Index:$hi, Builtin_Index:$lo);
    let results = (outs SignlessIntegerLike:$result);
    let assemblyFormat = [{
        `(` $num `:` type($num) `,` $hi `,` $lo `)` attr-dict `->` type($result)
    }];
}

def SetIntSliceOp : Allo_Op<"set_slice"> {
    let summary = "set slice of an integer";
    let arguments = (ins SignlessIntegerLike:$num, Builtin_Index:$hi, Builtin_Index:$lo, SignlessIntegerLike:$val);
    let results = (outs SignlessIntegerLike:$result);
    let assemblyFormat = [{
        `(` $num `:` type($num) `,` $hi `,` $lo `,` $val `:` type($val) `)` attr-dict `->` type($result)
    }];
}

def BitReverseOp : Allo_Op<"bit_reverse", [SameOperandsAndResultType]> {
    let summary = "reverse bits of an integer";
    let arguments = (ins SignlessIntegerLike:$num);
    let results = (outs SignlessIntegerLike:$result);
    let assemblyFormat = [{
        `(` $num `:` type($num) `)` attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// Logic operations
//===----------------------------------------------------------------------===//

def LogicalAndOp : Allo_Op<"logical_and"> {
    let summary = "logical and operation";
    let arguments = (ins Variadic<AnyType>:$input);
    let results = (outs BoolLike:$result);
    let assemblyFormat = [{
        ($input^)? attr-dict `:` type($input) `->` type($result)
    }];
}

def LogicalOrOp : Allo_Op<"logical_or"> {
    let summary = "logical or operation";
    let arguments = (ins Variadic<AnyType>:$input);
    let results = (outs BoolLike:$result);
    let assemblyFormat = [{
        ($input^)? attr-dict `:` type($input) `->` type($result)
    }];
}

def AndOp : Allo_Op<"and"> {
    let summary = "Logical and operation with variadic region, supporting short-circuiting";
    let results = (outs BoolLike:$result);
    let regions = (region VariadicRegion<AnyRegion>:$bodies);
    let assemblyFormat = [{
        ($bodies^)?  attr-dict `:` type($result)
    }];
}

def OrOp : Allo_Op<"or"> {
    let summary = "Logical or operation with variadic region, supporting short-circuiting";
    let results = (outs BoolLike:$result);
    let regions = (region VariadicRegion<AnyRegion>:$bodies);
    let assemblyFormat = [{
        ($bodies^)?  attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : Allo_Op<"yield", [NoMemoryEffect, Terminator,
                               ParentOneOf<["AndOp, OrOp"]>]> {
  let summary = "yield and termination operation";
  let description = [{
    "allo.yield" yields an SSA value from the Allo dialect op region and
    terminates the regions.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

//===----------------------------------------------------------------------===//
// Miscellaneous operations
//===----------------------------------------------------------------------===//

def PrintOp : Allo_Op<"print"> {
  let summary = "print value operation";
  let description = [{
    The "print" builtin operation prints a list of 
    values (variadic), and produces no results.
  }];

  // The print operation takes a list of values (variadic).
  let arguments = (ins Variadic<AnyType>:$input);
  let assemblyFormat = "`(` $input `)` attr-dict `:` type($input)";
}

def PrintMemRefOp : Allo_Op<"print_memref"> {
  let summary = "print memref operation";
  let description = [{
    The "print_memref" builtin operation prints a memref
    and produces no results. We define our own print_memref
    to support fixed-point type memrefs
  }];
  let arguments = (ins AnyMemRef:$input);
  let assemblyFormat = "`(` $input `)` attr-dict `:` type($input)";
}

def StructConstructOp : Allo_Op<"struct_construct", [NoMemoryEffect]> {
    let summary = "struct construct";
    let description = [{
        The "struct_construct" builtin operation constructs a struct from
        a list of values.
    }];
    let arguments = (ins Variadic<AnyType>:$input);
    let results = (outs Struct:$output);

    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `->` type($output)
    }];
}

def StructGetOp : Allo_Op<"struct_get", [NoMemoryEffect]> {
    let summary = "struct get";
    let description = [{
        The "struct_get" operation gets a field from a struct.
    }];
    let arguments = (ins Struct:$input, I64Attr:$index);
    let results = (outs AnyType:$output);

    let assemblyFormat = [{
        $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
    }];

    let builders = [
        OpBuilder<(ins "Value":$input, "size_t":$index)>
    ];

}

def IntToStructOp : Allo_Op<"int_to_struct"> {
  let summary = "Cast operation from an integer type to a struct type";
  let arguments = (ins SignlessIntegerLike:$input);
  let results = (outs Struct:$res);
  let assemblyFormat = [{
      `(` $input `)` attr-dict `:` type($input) `->` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Stream operations
//===----------------------------------------------------------------------===//

def StreamConstructOp : Allo_Op<"stream_construct"> {
  let summary = "Create a stream object";
  let arguments = (ins );
  let results = (outs StreamLike:$res);
  let assemblyFormat = [{
      `(` `)` attr-dict `:` qualified(type($res))
  }];
}

def StreamGetOp : Allo_Op<"stream_get"> {
  let summary = "Get an object from a stream";
  let arguments = (ins StreamLike:$stream, DenseI64ArrayAttr:$indices);
  let results = (outs AnyType:$res);
  let assemblyFormat = [{
      `(` $stream `,` $indices`)` attr-dict `:` qualified(type($stream)) `->` type($res)
  }];
}

def StreamPutOp : Allo_Op<"stream_put"> {
  let summary = "Put an object into a stream";
  let arguments = (ins StreamLike:$stream, DenseI64ArrayAttr:$indices, AnyType:$data);
  let results = (outs );
  let assemblyFormat = [{
      `(` $stream `,` $indices `,` $data `)` attr-dict `:` qualified(type($stream)) `contains` type($data)
  }];
}

//===----------------------------------------------------------------------===//
// Layout operations
//===----------------------------------------------------------------------===//

def TransformLayoutOp : Allo_Op<"transform_layout"> {
  let summary = "Transform the layout of a memref as specified by offsets, sizes, and strides";
  let description = [{
    This operation takes a memref and three constant integer arrays representing
    per-dimension `offsets`, `sizes`, and `strides`, and returns a new memref
    with a layout transformation based on those arguments.
    The result memref must have:
        - The same element type as the source
        - A total number of elements equal to the product of `sizes`. 
        - A layout that matches the access pattern defined by `offsets`, `sizes`, and `strides`, and does not access out-of-bounds data in the source

    Example: 
        %1 = allo.transform_layout %0 offsets = [0, 0, 0, 0] sizes = [4, 4, 4, 4] strides = [64, 4, 16, 1] : memref<16x16xi16> -> memref<16x16xi16>

        This creates a 4D logical view of the original 2D memref of shape [16, 16]. The interpretation is:
        - The original 2D data is virtually reshaped into 4 dimensions: [4, 4, 4, 4]
            - `offsets = [0, 0, 0, 0]`: starts at the origin
            - `sizes = [4, 4, 4, 4]`: the result is a 4D (4x4x4x4) logical view
            - `strides = [64, 4, 16, 1]`:
                - moving 1 step in the first dimension jumps 64 elements in original memory
                - second dimension jumps 4
                - third dimension jumps 16
                - fourth dimension is contiguous (stride 1)
        Even though the logical shape is 4D, the result memref type is still `memref<16x16xi16>`, 
        which is valid because the element number matches.
  }];

  let arguments = (ins
    AnyStaticShapeMemRef:$source,
    DenseI64ArrayAttr:$offsets,
    DenseI64ArrayAttr:$sizes,
    DenseI64ArrayAttr:$strides
  );

  let results = (outs AnyStaticShapeMemRef:$result);

  let assemblyFormat = [{
    $source `offsets` `=` $offsets
            `sizes` `=` $sizes
            `strides` `=` $strides
            attr-dict `:` type($source) `->` type($result)
  }];
}

#endif // ALLO_OPS
