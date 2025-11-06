# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Reference: taichi/python/taichi/lang/ast/transform.py
# pylint: disable=no-name-in-module, unused-argument, unexpected-keyword-arg, no-value-for-parameter, eval-used

import gc
import ast
import sys
import copy
import itertools
import traceback
import numpy as np
from .._mlir.ir import (
    OpView,
    Module,
    Location,
    InsertionPoint,
    FunctionType,
    MemRefType,
    RankedTensorType,
    ShapedType,
    IntegerType,
    F32Type,
    UnitAttr,
    IntegerAttr,
    StringAttr,
    DictAttr,
    AffineExpr,
    AffineConstantExpr,
    AffineMap,
    AffineMapAttr,
    IntegerSet,
    FlatSymbolRefAttr,
    DenseElementsAttr,
    TypeAttr,
    ArrayAttr,
    Attribute,
    OpResultList,
    StridedLayoutAttr,
)
from .._mlir.dialects import (
    allo as allo_d,
    func as func_d,
    memref as memref_d,
    tensor as tensor_d,
    affine as affine_d,
    scf as scf_d,
    arith as arith_d,
    math as math_d,
    linalg as linalg_d,
)
from .._mlir.exceptions import DTypeError
from .utils import (
    MockArg,
    MockScalar,
    MockConstant,
    MockBuffer,
    get_extra_type_hints,
    get_kwarg,
    get_func_id_from_param_types,
    resolve_generic_types,
)
from .types import (
    AlloType,
    Int,
    UInt,
    Index,
    Float,
    Fixed,
    UFixed,
    Struct,
    float32,
    Stream,
)
from .visitor import ASTVisitor, ASTContext, get_symbolic_expr
from .symbol_resolver import ASTResolver
from ..utils import (
    get_mlir_dtype_from_str,
    c2allo_type,
    freeze_list,
    construct_kernel_name,
    get_mlir_op_result,
)
from ..logging import print_error_message


class ASTBuilder(ASTVisitor):
    def __call__(self, ctx: ASTContext, node: ast.AST, file_name: str = None, **kwargs):
        if not ctx.file_name and file_name:
            ctx.file_name = file_name
        if node is None:
            return None
        method = getattr(self, "build_" + node.__class__.__name__, None)
        if method is None:
            error_msg = f'Unsupported node "{node.__class__.__name__}"'
            raise RuntimeError(error_msg)
        if ctx.file_name and hasattr(node, "lineno") and hasattr(node, "col_offset"):
            with ctx.mlir_ctx, Location.file(
                ctx.file_name, node.lineno, node.col_offset
            ):
                res = method(ctx, node, **kwargs)
                return res
        else:
            with ctx.mlir_ctx, Location.unknown():
                res = method(ctx, node, **kwargs)
                return res


# pylint: disable=too-many-public-methods
class ASTTransformer(ASTBuilder):
    # to resolve global variable naming conflict
    global_var_cnt: dict[str, int] = {}

    @staticmethod
    def build_assign_value(ctx: ASTContext, node: ast.Name, buffer, val):
        target = (
            buffer.op.result
            if isinstance(buffer, MockScalar)
            else get_mlir_op_result(buffer)
        )
        # FIXME (Shihan): We may need to look for some workarounds to support such cases.
        assert (
            not ctx.enable_tensor
        ), "MLIR tensors are immutable data structures, fail to resolve."
        if len(node.shape) == 0:
            # scalar
            affine_map = AffineMap.get(dim_count=0, symbol_count=0, exprs=[])
            affine_attr = AffineMapAttr.get(affine_map)
            store_op = affine_d.AffineStoreOp(
                val, target, [], affine_attr, ip=ctx.get_ip()
            )
        else:
            # tensor
            store_op = memref_d.CopyOp(val, target, ip=ctx.get_ip())
        store_op.attributes["to"] = StringAttr.get(node.id)
        return store_op

    @staticmethod
    def build_Name(ctx: ASTContext, node: ast.Name, val=None):
        if val is not None and isinstance(node.ctx, ast.Store):
            buffer = ctx.get_symbol(node.id)
            # FIXME (Shihan): We may need to look for some workarounds to support such cases.
            assert (
                not ctx.enable_tensor
            ), "MLIR tensors are immutable data structures, fail to resolve"
            return ASTTransformer.build_assign_value(
                ctx, node, buffer, get_mlir_op_result(val)
            )
        ret = ctx.get_symbol(node.id, allow_missing=True)
        if ret is not None:
            ret_result = get_mlir_op_result(ret)
            if (
                isinstance(ret_result.type, (MemRefType, RankedTensorType))
                and len(ret_result.type.shape) == 0
            ):
                op_ = ASTTransformer.build_scalar(ctx, ret)
                if not ctx.enable_tensor:
                    op_.attributes["from"] = StringAttr.get(node.id)
                return op_
            return ret
        if node.id in ctx.global_vars:
            return MockConstant(ctx.global_vars[node.id], ctx)
        raise RuntimeError("Unsupported Name")

    @staticmethod
    def build_Constant(ctx: ASTContext, node: ast.Constant):
        return MockConstant(node.value, ctx)

    @staticmethod
    def build_shaped_type(
        ctx: ASTContext, dtype: AlloType, shape: list[int], layout: Attribute = None
    ):
        if len(shape) == 0:
            return dtype.build()
        if not ctx.enable_tensor:
            shape = [
                ShapedType.get_dynamic_size() if s == Ellipsis else s for s in shape
            ]
            return MemRefType.get(shape, dtype.build(), layout)
        return RankedTensorType.get(shape, dtype.build())

    @staticmethod
    def build_array(ctx: ASTContext, dtype: AlloType, shape: list[int]):
        if not ctx.enable_tensor:
            memref_type = MemRefType.get(shape, dtype.build())
            alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ctx.get_ip())
            if isinstance(dtype, UInt):
                alloc_op.attributes["unsigned"] = UnitAttr.get()
            return alloc_op
        return tensor_d.EmptyOp(shape, dtype.build(), ip=ctx.get_ip())

    @staticmethod
    def build_scalar(ctx: ASTContext, res):
        if ctx.enable_tensor:
            res = tensor_d.ExtractOp(tensor=res, indices=[], ip=ctx.get_ip())
        else:
            res_result = get_mlir_op_result(res)
            affine_map = AffineMap.get_identity(0)
            affine_attr = AffineMapAttr.get(affine_map)
            res = affine_d.AffineLoadOp(
                res_result.type.element_type, res, [], affine_attr, ip=ctx.get_ip()
            )
        return res

    @staticmethod
    def attach_op_name(
        ctx: ASTContext, node: ast.AST, op: OpView, name: str, postfix: str = ""
    ):
        if hasattr(node, "keywords") and len(node.keywords) > 0:
            op.attributes["op_name"] = StringAttr.get(
                f"{node.keywords[0].value.value}{postfix}"
            )
        else:
            op.attributes["op_name"] = StringAttr.get(
                f"{name}_{ctx.unnamed_linalg_op_count}"
            )
            ctx.unnamed_linalg_op_count += 1

    @staticmethod
    def build_Attribute(ctx: ASTContext, node: ast.Attribute):
        value = build_stmt(ctx, node.value)

        if node.attr == "T":  # transpose
            shape = node.shape
            alloc_op = ASTTransformer.build_array(ctx, node.dtype, shape)
            with ctx.get_ip():
                transpose_op = linalg_d.transpose(
                    input=get_mlir_op_result(value),
                    outs=[alloc_op.result],
                    permutation=list(range(len(shape)))[::-1],
                )
            ASTTransformer.attach_op_name(ctx, node, transpose_op, "transpose")
            return transpose_op if ctx.enable_tensor else alloc_op

        if node.attr == "reverse":
            return allo_d.BitReverseOp(get_mlir_op_result(value), ip=ctx.get_ip())

        if node.attr == "copy":
            return ASTTransformer.build_library_op(
                ctx, node=node, attr="copy", new_args=[value]
            )

        if node.attr == "bits":
            return MockConstant(value.val.bits, ctx, dtype=Index())
        if node.attr == "fracs":
            return MockConstant(value.val.fracs, ctx, dtype=Index())
        raise RuntimeError("Unsupported Attribute")

    @staticmethod
    def build_Index(ctx: ASTContext, node: ast.Index):
        return build_stmt(ctx, node.value)

    @staticmethod
    def build_Tuple(ctx: ASTContext, node: ast.Tuple):
        return build_stmts(ctx, node.elts)

    @staticmethod
    def build_single_for(ctx: ASTContext, args: list[ast.AST], stage: str, name: str):
        if len(args) == 1:
            # e.g., for i in range(10) // for i, j in grid(10, 10)
            lb_expr, lb_map_attr = ASTTransformer.build_affine_map_attr(
                ctx, ast.Constant(value=0)
            )
            ub_expr, ub_map_attr = ASTTransformer.build_affine_map_attr(ctx, args[0])
            step = 1
        elif len(args) < 4:
            # e.g., for i in range(1, 10)
            #       for i in range(1, 10, 2)
            lb_expr, lb_map_attr = ASTTransformer.build_affine_map_attr(ctx, args[0])
            ub_expr, ub_map_attr = ASTTransformer.build_affine_map_attr(ctx, args[1])
            if len(args) == 3:
                step = ASTResolver.resolve_constant(args[2], ctx)
            else:
                step = 1
        else:
            raise RuntimeError("Unsupported range")

        if (
            lb_map_attr is not None
            and ub_map_attr is not None
            and isinstance(step, int)
        ):  # build AffineForOp
            for_op = affine_d.AffineForOp(
                lower_bound=lb_map_attr,
                upper_bound=ub_map_attr,
                step=step,
                iter_args=[],
                lower_bound_operands=lb_expr,
                upper_bound_operands=ub_expr,
                ip=ctx.get_ip(),
            )
            for_op.attributes["loop_name"] = StringAttr.get(name)
            if stage != "":
                for_op.attributes["op_name"] = StringAttr.get(stage)
            affine_d.AffineYieldOp([], ip=InsertionPoint(for_op.body))
        else:  # build SCFForOp
            lb_expr = build_stmt(ctx, args[0] if len(args) > 1 else ast.Constant(0))
            ub_expr = build_stmt(ctx, args[1] if len(args) >= 2 else args[0])
            # https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop
            # The step is a value of same type but required to be positive.
            if step is not None and step <= 0:
                raise RuntimeError(
                    f"Step in for loop range should be positive, got: {step}"
                )
            step = build_stmt(ctx, args[2] if len(args) >= 3 else ast.Constant(1))
            lb_expr = ASTTransformer.build_cast_op(
                ctx,
                lb_expr,
                args[0].dtype if len(args) >= 1 else Int(32),
                Index(),
            )
            ub_expr = ASTTransformer.build_cast_op(
                ctx,
                ub_expr,
                args[1].dtype if len(args) >= 2 else args[0].dtype,
                Index(),
            )
            step = ASTTransformer.build_cast_op(
                ctx,
                step,
                args[2].dtype if len(args) >= 3 else Int(32),
                Index(),
            )
            for_op = scf_d.ForOp(
                lb_expr.result,
                ub_expr.result,
                step.result,
                ip=ctx.get_ip(),
            )
            for_op.attributes["loop_name"] = StringAttr.get(name)
            if stage:
                for_op.attributes["op_name"] = StringAttr.get(stage)
            scf_d.YieldOp([], ip=InsertionPoint(for_op.body))
        return for_op

    @staticmethod
    def build_all_for(ctx: ASTContext, node: ast.For, attr: str):
        # get loop names
        if isinstance(node.target, ast.Tuple):
            names = [x.id for x in node.target.elts]
        else:
            names = [node.target.id]
        # avoid name conflicts
        names += [str(ctx.loop_band_count)]

        # get stage name
        if len(node.iter.keywords) == 0:
            stage_name = None
        else:
            stage_name = get_kwarg(node.iter.keywords, "name").value

        # build for loops
        is_affine = True
        iter_args = node.iter.args
        if attr in {"grid", "reduction"}:
            for_loops = []
            if stage_name is None:
                stage_name = "S_" + "_".join(names)

            ip_handle = ctx.get_ip()
            for ind, arg in enumerate(iter_args):  # Traversal
                stage_handle = stage_name if ind == 0 else ""
                ctx.set_ip(ip_handle)
                for_op = ASTTransformer.build_single_for(
                    ctx, [arg], stage_handle, names[ind]
                )
                ctx.pop_ip()
                # Iteration Update
                ip_handle = InsertionPoint(for_op.body.operations[0])
                for_loops.append(for_op)
                if not isinstance(for_op, affine_d.AffineForOp):
                    is_affine = False

        elif attr == "range":
            if stage_name is None:
                stage_name = "S_" + "_".join(names)

            for_op = ASTTransformer.build_single_for(
                ctx, iter_args, stage_name, names[0]
            )
            for_loops = [for_op]
            if not isinstance(for_op, affine_d.AffineForOp):
                is_affine = False

        with ctx.block_scope_guard():
            ivs = [loop.induction_variable for loop in for_loops]
            for name, iv in zip(names, ivs):
                ctx.put_symbol(name=name, val=MockArg(iv, is_affine))
            ctx.set_ip(for_loops[-1].body.operations[0])

            # build loop body
            build_stmts(ctx, node.body)

            # attach necessary attributes
            if (
                isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr == "reduction"
            ):
                for loop in for_loops:
                    loop.attributes["reduction"] = UnitAttr.get()

        for_loops = None
        # Not sure why the for loops will not be collected if we do not call gc.collect()
        gc.collect()
        ctx.pop_ip()

    @staticmethod
    def build_For(ctx: ASTContext, node: ast.For):
        if node.orelse:
            raise RuntimeError("'else' clause for 'for' not supported in Allo kernels")
        with ctx.loop_scope_guard():
            if isinstance(node.iter, ast.Call):
                obj = ASTResolver.resolve(node.iter.func, ctx.global_vars)
                if (
                    obj is None
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == "range"
                ):
                    return ASTTransformer.build_all_for(ctx, node, "range")
                if obj is not None and obj.__name__ in {"grid", "reduction"}:
                    return ASTTransformer.build_all_for(ctx, node, obj.__name__)
            raise RuntimeError("Unsupported for loop")

    # pylint: disable=too-many-branches
    @staticmethod
    def build_cast_op(
        ctx: ASTContext,
        op: OpView,
        src_type: AlloType,
        res_type: AlloType,
        shape: list[int] = None,
    ):
        # No need to cast
        if type(res_type) is type(src_type) and res_type == src_type:
            return op

        # Single-step type conversions
        cast_map = {
            # Index <-> UInt/Int
            (Int, Index): arith_d.IndexCastOp,
            (UInt, Index): arith_d.IndexCastOp,
            (Index, Int): arith_d.IndexCastOp,
            (Index, UInt): arith_d.IndexCastOp,
            # UInt/Int <-> Float
            (Int, Float): arith_d.SIToFPOp,
            (UInt, Float): arith_d.UIToFPOp,
            (Float, Int): arith_d.FPToSIOp,
            (Float, UInt): arith_d.FPToUIOp,
            # Float <-> Fixed/UFixed
            (Float, Fixed): allo_d.FloatToFixedOp,
            (Float, UFixed): allo_d.FloatToFixedOp,
            (Fixed, Float): allo_d.FixedToFloatOp,
            (UFixed, Float): allo_d.FixedToFloatOp,
            # Int/UInt <-> Fixed/UFixed
            (Fixed, Int): allo_d.FixedToIntOp,
            (Fixed, UInt): allo_d.FixedToIntOp,
            (UFixed, Int): allo_d.FixedToIntOp,
            (UFixed, UInt): allo_d.FixedToIntOp,
            (Int, Fixed): allo_d.IntToFixedOp,
            (Int, UFixed): allo_d.IntToFixedOp,
            (UInt, Fixed): allo_d.IntToFixedOp,
            (UInt, UFixed): allo_d.IntToFixedOp,
            # Fixed/UFixed <-> Fixed/UFixed
            (Fixed, Fixed): allo_d.FixedToFixedOp,
            (Fixed, UFixed): allo_d.FixedToFixedOp,
            (UFixed, Fixed): allo_d.FixedToFixedOp,
            (UFixed, UFixed): allo_d.FixedToFixedOp,
        }
        op_result = get_mlir_op_result(op)
        if (type(src_type), type(res_type)) in cast_map:
            opcls = cast_map[(type(src_type), type(res_type))]
        elif isinstance(src_type, Float) and isinstance(res_type, Index):
            # FP to Index is not supported in MLIR
            # we need to cast to UInt first, then cast to Index
            op = arith_d.FPToUIOp(
                IntegerType.get_signless(32), op_result, ip=ctx.get_ip()
            )
            opcls = arith_d.IndexCastOp  # proceed to build cast to index
        elif isinstance(src_type, Index) and isinstance(res_type, Float):
            op = arith_d.IndexCastOp(
                IntegerType.get_signless(32), op_result, ip=ctx.get_ip()
            )
            opcls = arith_d.SIToFPOp  # proceed to build cast to float
        elif isinstance(src_type, Index) and isinstance(res_type, (Fixed, UFixed)):
            op = arith_d.IndexCastOp(
                IntegerType.get_signless(32), op_result, ip=ctx.get_ip()
            )
            opcls = allo_d.IntToFixedOp  # proceed to build cast to float
        elif isinstance(src_type, (Fixed, UFixed)) and isinstance(res_type, Index):
            op = allo_d.FixedToIntOp(
                IntegerType.get_signless(32), op_result, ip=ctx.get_ip()
            )
            opcls = arith_d.IndexCastOp
        elif isinstance(src_type, (Int, UInt)) and isinstance(res_type, (Int, UInt)):
            if src_type.bits > res_type.bits:
                opcls = arith_d.TruncIOp
            elif src_type.bits == res_type.bits:
                return op
            else:  # src_type.bits < res_type.bits
                # pylint: disable=else-if-used
                if (
                    isinstance(
                        op, (allo_d.GetIntBitOp, allo_d.GetIntSliceOp, arith_d.ShLIOp)
                    )
                    or src_type.bits == 1
                ):
                    opcls = arith_d.ExtUIOp
                elif isinstance(src_type, UInt):
                    opcls = arith_d.ExtUIOp
                else:
                    opcls = arith_d.ExtSIOp
        elif isinstance(src_type, Float) and isinstance(res_type, Float):
            if res_type.bits < src_type.bits:
                opcls = arith_d.TruncFOp
            elif res_type.bits > src_type.bits:
                opcls = arith_d.ExtFOp
            else:
                return op
        elif isinstance(src_type, Struct) and isinstance(res_type, Struct):
            # We don't actually cast between struct types,
            # here we check if two structs are identical when all
            # integer fields are signless.
            if len(src_type.dtype_dict) != len(res_type.dtype_dict):
                raise DTypeError(
                    "Casting between structs with different number of fields. "
                    + f"src type: {src_type}, dst type: {res_type}"
                )
            for res_ftype, src_ftype in zip(
                res_type.dtype_dict.values(), src_type.dtype_dict.values()
            ):
                if isinstance(src_ftype, (Int, UInt)) and isinstance(
                    res_ftype, (Int, UInt)
                ):
                    if src_ftype.width != res_ftype.width:
                        raise DTypeError(
                            "Casting between structs with different field width. "
                            + f"src type: {src_type}, dst type: {res_type}"
                        )
                else:
                    raise DTypeError(
                        "Casting between structs with different field types. "
                        + f"src type: {src_type}, dst type: {res_type}"
                    )
            return op
        elif isinstance(src_type, (Int, UInt)) and isinstance(res_type, Struct):
            # Int -> Struct Cast
            def is_all_field_int(dtype):
                """Check if a struct type has all integer fields
                When it has nested struct field, recursively check
                the nested struct field.
                """
                if not isinstance(dtype, Struct):
                    return False
                for field_type in dtype.dtype_dict.values():
                    if isinstance(field_type, Struct):
                        if not is_all_field_int(field_type):
                            return False
                    elif not isinstance(field_type, (Int, UInt)):
                        return False
                return True

            if not is_all_field_int(res_type):
                raise DTypeError(
                    "Casting from integer to struct with non-integer fields. "
                    + f"src type: {src_type}, dst type: {res_type}"
                )

            def get_struct_bitwidth(struct_type):
                bitwidth = 0
                for field in struct_type.dtype_dict.values():
                    if isinstance(field, Struct):
                        bitwidth += get_struct_bitwidth(field)
                    else:
                        bitwidth += field.bits
                return bitwidth

            total_width = get_struct_bitwidth(res_type)
            if total_width != src_type.bits:
                raise DTypeError(
                    "Casting from integer to struct with different width. "
                    + f"src type: {src_type}, dst type: {res_type}"
                )
            opcls = allo_d.IntToStructOp
        elif isinstance(src_type, Struct) and isinstance(res_type, (Int, UInt)):
            # Struct -> Int Cast
            raise NotImplementedError(
                "Struct -> Int Cast is not implemented yet. "
                + "We plan to add an as_int() API for struct values."
            )
        else:
            raise DTypeError(
                "Casting between unsupported types. "
                + f"src type: {src_type}, dst type: {res_type}"
            )
        mlir_type = res_type.build()
        if isinstance(res_type, (Int, UInt, Struct)):
            # use linalg.generic to cast tensors by element
            if shape is not None and len(shape) > 0:
                # create output tensor
                alloc_op = ASTTransformer.build_array(ctx, res_type, shape)
                # get mapping from index to index
                index_exprs = []
                for dim in range(len(shape)):
                    index_exprs.append(AffineExpr.get_dim(dim))
                affine_map = AffineMap.get(
                    dim_count=len(shape),
                    symbol_count=0,
                    exprs=index_exprs,
                )
                indexing_maps_attr = ArrayAttr.get(
                    [AffineMapAttr.get(affine_map), AffineMapAttr.get(affine_map)]
                )
                iterator_types_attr = ArrayAttr.get(
                    [Attribute.parse("#linalg.iterator_type<parallel>")] * len(shape)
                )
                cast_op = linalg_d.GenericOp(
                    indexing_maps=indexing_maps_attr,
                    ip=ctx.get_ip(),
                    inputs=[get_mlir_op_result(op)],
                    outputs=[alloc_op.result],
                    result_tensors=(
                        [RankedTensorType.get(shape, mlir_type)]
                        if ctx.enable_tensor
                        else []
                    ),
                    iterator_types=iterator_types_attr,
                )
                cast_op.attributes["cast_from"] = StringAttr.get(str(src_type))
                cast_op.attributes["cast_to"] = StringAttr.get(str(res_type))
                # create block
                block_arg_types = [src_type.build(), mlir_type]
                block = cast_op.regions[0].blocks.append(*block_arg_types)
                ctx.set_ip(block)
                # add cast op to block
                yield_value = opcls(mlir_type, block.arguments[0], ip=ctx.get_ip())
                linalg_d.YieldOp([yield_value], ip=ctx.get_ip())
                ctx.pop_ip()
                cast_op = cast_op if ctx.enable_tensor else alloc_op
            else:
                cast_op = opcls(mlir_type, get_mlir_op_result(op), ip=ctx.get_ip())
            if isinstance(res_type, UInt):
                cast_op.attributes["unsigned"] = UnitAttr.get()
        else:
            cast_op = opcls(mlir_type, get_mlir_op_result(op), ip=ctx.get_ip())
        return cast_op

    @staticmethod
    def build_broadcast_op(
        ctx: ASTContext,
        op: OpView,
        dtype: AlloType,
        src_shape: list[int],
        dst_shape: list[int],
        dims: list[int],
    ):
        # No shape checking in this function, since it has been done in
        # type inference pass in infer.py
        if src_shape == dst_shape:
            return op
        if len(src_shape) == 0:
            # Get zero-rank memref for constant
            in_cst = ASTTransformer.build_array(ctx, dtype, tuple())
            with ctx.get_ip():
                fill = linalg_d.fill(get_mlir_op_result(op), outs=[in_cst.result])
            op = fill.owner if ctx.enable_tensor else in_cst
        # target
        alloc_op = ASTTransformer.build_array(ctx, dtype, dst_shape)
        with ctx.get_ip():
            broadcast_op = linalg_d.broadcast(
                input=get_mlir_op_result(op),
                outs=[alloc_op.result],
                dimensions=dims,
            )
        return broadcast_op if ctx.enable_tensor else alloc_op

    @staticmethod
    def build_general_binop(
        ctx: ASTContext, node: ast.BinOp | ast.AugAssign, lhs: OpView, rhs: OpView
    ):
        if len(node.shape) > 0:
            attr = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
            }.get(type(node.op))
            return ASTTransformer.build_library_op(
                ctx, node=node, attr=attr, new_args=[lhs, rhs]
            )

        # scalar operations
        opcls = {
            ast.Add: {
                Float: arith_d.AddFOp,
                Int: arith_d.AddIOp,
                UInt: arith_d.AddIOp,
                Fixed: allo_d.AddFixedOp,
                UFixed: allo_d.AddFixedOp,
            },
            ast.Sub: {
                Float: arith_d.SubFOp,
                Int: arith_d.SubIOp,
                UInt: arith_d.SubIOp,
                Fixed: allo_d.SubFixedOp,
                UFixed: allo_d.SubFixedOp,
            },
            ast.Mult: {
                Float: arith_d.MulFOp,
                Int: arith_d.MulIOp,
                UInt: arith_d.MulIOp,
                Fixed: allo_d.MulFixedOp,
                UFixed: allo_d.MulFixedOp,
            },
            ast.Div: {
                Float: arith_d.DivFOp,
                Int: arith_d.DivSIOp,
                UInt: arith_d.DivUIOp,
                Fixed: allo_d.DivFixedOp,
                UFixed: allo_d.DivFixedOp,
            },
            ast.FloorDiv: {
                Float: RuntimeError,
                Int: arith_d.FloorDivSIOp,
                UInt: RuntimeError,
                Fixed: RuntimeError,
                UFixed: RuntimeError,
            },
            ast.Mod: {
                Float: arith_d.RemFOp,
                Int: arith_d.RemSIOp,
                UInt: arith_d.RemUIOp,
                Fixed: RuntimeError,
                UFixed: RuntimeError,
            },
            ast.Pow: {
                Float: math_d.PowFOp,
                Int: RuntimeError,
                UInt: RuntimeError,
                Fixed: RuntimeError,
                UFixed: RuntimeError,
            },
            ast.LShift: {
                Float: RuntimeError,
                Int: arith_d.ShLIOp,
                UInt: arith_d.ShLIOp,
                Fixed: allo_d.ShLFixedOp,
                UFixed: allo_d.ShLFixedOp,
            },
            ast.RShift: {
                Float: RuntimeError,
                Int: arith_d.ShRSIOp,
                UInt: arith_d.ShRUIOp,
                Fixed: allo_d.ShRFixedOp,
                UFixed: allo_d.ShRFixedOp,
            },
            ast.BitOr: {
                Float: RuntimeError,
                Int: arith_d.OrIOp,
                UInt: arith_d.OrIOp,
                Fixed: RuntimeError,
                UFixed: RuntimeError,
            },
            ast.BitXor: {
                Float: RuntimeError,
                Int: arith_d.XOrIOp,
                UInt: arith_d.XOrIOp,
                Fixed: RuntimeError,
                UFixed: RuntimeError,
            },
            ast.BitAnd: {
                Float: RuntimeError,
                Int: arith_d.AndIOp,
                UInt: arith_d.AndIOp,
                Fixed: RuntimeError,
                UFixed: RuntimeError,
            },
        }.get(type(node.op))
        ty_cls = Int if isinstance(node.dtype, Index) else type(node.dtype)
        if isinstance(node.op, (ast.LShift, ast.RShift)) and isinstance(
            node.dtype, (Fixed, UFixed)
        ):
            op = opcls[ty_cls](
                node.dtype.build(),
                get_mlir_op_result(lhs),
                get_mlir_op_result(rhs),
                ip=ctx.get_ip(),
            )
        else:
            op = opcls[ty_cls](
                get_mlir_op_result(lhs), get_mlir_op_result(rhs), ip=ctx.get_ip()
            )
        if isinstance(node.dtype, UInt):
            op.attributes["unsigned"] = UnitAttr.get()
        return op

    @staticmethod
    def build_UnaryOp(ctx: ASTContext, node: ast.UnaryOp):
        value = build_stmt(ctx, node.operand)
        value_result = get_mlir_op_result(value)
        if isinstance(node.op, ast.USub):
            # MLIR does not provide integer negation
            if isinstance(node.dtype, (Int, UInt)):
                value = ASTTransformer.build_cast_op(
                    ctx, value, node.operand.dtype, node.dtype
                )
                return arith_d.SubIOp(
                    # pylint: disable=too-many-function-args
                    arith_d.ConstantOp(node.dtype.build(), 0, ip=ctx.get_ip()).result,
                    value_result,
                    ip=ctx.get_ip(),
                )
            # float
            value = ASTTransformer.build_cast_op(
                ctx, value, node.operand.dtype, node.dtype
            )
            return arith_d.NegFOp(value_result, ip=ctx.get_ip())
        if isinstance(node.op, ast.Not):
            if not (
                isinstance(value_result.type, IntegerType)
                and value_result.type.width == 1
            ):
                raise RuntimeError("The operand of 'not' should be a boolean value")
            # test if value.val is a bool value
            # pylint: disable=too-many-function-args
            c0 = arith_d.ConstantOp(IntegerType.get_signless(1), 0, ip=ctx.get_ip())
            # predicate=0 means "eq"
            predicate = IntegerAttr.get(IntegerType.get_signless(64), 0)
            return arith_d.CmpIOp(predicate, value_result, c0, ip=ctx.get_ip())
        # ast.UAdd
        return value

    @staticmethod
    def build_BinOp(ctx: ASTContext, node: ast.BinOp):
        lhs = build_stmt(ctx, node.left)
        rhs = build_stmt(ctx, node.right)
        # Cast lhs and rhs to the same type
        lhs = ASTTransformer.build_cast_op(
            ctx, lhs, node.left.dtype, node.dtype, node.left.shape
        )
        if isinstance(node.op, (ast.LShift, ast.RShift)) and isinstance(
            node.dtype, (Fixed, UFixed)
        ):
            target_rhs_type = Int(32)
        else:
            target_rhs_type = node.dtype
        rhs = ASTTransformer.build_cast_op(
            ctx, rhs, node.right.dtype, target_rhs_type, node.right.shape
        )
        lhs = ASTTransformer.build_broadcast_op(
            ctx, lhs, node.dtype, node.left.shape, node.shape, node.dims[0]
        )
        rhs = ASTTransformer.build_broadcast_op(
            ctx, rhs, node.dtype, node.right.shape, node.shape, node.dims[1]
        )
        return ASTTransformer.build_general_binop(ctx, node, lhs, rhs)

    @staticmethod
    def build_indices(ctx: ASTContext, node: ast.AST, enable_affine: bool = True):
        indices = node.value if isinstance(node, ast.Index) else node
        elts = indices.elts if isinstance(indices, ast.Tuple) else [indices]
        ctx.dim_count = 0
        ctx.affine_vars = []
        new_indices = []
        if not ctx.enable_tensor and enable_affine:
            is_affine = True
            for index in elts:
                expr = ASTTransformer.build_affine_expr(ctx, index)
                if expr is None:
                    is_affine = False
                    break
                new_indices.append(expr)
            if is_affine:
                return new_indices, True
        # not affine
        new_indices = []
        for index in elts:
            expr = build_stmt(ctx, index)
            expr = ASTTransformer.build_cast_op(ctx, expr, index.dtype, Index())
            new_indices.append(get_mlir_op_result(expr))
        return new_indices, False

    @staticmethod
    def build_assign_stmt(
        ctx: ASTContext, target: ast.expr, value: ast.expr, val=None, idx=None
    ):
        """
        Parameters `val` and `idx` are used when the right-hand side of an assign statement
        is a function call that returns a tuple.
        Type casting is currently not supported for this kind of case.
        """
        # Compute RHS
        conversion_enabled = False
        if val is not None:
            rhs = val
            conversion_enabled = idx is None
        elif hasattr(value, "np_values"):
            # - np array -> constant tensor
            rhs = ASTTransformer.build_constant_tensor(
                ctx, target, value.np_values, dtype=target.dtype, shape=target.shape
            )
        else:
            # - other
            rhs = build_stmt(ctx, value)
            conversion_enabled = rhs is not None
        if conversion_enabled:
            if isinstance(rhs, (allo_d.StreamConstructOp, allo_d.StreamGetOp)):
                pass
            else:
                # dtype cast & broadcast
                # for some OpView (e.g., linalg.transpose), op.result return a OpResultList
                rhs = ASTTransformer.build_cast_op(
                    ctx, rhs, value.dtype, target.dtype, value.shape
                )
                rhs_result = get_mlir_op_result(rhs)
                # - scalar -> tesnor: use linalg_d.fill to broadcast
                if len(target.shape) > 0 and len(value.shape) == 0:
                    alloc_op = ASTTransformer.build_array(
                        ctx, target.dtype, target.shape
                    )
                    with ctx.get_ip():
                        linalg_op = linalg_d.fill(rhs_result, outs=[alloc_op.result])
                    rhs = linalg_op.owner if ctx.enable_tensor else alloc_op
                else:
                    rhs = ASTTransformer.build_broadcast_op(
                        ctx,
                        rhs,
                        target.dtype,
                        value.shape,
                        target.shape,
                        target.dims[1],  # rhs
                    )
        if rhs is None:
            rhs_value = None
        elif idx is None:
            rhs_value = get_mlir_op_result(rhs)
        else:
            rhs_value = rhs.results[idx]
        if isinstance(target, ast.Name):
            target_ = ctx.get_symbol(name=target.id, allow_missing=True)
            if target_ is None:
                # declare
                # - if rhs is constant, allocate on stack to make it a real variable
                if rhs_value is None or not isinstance(
                    rhs_value.type, (MemRefType, RankedTensorType)
                ):
                    if ctx.enable_tensor and rhs is not None:
                        alloc_op = rhs
                    else:
                        alloc_op = ASTTransformer.build_array(
                            ctx, target.dtype, target.shape
                        )
                    ctx.buffers[target.id] = alloc_op
                    if isinstance(alloc_op, OpView):
                        alloc_op.attributes["name"] = StringAttr.get(target.id)
                    ctx.put_symbol(name=target.id, val=alloc_op)
                    if not ctx.enable_tensor and rhs is not None:
                        ASTTransformer.build_assign_value(
                            ctx, target, buffer=alloc_op, val=rhs_value
                        )
                else:
                    # if rhs is already a buffer def, we need to allocate a new buffer and copy the value
                    # FIXME (Shihan): It would be better if we can remove `isinstance(rhs, memref_d.GetGlobalOp)` in the condition.
                    #                   I haven't fully figured out how `partition` is propagated to global variables,
                    #                   and removing this filtering for `GetGlobalOp` might cause performance regressions in several test cases.
                    if isinstance(rhs, MockArg) or (
                        hasattr(rhs, "attributes")
                        and not isinstance(rhs, memref_d.GetGlobalOp)
                        and "name" in rhs.attributes
                    ):
                        alloc_op = ASTTransformer.build_array(
                            ctx, target.dtype, target.shape
                        )
                        with ctx.get_ip():
                            linalg_op = linalg_d.copy(
                                get_mlir_op_result(rhs), outs=[alloc_op.result]
                            )
                        rhs = linalg_op.owner if ctx.enable_tensor else alloc_op
                    assert idx is None, "Not Supported"
                    ctx.buffers[target.id] = rhs
                    # FIXME (Shihan): GetGlobalOp has a "name" attribute, which may have assignment conflict
                    if "name" in rhs.attributes:
                        assert rhs.attributes["name"].value == target.id
                    else:
                        rhs.attributes["name"] = StringAttr.get(target.id)
                    ctx.put_symbol(name=target.id, val=rhs)
            else:
                # FIXME (Shihan): We may need to look for some workarounds to support such cases.
                assert (
                    not ctx.enable_tensor
                ), "MLIR tensors are immutable data structures, fail to resolve"
                ASTTransformer.build_assign_value(
                    ctx, target, buffer=target_, val=rhs_value
                )
        elif idx is not None:
            build_stmt(ctx, target, val=rhs, idx=idx)
        else:
            build_stmt(ctx, target, val=rhs)

    @staticmethod
    def build_Assign(ctx: ASTContext, node: ast.Assign):
        targets, values = [], []
        rhs_visited = False
        if isinstance(node.targets[0], ast.Tuple):
            targets.extend(node.targets[0].elts)
            if isinstance(node.value, ast.Tuple):
                values.extend(node.value.elts)
            else:
                values.append(node.value)
        else:
            targets.append(node.targets[0])
            values.append(node.value)
        if isinstance(node.value, ast.Call):
            # special case: the builtin get_pid()
            if (
                isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "get_pid"
            ):
                for i, target in enumerate(targets):
                    # TODO: add target symbol for pid?? # pid = MockConstant(ctx.global_vars[f"df.p{i}"], ctx, dtype=Index())
                    ctx.global_vars[ast.unparse(target)] = ctx.global_vars[f"df.p{i}"]
                    ctx.symbolic[ast.unparse(target)] = f"p{i}"
                return None
            rhs = build_stmt(ctx, node.value)
            rhs_visited = True
        for idx, target in enumerate(targets):
            if rhs_visited:
                ASTTransformer.build_assign_stmt(
                    ctx,
                    target,
                    values[0],
                    val=rhs,
                    idx=idx if len(targets) > 1 else None,
                )
            else:
                ASTTransformer.build_assign_stmt(ctx, target, values[idx])
        return None

    @staticmethod
    def build_constant_tensor(
        ctx: ASTContext,
        node: ast.AST,
        np_values: np.array,
        dtype: AlloType,
        shape: list[int],
        constant: bool = False,
    ):
        value_attr = DenseElementsAttr.get(np_values, type=dtype.build())
        assert dtype is not None and shape is not None
        if ctx.enable_tensor:
            tensor_type = RankedTensorType.get(shape, dtype.build())
            # pylint: disable=too-many-function-args
            const_tensor = arith_d.ConstantOp(tensor_type, value_attr, ip=ctx.get_ip())
        else:
            if hasattr(node, "id"):
                name = node.id
            else:
                name = f"const_{abs(hash(str(np_values)))}"
            if name not in ASTTransformer.global_var_cnt:
                ASTTransformer.global_var_cnt[name] = 0
            else:
                cnt = ASTTransformer.global_var_cnt[name]
                ASTTransformer.global_var_cnt[name] = cnt + 1
                name = f"{name}_{cnt}"
            sym_name = StringAttr.get(name)
            sym_visibility = StringAttr.get("private")
            memref_type = MemRefType.get(shape, dtype.build())
            type_attr = TypeAttr.get(memref_type)
            memref_d.GlobalOp(
                sym_name=sym_name,
                type_=type_attr,
                sym_visibility=sym_visibility,
                initial_value=value_attr,
                # TODO: Use dataflow analysis to determine whether some store ops
                #       are operated on this tensor
                constant=constant,
                alignment=None,
                ip=InsertionPoint(ctx.top_func),
            )
            const_tensor = memref_d.GetGlobalOp(
                memref_type,
                FlatSymbolRefAttr.get(name),
                ip=ctx.get_ip(),
            )
        return const_tensor

    @staticmethod
    def build_AugAssign(ctx: ASTContext, node: ast.AugAssign):
        # Compute RHS
        rhs = build_stmt(ctx, node.value)
        # Load LHS
        node.target.ctx = ast.Load()
        lhs = build_stmt(ctx, node.target)
        node.target.ctx = ast.Store()
        # Cast rhs to the target type
        rhs = ASTTransformer.build_cast_op(ctx, rhs, node.value.dtype, node.dtype)
        # Aug LHS
        res = ASTTransformer.build_general_binop(ctx, node, lhs, rhs)
        # Store LHS
        store_op = build_stmt(ctx, node.target, val=res)
        return store_op

    @staticmethod
    def build_affine_map_attr(ctx: ASTContext, node: ast.AST):
        with ctx.affine_scope_guard():
            expr = ASTTransformer.build_affine_expr(ctx, node)
            if expr is not None:
                variables = [ctx.get_symbol(x).result for x in ctx.affine_vars]
                affine_map = AffineMap.get(
                    dim_count=ctx.dim_count, symbol_count=0, exprs=[expr]
                )
                # attr = AffineMapAttr.get(affine_map)
                attr = affine_map
            else:
                variables, attr = [], None
        return variables, attr

    @staticmethod
    def build_affine_expr(ctx: ASTContext, node: ast.AST):
        if isinstance(node, ast.Name):
            val = ctx.get_symbol(node.id, allow_missing=True)
            if (
                val is not None
                and isinstance(val, MockArg)
                and str(val.result.type) == "index"
                and val.is_affine
            ):
                ctx.dim_count += 1
                ctx.affine_vars.append(node.id)
                return AffineExpr.get_dim(ctx.dim_count - 1)
            if (
                # Note: Variables may be changed inside a loop (or a non-top-down structure),
                # so we cannot safely take its value as a constant
                # e.g.,
                # x = 0
                # for i in range(10):
                #     A[x] = 1
                #     x = x + 1
                ctx.nested_loops == 0
                and val is not None
                and isinstance(val, MockScalar)
                and isinstance(val.dtype, Index)
            ):
                return ASTTransformer.build_affine_expr(ctx, val.value)
            if node.id in ctx.global_vars and isinstance(ctx.global_vars[node.id], int):
                return ASTTransformer.build_affine_expr(
                    ctx, ast.Constant(ctx.global_vars[node.id])
                )
            return None
        if isinstance(node, ast.BinOp):
            lhs = ASTTransformer.build_affine_expr(ctx, node.left)
            rhs = ASTTransformer.build_affine_expr(ctx, node.right)
            if lhs is None or rhs is None:
                return None
            op = {
                ast.Add: lambda l, r: l + r,
                ast.Sub: lambda l, r: l - r,
                ast.Mult: lambda l, r: l * r,
                ast.Div: AffineExpr.get_floor_div,
                ast.FloorDiv: AffineExpr.get_floor_div,
                ast.Mod: lambda l, r: l % r,
                ast.Pow: lambda l, r: l**r,
                ast.LShift: lambda l, r: l << r,
                ast.RShift: lambda l, r: l >> r,
                ast.BitOr: lambda l, r: l | r,
                ast.BitXor: lambda l, r: l ^ r,
                ast.BitAnd: lambda l, r: l & r,
            }.get(type(node.op))
            return op(lhs, rhs)
        if isinstance(node, ast.Constant):
            return AffineConstantExpr.get(node.value)
        return None

    @staticmethod
    def build_slices(ctx: ASTContext, node: ast.Subscript, in_shape: list[int]):
        # calculate the (static) offsets, sizes, strides for ExtractSlice and InsertSlice
        if isinstance(node.slice, ast.Tuple):
            slices = list(node.slice.elts)
        else:
            slices = node.slice.dims if len(node.shape) > 1 else [node.slice]
        offsets = []
        sizes = []
        strides = []
        static_offsets = []
        static_sizes = []
        static_strides = []
        for index, size in zip(slices, in_shape):
            if isinstance(index, ast.Slice):
                # fixme: slice lower/upper/step is not constant
                lower = (
                    0
                    if index.lower is None
                    else ASTResolver.resolve_constant(index.lower, ctx)
                )
                upper = (
                    size
                    if index.upper is None
                    else ASTResolver.resolve_constant(index.upper, ctx)
                )
                if index.step is None:
                    step = 1
                elif isinstance(index.step, ast.Constant):
                    step = index.step.value
                else:
                    raise RuntimeError("Unsupported step type")
                if lower is None:
                    static_offsets.append(ShapedType.get_dynamic_size())
                    offset_expr = build_stmt(ctx, index.lower)
                    static_sizes.append(ShapedType.get_dynamic_size())
                    if upper is None:
                        upper_expr = build_stmt(ctx, index.upper)
                        size_expr = tensor_d.FloorDivSOp(
                            tensor_d.SubOp(upper_expr, offset_expr).result, step
                        )
                    else:
                        size_expr = tensor_d.FloorDivSOp(
                            tensor_d.SubOp(upper, offset_expr).result, step
                        )
                    size = ASTTransformer.build_cast_op(
                        ctx, size_expr, index.dtype, Index()
                    ).result
                    continue
                if upper is None:
                    static_sizes.append(ShapedType.get_dynamic_size())
                    upper_expr = build_stmt(ctx, index.upper)
                    size_expr = tensor_d.FloorDivSOp(
                        tensor_d.SubOp(upper_expr, lower).result, step
                    )
                    size = ASTTransformer.build_cast_op(
                        ctx, size_expr, index.dtype, Index()
                    ).result
                    continue
                if lower < 0 or upper < 0:
                    raise RuntimeError("Unsupported negative index")
                if lower >= size or upper > size:
                    raise RuntimeError("Index out of range")
                if step <= 0:
                    raise RuntimeError("Unsupported negative step")
                if step > upper - lower:
                    raise RuntimeError("Step larger than range")
                static_offsets.append(lower)
                static_sizes.append((upper - lower) // step)
                static_strides.append(step)
            elif isinstance(index, (ast.Index, ast.Constant)):
                lower = (
                    index.value.value if isinstance(index, ast.Index) else index.value
                )
                if lower < 0:
                    raise RuntimeError("Unsupported negative index")
                if lower >= size:
                    raise RuntimeError("Index out of range")
                static_offsets.append(lower)
                static_sizes.append(1)
                static_strides.append(1)
                continue
            elif isinstance(index, ast.Name):
                index = build_stmt(ctx, index)
                if isinstance(index, MockConstant):
                    lower = index.val
                    if lower < 0:
                        raise RuntimeError("Unsupported negative index")
                    if lower >= size:
                        raise RuntimeError("Index out of range")
                    static_offsets.append(lower)
                    static_sizes.append(1)
                    static_strides.append(1)
                    continue
                if isinstance(index, MockArg):
                    offsets.append(index.val)
                    static_offsets.append(ShapedType.get_dynamic_size())
                    static_sizes.append(1)
                    static_strides.append(1)
                    continue
                raise ValueError(
                    f"Index type ({type(index)}) not supported. Dynamic offset is incompatible with many builtin opt passes."
                )
            else:
                raise ValueError(f"Unsupported slice index type ({type(index)})")
        return offsets, sizes, strides, static_offsets, static_sizes, static_strides

    @staticmethod
    def build_tensor_access(
        ctx: ASTContext, node: ast.Subscript, val: OpView = None, idx: int = 0
    ):
        # TODO: Fix tuple idx
        value = build_stmt(ctx, node.value)
        value_result = get_mlir_op_result(value)
        # access tensor
        if len(node.shape) >= 1:
            dtype = RankedTensorType(value_result.type).element_type
            in_shape = RankedTensorType(value_result.type).shape
            (
                offsets,
                sizes,
                strides,
                static_offsets,
                static_sizes,
                static_strides,
            ) = ASTTransformer.build_slices(ctx, node, in_shape)
            result = RankedTensorType.get(static_sizes, dtype)
            # pylint: disable=no-else-return
            if isinstance(node.ctx, ast.Load):
                return tensor_d.ExtractSliceOp(
                    result=result,
                    source=value_result,
                    static_sizes=static_sizes,
                    static_strides=static_strides,
                    static_offsets=static_offsets,
                    offsets=offsets,
                    sizes=sizes,
                    strides=strides,
                    ip=ctx.get_ip(),
                )
            else:  # ast.Store
                return tensor_d.InsertSliceOp(
                    source=get_mlir_op_result(val),
                    dest=value_result,
                    static_offsets=static_offsets,
                    static_sizes=static_sizes,
                    static_strides=static_strides,
                    offsets=[],
                    sizes=[],
                    strides=[],
                    ip=ctx.get_ip(),
                )
        # access an element
        if isinstance(node.slice, (ast.Index, ast.Tuple, ast.Constant)):
            index_exprs, _ = ASTTransformer.build_indices(ctx, node.slice)
            # pylint: disable=no-else-return
            if isinstance(node.ctx, ast.Load):
                return tensor_d.ExtractOp(
                    tensor=value_result,
                    indices=index_exprs,
                    ip=ctx.get_ip(),
                )
            else:  # ast.Store
                return tensor_d.InsertOp(
                    scalar=get_mlir_op_result(val),
                    dest=value_result,
                    indices=index_exprs,
                    ip=ctx.get_ip(),
                )
        raise RuntimeError("Unsupported load subscript")

    # pylint: disable=redefined-variable-type
    @staticmethod
    def build_memory_access(
        ctx: ASTContext, node: ast.Subscript, val: OpView = None, idx: int = 0
    ):
        value = build_stmt(ctx, node.value)
        value_result = get_mlir_op_result(value)
        if isinstance(node.slice, ast.Slice) or (
            isinstance(node.slice, ast.Tuple)
            and any(isinstance(elt, ast.Slice) for elt in node.slice.elts)
        ):
            dtype = MemRefType(value_result.type).element_type
            in_shape = MemRefType(value_result.type).shape
            (
                offsets,
                sizes,
                strides,
                static_offsets,
                static_sizes,
                static_strides,
            ) = ASTTransformer.build_slices(ctx, node, in_shape)
            orig_type = value_result.type
            orig_layout = orig_type.layout
            if isinstance(orig_layout, StridedLayoutAttr):
                orig_offset = orig_layout.offset
                orig_strides = orig_layout.strides
            elif isinstance(orig_layout, AffineMapAttr):
                # TODO: need to support non-identity affine map
                orig_offset = 0
                orig_strides = []
                times = 1
                for i in range(orig_type.rank):
                    orig_strides.append(times)
                    times *= orig_type.shape[orig_type.rank - i - 1]
                orig_strides = list(reversed(orig_strides))
            else:
                raise RuntimeError("Unsupported layout type")
            new_strides = [
                orig_strides[i] * static_strides[i] for i in range(len(static_strides))
            ]
            result_sizes = []
            result_strides = []
            for idx_, size in enumerate(static_sizes):
                if size > 1:
                    result_sizes.append(size)
                    result_strides.append(new_strides[idx_])
            new_offset = orig_offset + sum(
                o * s for o, s in zip(static_offsets, orig_strides)
            )
            if new_offset < 0:
                new_offset = "?"
            result = MemRefType.get(
                shape=result_sizes,
                element_type=dtype,
                layout=StridedLayoutAttr.get(
                    offset=(
                        ShapedType.get_dynamic_stride_or_offset()
                        if new_offset == "?"
                        else new_offset
                    ),
                    strides=result_strides,
                ),
            )
            subview = memref_d.SubViewOp(
                source=value_result,
                result=result,
                static_offsets=static_offsets,
                static_sizes=static_sizes,
                static_strides=static_strides,
                offsets=offsets,
                sizes=sizes,
                strides=strides,
                ip=ctx.get_ip(),
            )
            if isinstance(node.ctx, ast.Load):
                op = subview
            else:
                op = memref_d.CopyOp(
                    get_mlir_op_result(val), subview.result, ip=ctx.get_ip()
                )
        else:
            new_indices, is_affine = ASTTransformer.build_indices(ctx, node.slice)
            if len(node.value.shape) > len(new_indices):  # partial access
                # In this case, always access the first few dimensions
                assert (
                    is_affine
                ), "Non-affine memory access for memref.subview is not supported yet"
                static_strides = [1] * len(node.value.shape)
                static_offsets = [0] * len(node.value.shape)
                static_sizes = list(node.value.shape)
                slices = ASTResolver.resolve_slice(node.slice, ctx)
                if isinstance(slices, int):
                    slices = [slices]
                    offsets = []
                elif slices is None or slices == [None] * len(slices):
                    offsets, _ = ASTTransformer.build_indices(
                        ctx, node.slice, enable_affine=False
                    )
                    slices = [None] * len(offsets)
                else:
                    offsets = []
                offset = 0
                for i, index in enumerate(slices):
                    if isinstance(index, int):
                        static_offsets[i] = index
                        offset += int(index * np.prod(node.value.shape[i + 1 :]))
                    elif index is None:
                        static_offsets[i] = (
                            ShapedType.get_dynamic_size()
                        )  # dynamic offset
                        offset = "?"
                    else:
                        raise RuntimeError("Unsupported slice type")
                    static_sizes[i] = 1
                strides = [1]
                for i in range(len(node.shape) - 2, -1, -1):
                    strides.insert(0, strides[-1] * node.shape[i + 1])
                result = MemRefType.get(
                    shape=node.shape,
                    element_type=node.dtype.build(),
                    layout=StridedLayoutAttr.get(
                        offset=(
                            ShapedType.get_dynamic_stride_or_offset()
                            if offset == "?"
                            else offset
                        ),
                        strides=strides,
                    ),
                )
                subview = memref_d.SubViewOp(
                    source=value_result,
                    result=result,
                    static_offsets=static_offsets,
                    static_sizes=static_sizes,
                    static_strides=static_strides,
                    offsets=offsets,
                    sizes=[],
                    strides=[],
                    ip=ctx.get_ip(),
                )
                op = subview
            elif is_affine:
                affine_map = AffineMap.get(
                    dim_count=ctx.dim_count, symbol_count=0, exprs=new_indices
                )
                affine_attr = AffineMapAttr.get(affine_map)
                ivs = [ctx.get_symbol(x).result for x in ctx.affine_vars]
                if isinstance(node.ctx, ast.Load):
                    op = affine_d.AffineLoadOp(
                        node.value.dtype.build(),
                        value_result,
                        ivs,
                        affine_attr,
                        ip=ctx.get_ip(),
                    )
                    if isinstance(node.value.dtype, UInt):
                        op.attributes["unsigned"] = UnitAttr.get()
                else:  # ast.Store
                    op = affine_d.AffineStoreOp(
                        val.results[idx],
                        value_result,
                        ivs,
                        affine_attr,
                        ip=ctx.get_ip(),
                    )
            else:  # Not affine
                # pylint: disable=else-if-used
                if isinstance(node.ctx, ast.Load):
                    op = memref_d.LoadOp(value_result, new_indices, ip=ctx.get_ip())
                    if isinstance(node.value.dtype, UInt):
                        op.attributes["unsigned"] = UnitAttr.get()
                else:  # ast.Store
                    op = memref_d.StoreOp(
                        get_mlir_op_result(val),
                        value_result,
                        new_indices,
                        ip=ctx.get_ip(),
                    )
        attr = "from" if isinstance(node.ctx, ast.Load) else "to"
        op.attributes[attr] = StringAttr.get(node.value.id)
        return op

    # pylint: disable=inconsistent-return-statements
    @staticmethod
    def build_bit_operation(
        ctx: ASTContext, node: ast.Subscript, val: OpView = None, idx: int = 0
    ):
        # TODO: Fix tuple idx
        if not (
            len(node.value.shape) == 0 and isinstance(node.value.dtype, (Int, UInt))
        ):
            raise RuntimeError("Can only access bit (slice) for integers")
        # Bit operations should follow the convention in
        # https://github.com/cornell-zhang/heterocl/issues/443
        # >>> a = 0xabcd0123
        # >>> a[28:32] # containing the bit of 28, 29, 30, 31
        # 0xa
        # >>> a[4:24]
        # 0xcd012
        # >>> a[28:32].reverse()
        # 0x5
        value = build_stmt(ctx, node.value)
        value_result = get_mlir_op_result(value)
        if isinstance(node.slice, (ast.Index, ast.Constant, ast.Name, ast.BinOp)):
            index = build_stmt(ctx, node.slice)
            # pylint: disable=no-else-return
            if isinstance(node.ctx, ast.Load):
                index = ASTTransformer.build_cast_op(
                    ctx, index, node.slice.dtype, Index()
                )
                return allo_d.GetIntBitOp(
                    node.dtype.build(),
                    value_result,
                    index.result,
                    ip=ctx.get_ip(),
                )
            else:
                value_dtype = (
                    node.slice.value.dtype
                    if isinstance(node.slice, ast.Index)
                    else node.slice.dtype
                )
                index = ASTTransformer.build_cast_op(ctx, index, value_dtype, Index())
                # TODO: Test if rhs is uint1
                set_bit_op = allo_d.SetIntBitOp(
                    node.value.dtype.build(),
                    value_result,
                    index.result,
                    get_mlir_op_result(val),
                    ip=ctx.get_ip(),
                )
                # write the updated integer back to the scalar
                node.value.ctx = ast.Store()
                store_op = build_stmt(ctx, node.value, val=set_bit_op)
                return store_op

        if isinstance(node.slice, ast.Slice):
            # The backend implementation is different from the Python convention
            # The lower bound is inclusive and the upper bound is also inclusive
            lower = build_stmt(ctx, node.slice.lower)
            upper = build_stmt(ctx, node.slice.upper)
            cst = ASTTransformer.build_cast_op(
                ctx, MockConstant(1, ctx), Int(32), node.slice.upper.dtype
            )
            upper = arith_d.SubIOp(
                get_mlir_op_result(upper), cst.result, ip=ctx.get_ip()
            )
            lower = ASTTransformer.build_cast_op(
                ctx, lower, node.slice.lower.dtype, Index()
            )
            upper = ASTTransformer.build_cast_op(
                ctx, upper, node.slice.upper.dtype, Index()
            )
            # pylint: disable=no-else-return
            if isinstance(node.ctx, ast.Load):
                return allo_d.GetIntSliceOp(
                    node.dtype.build(),
                    value_result,
                    upper.result,
                    lower.result,
                    ip=ctx.get_ip(),
                )
            else:  # ast.Store
                set_slice_op = allo_d.SetIntSliceOp(
                    node.value.dtype.build(),
                    value_result,
                    upper.result,
                    lower.result,
                    val.result,
                    ip=ctx.get_ip(),
                )
                # write the updated integer back to the scalar
                node.value.ctx = ast.Store()
                store_op = build_stmt(ctx, node.value, val=set_slice_op)
                return store_op

    @staticmethod
    def build_Subscript(
        ctx: ASTContext, node: ast.Subscript, val: OpView = None, idx: int = 0
    ):
        # pylint: disable=no-else-return
        if len(node.value.shape) > 0 and not ctx.enable_tensor:
            return ASTTransformer.build_memory_access(ctx, node, val=val, idx=idx)
        elif len(node.value.shape) > 0 and ctx.enable_tensor:
            return ASTTransformer.build_tensor_access(ctx, node, val=val, idx=idx)
        elif isinstance(node.value.dtype, Struct):
            # Get the struct value
            value = build_stmt(ctx, node.value)
            value_result = get_mlir_op_result(value)
            # Get the field name from the string slice
            field_name = node.slice.value
            # Get the field index from the struct type
            field_idx = list(node.value.dtype.dtype_dict.keys()).index(field_name)
            # Create index attribute
            idx_attr = IntegerAttr.get(IntegerType.get_signless(64), field_idx)
            # Extract the field using struct get op
            return allo_d.StructGetOp(
                node.value.dtype[field_name].build(),
                value_result,
                idx_attr,
                ip=ctx.get_ip(),
            )
        else:  # bit operation
            return ASTTransformer.build_bit_operation(ctx, node, val=val, idx=idx)

    @staticmethod
    def build_Dict(ctx: ASTContext, node: ast.Dict):
        # Build each value in the dictionary
        values = [build_stmt(ctx, value) for value in node.values]

        # Create a struct construct op with the values
        return allo_d.StructConstructOp(
            node.dtype.build(),  # The struct type should already be inferred
            [get_mlir_op_result(value) for value in values],
            ip=ctx.get_ip(),
        )

    @staticmethod
    def build_AnnAssign(ctx: ASTContext, node: ast.AnnAssign):
        shape, dtype = node.target.shape, node.target.dtype
        # stream can only be declared with annotated assign stmt
        # TODO: guard, stream declaration has no rhs
        if isinstance(dtype, Stream):
            stream_type = allo_d.StreamType.get(dtype.build(), depth=dtype.depth)
            if len(shape) == 0:
                # single stream
                stream_op = allo_d.StreamConstructOp(stream_type, ip=ctx.get_ip())
                if isinstance(dtype.dtype, UInt):
                    stream_op.attributes["unsigned"] = UnitAttr.get()
                stream_op.attributes["name"] = StringAttr.get(node.target.id)
                ctx.buffers[node.target.id] = stream_op
                ctx.put_symbol(name=node.target.id, val=stream_op)
            else:
                # stream array
                for dim in np.ndindex(*shape):
                    stream_op = allo_d.StreamConstructOp(stream_type, ip=ctx.get_ip())
                    if isinstance(dtype.dtype, UInt):
                        stream_op.attributes["unsigned"] = UnitAttr.get()
                    # pylint: disable=bad-builtin
                    new_name = node.target.id + "_" + "_".join(map(str, dim))
                    stream_op.attributes["name"] = StringAttr.get(new_name)
                    ctx.buffers[new_name] = stream_op
                    assert ctx.get_symbol(name=new_name, allow_missing=True) is None, (
                        f"Instantiated stream name '{new_name}' conflicts with an existing symbol. "
                        f"Please choose a different name to avoid the conflict."
                    )
                    ctx.put_symbol(name=new_name, val=stream_op)
                # placeholder for array of FIFOs
                ctx.put_symbol(name=node.target.id, val=shape)
            return None
        return ASTTransformer.build_assign_stmt(ctx, node.target, node.value)

    @staticmethod
    def build_FunctionDef(ctx: ASTContext, node: ast.FunctionDef):
        func_name = node.name if ctx.func_id is None else f"{node.name}_{ctx.func_id}"
        # pylint: disable=too-many-nested-blocks
        if ctx.top_func is not None:
            # Nested function def
            # Create a new context to avoid name collision
            old_ctx = ctx
            ctx = old_ctx.copy()
            ctx.set_ip(old_ctx.top_func)
            ctx.top_func_tree = node
            ctx.buffers = old_ctx.buffers.copy()
            ctx.scopes = old_ctx.scopes
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute):
                        if decorator.func.attr == "kernel":
                            assert len(decorator.keywords) > 0, "Missing kernel mapping"
                            mapping = eval(
                                ast.unparse(decorator.keywords[0].value),
                                ctx.global_vars,
                            )
                            orig_name = node.name
                            if orig_name not in ctx.func_tag2instance:
                                ctx.func_tag2instance[orig_name] = {}
                            for dim in np.ndindex(*mapping):
                                if not ctx.unroll:
                                    # If not unrolled, assign tag to each instance.
                                    # Different tags indeicate different execution (control flow only)
                                    predicate_tag = freeze_list(
                                        ctx.func_predicate_tags[orig_name][dim]
                                    )
                                    if (
                                        predicate_tag
                                        in ctx.func_tag2instance[orig_name]
                                    ):
                                        continue
                                new_ctx = old_ctx.copy()
                                new_ctx.set_ip(old_ctx.top_func)
                                new_ctx.top_func_tree = node
                                new_ctx.buffers = old_ctx.buffers.copy()
                                new_ctx.scopes = old_ctx.scopes
                                new_ctx.global_vars = old_ctx.global_vars.copy()
                                for axis, val in enumerate(dim):
                                    new_ctx.global_vars.update(
                                        {"df.p" + str(axis): val}
                                    )
                                node.name = construct_kernel_name(orig_name, dim)
                                func_op = ASTTransformer.build_FunctionDef(
                                    new_ctx, node
                                )
                                func_op.attributes["df.kernel"] = UnitAttr.get()
                                if not ctx.unroll:
                                    func_op.attributes["tag"] = StringAttr.get(
                                        f"{orig_name}_{str(predicate_tag)}"
                                    )
                                    ctx.func_tag2instance[orig_name][
                                        predicate_tag
                                    ] = func_op
                            return
        else:
            old_ctx = None

        # Generic function
        if hasattr(node, "type_params") and len(node.type_params) > 0:
            assert len(ctx.inst) == len(
                node.type_params
            ), f"Type parameters mismatch, got {ctx.inst} and {node.type_params}"
            for type_var, call_val in zip(node.type_params, ctx.inst):
                name, call_val = resolve_generic_types(
                    ctx.global_vars, type_var, call_val
                )
                ctx.global_vars[name] = call_val

        # Build input types
        dtensors = []
        input_types = []
        input_typehints = []
        for i, arg in enumerate(node.args.args):
            if (
                len(ctx.call_args) > 0
                and isinstance(ctx.call_args[i].type, MemRefType)
                and arg.shape != tuple(ctx.call_args[i].type.shape)
            ):
                raise DTypeError(
                    f"Argument {i} of {node.name} shape mismatch, got {arg.shape} and {tuple(ctx.call_args[i].type.shape)}"
                )
            layout = (
                ctx.call_args[i].type.layout
                if len(ctx.call_args) > 0 and hasattr(ctx.call_args[i].type, "layout")
                else None
            )
            input_types.append(
                ASTTransformer.build_shaped_type(
                    ctx, arg.dtype, arg.shape, layout=layout
                )
            )
            input_typehints.append(get_extra_type_hints(arg.dtype))
            dtensors.append(arg.dtensor)

        # Build return type
        output_types = []
        output_typehints = []
        if not (
            (isinstance(node.returns, ast.Constant) and node.returns.value is None)
            or node.returns is None
        ):
            if isinstance(node.returns, ast.Tuple):
                # Multiple returns
                for ret in node.returns.elts:
                    output_types.append(
                        ASTTransformer.build_shaped_type(ctx, ret.dtype, ret.shape)
                    )
                    output_typehints.append(get_extra_type_hints(ret.dtype))
            else:
                # Single return
                output_types.append(
                    ASTTransformer.build_shaped_type(
                        ctx, node.returns.dtype, node.returns.shape
                    )
                )
                output_typehints.append(get_extra_type_hints(node.returns.dtype))

        # Build function
        with ctx.block_scope_guard():
            # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            func_type = FunctionType.get(input_types, output_types)
            func_op = func_d.FuncOp(name=func_name, type=func_type, ip=ctx.get_ip())
            func_op.add_entry_block()
            # attach type hints
            func_op.attributes["otypes"] = StringAttr.get("".join(output_typehints))
            func_op.attributes["itypes"] = StringAttr.get("".join(input_typehints))
            # set context
            ctx.top_func = func_op
            ctx.top_func_tree = node
            for i, (dtensor, arg) in enumerate(zip(dtensors, func_op.arguments)):
                name = dtensor.name
                mock_arg = MockArg(arg, idx=i)
                ctx.buffers[name] = mock_arg
                ctx.put_symbol(name=name, val=mock_arg)
            ctx.func_args[func_name] = dtensors
            ctx.set_ip(func_op.entry_block)
            stmts = build_stmts(ctx, node.body)
            # node.returns is the function definition, not the actual return operation
            if len(stmts) > 0 and not ctx.has_return:
                if (
                    isinstance(node.returns, ast.Constant)
                    and node.returns.value is None
                ) or node.returns is None:
                    func_d.ReturnOp([], ip=ctx.pop_ip())
                else:
                    raise RuntimeError("Missing return statement")
        # Recover the old context
        if old_ctx is not None:
            ctx = old_ctx
        # Add the built function to global variable for later reference
        ctx.global_vars[func_name] = func_op
        return func_op

    @staticmethod
    def build_Compare(ctx: ASTContext, node: ast.Compare, is_affine: bool = False):
        ATTR_MAP = {
            "int": {
                ast.Eq: 0,
                ast.NotEq: 1,
                ast.Lt: 2,
                ast.LtE: 3,
                ast.Gt: 4,
                ast.GtE: 5,
            },
            "uint": {
                ast.Eq: 0,
                ast.NotEq: 1,
                ast.Lt: 6,
                ast.LtE: 7,
                ast.Gt: 8,
                ast.GtE: 9,
            },
            "float": {
                "false": 0,
                ast.Eq: 1,
                ast.Gt: 2,
                ast.GtE: 3,
                ast.Lt: 4,
                ast.LtE: 5,
                ast.NotEq: 6,
                # The u prefix indicates unordered comparison, not unsigned comparison,
                # so “une” means unordered or not equal.
                # Unordered comparison of floating-point values refers to the comparison of
                # floating-point numbers in a way that takes into account special cases like
                # NaN (Not-a-Number) values and considers them as unordered
                # with respect to other values
                "ord": 7,
                "ueq": 8,
                "ugt": 9,
                "uge": 10,
                "ult": 11,
                "ule": 12,
                "une": 13,
                "uno": 14,
                "true": 15,
            },
            "fixed": {
                ast.Eq: 0,
                ast.NotEq: 1,
                ast.Lt: 2,
                ast.LtE: 3,
                ast.Gt: 4,
                ast.GtE: 5,
            },
            "ufixed": {
                ast.Eq: 0,
                ast.NotEq: 1,
                ast.Lt: 6,
                ast.LtE: 7,
                ast.Gt: 8,
                ast.GtE: 9,
            },
        }
        # pylint: disable=no-else-return
        if is_affine:
            eq_flags = []
            cond_op = node.ops[0]
            if not isinstance(cond_op, ast.Eq):
                raise NotImplementedError("Only support '==' for now")
            exprs = []
            exprs.append(
                AffineExpr.get_dim(0)
                - AffineConstantExpr.get(node.comparators[0].value)
            )
            eq_flags.append(True)
            if_cond_set = IntegerSet.get(1, 0, exprs, eq_flags)
            attr = allo_d.IntegerSetAttr.get(if_cond_set)
            return attr, ctx.get_symbol(node.left.id)
        else:
            lhs = build_stmt(ctx, node.left)
            rhs = build_stmt(ctx, node.comparators[0])
            # Cast lhs and rhs to the same type
            lhs = ASTTransformer.build_cast_op(ctx, lhs, node.left.dtype, node.dtype)
            rhs = ASTTransformer.build_cast_op(
                ctx, rhs, node.comparators[0].dtype, node.dtype
            )
            # avoid rebuilding the same op
            rhs_res, lhs_res = get_mlir_op_result(rhs), get_mlir_op_result(lhs)
            dtype = str(rhs_res.type)
            if dtype.startswith("i") or dtype.startswith("ui"):
                op = ATTR_MAP["int" if dtype.startswith("i") else "uint"][
                    type(node.ops[0])
                ]
                predicate = IntegerAttr.get(IntegerType.get_signless(64), op)
                return arith_d.CmpIOp(predicate, lhs_res, rhs_res, ip=ctx.get_ip())
            if dtype.startswith("!allo.Fixed") or dtype.startswith("!allo.UFixed"):
                op = ATTR_MAP["fixed" if dtype.startswith("f") else "ufixed"][
                    type(node.ops[0])
                ]
                predicate = IntegerAttr.get(IntegerType.get_signless(64), op)
                return allo_d.CmpFixedOp(predicate, lhs_res, rhs_res, ip=ctx.get_ip())
            if dtype.startswith("f"):
                op = ATTR_MAP["float"][type(node.ops[0])]
                predicate = IntegerAttr.get(IntegerType.get_signless(64), op)
                return arith_d.CmpFOp(predicate, lhs_res, rhs_res, ip=ctx.get_ip())
            raise RuntimeError(f"Unsupported types for binary op: {dtype}")

    @staticmethod
    def build_BoolOp(ctx: ASTContext, node: ast.BoolOp):
        stmts = build_stmts(ctx, node.values)
        opcls = {
            ast.And: arith_d.AndIOp,
            ast.Or: arith_d.OrIOp,
        }.get(type(node.op))
        result = opcls(
            get_mlir_op_result(stmts[0]), get_mlir_op_result(stmts[1]), ip=ctx.get_ip()
        )
        for i in range(2, len(stmts)):
            result = opcls(
                get_mlir_op_result(result),
                get_mlir_op_result(stmts[i]),
                ip=ctx.get_ip(),
            )
        return result

    @staticmethod
    def build_IfExp(ctx: ASTContext, node: ast.IfExp):
        cond = build_stmt(ctx, node.test)
        true_val = build_stmt(ctx, node.body)
        false_val = build_stmt(ctx, node.orelse)
        true_val = ASTTransformer.build_cast_op(
            ctx, true_val, node.body.dtype, node.dtype
        )
        false_val = ASTTransformer.build_cast_op(
            ctx, false_val, node.orelse.dtype, node.dtype
        )
        return arith_d.SelectOp(
            get_mlir_op_result(cond),
            get_mlir_op_result(true_val),
            get_mlir_op_result(false_val),
            ip=ctx.get_ip(),
        )

    @staticmethod
    def build_If(ctx: ASTContext, node: ast.If, is_affine: bool = False):
        if is_affine:
            # Should build the condition on-the-fly
            cond, var = build_stmt(ctx, node.test)
            if_op = affine_d.AffineIfOp(
                cond,
                [get_mlir_op_result(var)],
                ip=ctx.get_ip(),
                hasElse=len(node.orelse),
            )
            # TODO: MLIR bug, need to create a then_block function
            then_block = if_op.thenRegion.blocks[0]
        else:
            cond = build_stmt(ctx, node.test)
            if_op = scf_d.IfOp(
                get_mlir_op_result(cond), ip=ctx.get_ip(), hasElse=len(node.orelse)
            )
            then_block = if_op.then_block
        ctx.set_ip(then_block)
        with ctx.block_scope_guard():
            build_stmts(ctx, node.body)
            if is_affine:
                affine_d.AffineYieldOp([], ip=ctx.get_ip())
            else:
                scf_d.YieldOp([], ip=ctx.get_ip())
        ctx.pop_ip()
        if len(node.orelse) > 0:
            else_block = if_op.elseRegion.blocks[0]
            ctx.set_ip(else_block)
            with ctx.block_scope_guard():
                build_stmts(ctx, node.orelse)
                if is_affine:
                    affine_d.AffineYieldOp([], ip=ctx.get_ip())
                else:
                    scf_d.YieldOp([], ip=ctx.get_ip())
            ctx.pop_ip()

    @staticmethod
    def build_While(ctx: ASTContext, node: ast.While):
        """
        Example: https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfwhile-scfwhileop
        %res = scf.while (%arg1 = %init1) : (f32) -> f32 {
            // "Before" region.
            // In a "while" loop, this region computes the condition.
            %condition = call @evaluate_condition(%arg1) : (f32) -> i1
            // Forward the argument (as result or "after" region argument).
            scf.condition(%condition) %arg1 : f32
        } do {
        ^bb0(%arg2: f32):
            // "After" region.
            // In a "while" loop, this region is the loop body.
            %next = call @payload(%arg2) : (f32) -> f32
            // Forward the new value to the "before" region.
            // The operand types must match the types of the `scf.while` operands.
            scf.yield %next : f32
        }
        """
        while_op = scf_d.WhileOp([], [], ip=ctx.get_ip())
        while_op.before.blocks.append(*[])
        while_op.after.blocks.append(*[])
        with ctx.loop_scope_guard():
            ctx.set_ip(while_op.before.blocks[0])
            cond = build_stmt(ctx, node.test)
            scf_d.ConditionOp(get_mlir_op_result(cond), [], ip=ctx.get_ip())
            ctx.pop_ip()
            ctx.set_ip(while_op.after.blocks[0])
            with ctx.block_scope_guard():
                build_stmts(ctx, node.body)
                scf_d.YieldOp([], ip=ctx.get_ip())
            ctx.pop_ip()
        return while_op

    @staticmethod
    def build_Module(ctx: ASTContext, node: ast.Module):
        ASTTransformer.global_var_cnt.clear()
        with ctx.mlir_ctx:
            module = Module.create()
        ctx.set_ip(module.body)
        for stmt in node.body:
            build_stmt(ctx, stmt)
        ctx.pop_ip()
        return module

    @staticmethod
    def get_stream_name(ctx: ASTContext, node: ast.expr):
        vid = node.id if isinstance(node, ast.Name) else node.value.id
        symbolic_slice = None
        iterator_infos = {}
        if isinstance(node, ast.Subscript):
            # pylint: disable=redefined-builtin
            slice = eval(ast.unparse(node.slice), ctx.global_vars)
            symbolic_slice, iterator_info = get_symbolic_expr(
                copy.deepcopy(node.slice),
                ctx.symbolic,
                ctx.global_vars,
                ctx.get_alive_var_names(),
            )
            for info in iterator_info:
                iterator_infos[info[0]] = ArrayAttr.get(
                    [IntegerAttr.get(IntegerType.get_signless(64), v) for v in info[1]]
                )
            if isinstance(slice, int):
                slice = tuple([slice])
            else:
                slice = tuple(slice) if not isinstance(slice, tuple) else slice
            # access a specific stream
            slice_str = "_".join([str(x) for x in slice])
            new_name = f"{vid}_{slice_str}"
        else:
            slice = tuple()
            new_name = vid
        return new_name, symbolic_slice, iterator_infos

    # pylint: disable=too-many-return-statements, too-many-function-args, too-many-nested-blocks
    @staticmethod
    def build_Call(ctx: ASTContext, node: ast.Call, out_buffer: OpView = None):
        original_func_id = ctx.func_id
        if isinstance(node.func, ast.Name):
            obj = ASTResolver.resolve(node.func, ctx.global_vars)
            obj_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            obj = ASTResolver.resolve(node.func, ctx.global_vars)
            obj_name = node.func.attr
        elif isinstance(node.func, ast.Subscript):
            obj = ASTResolver.resolve(node.func.value, ctx.global_vars)
            assert obj is not None, "Unsupported function call"
            obj_name = (
                node.func.value.id if isinstance(obj, func_d.FuncOp) else obj.__name__
            )
            ctx.global_vars[obj_name] = obj
            ctx.inst = ASTResolver.resolve_param_types(node.func.slice, ctx.global_vars)
            if ctx.func_id is None:
                func_id = get_func_id_from_param_types(ctx.inst)
                if func_id is None:
                    func_dict = ctx.func_name2id.setdefault(obj_name, {})
                    for key, value in func_dict.items():
                        if value == tuple(ctx.inst):
                            func_id = key
                            break
                    else:
                        func_id = len(func_dict) if len(func_dict) > 0 else None
                        func_dict[func_id] = tuple(ctx.inst)
                else:
                    ctx.inst.remove(func_id)
                    func_dict = ctx.func_name2id.setdefault(obj_name, {})
                    func_dict[func_id] = tuple(ctx.inst)
                ctx.func_id = func_id
        else:
            raise RuntimeError("Unsupported function call")

        if obj is None:
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in {"T", "reverse"}:
                    # x.T or x.reverse
                    return build_stmt(ctx, node.func)
                if node.func.attr == "put":
                    stmts = build_stmts(ctx, node.args)
                    assert len(stmts) == 1, "Stream can only have one argument"
                    new_name, symbolic_slice, iterator_infos = (
                        ASTTransformer.get_stream_name(ctx, node.func.value)
                    )
                    stream = ctx.get_symbol(new_name).clone(
                        ip=ctx.get_stream_construct_ip()
                    )
                    if symbolic_slice is not None:
                        stream.attributes["symbolic_slice"] = StringAttr.get(
                            symbolic_slice
                        )
                        stream.attributes["iterators"] = DictAttr.get(iterator_infos)
                    put_op = allo_d.StreamPutOp(
                        stream.result,
                        [],
                        get_mlir_op_result(stmts[0]),
                        ip=ctx.get_ip(),
                    )
                    if isinstance(node.func.value.dtype, UInt):
                        put_op.attributes["unsigned"] = UnitAttr.get()
                    return
                if node.func.attr == "get":
                    new_name, symbolic_slice, iterator_infos = (
                        ASTTransformer.get_stream_name(ctx, node.func.value)
                    )
                    # insert after the last stream construct op to preserve ordering
                    stream = ctx.get_symbol(new_name).clone(
                        ip=ctx.get_stream_construct_ip()
                    )
                    if symbolic_slice is not None:
                        stream.attributes["symbolic_slice"] = StringAttr.get(
                            symbolic_slice
                        )
                        stream.attributes["iterators"] = DictAttr.get(iterator_infos)
                    get_op = allo_d.StreamGetOp(
                        node.func.value.dtype.build(),
                        stream.result,
                        [],
                        ip=ctx.get_ip(),
                    )
                    if isinstance(node.func.value.dtype.dtype, UInt):
                        get_op.attributes["unsigned"] = UnitAttr.get()
                    return get_op
                if node.func.attr == "bitcast":
                    val = build_stmt(ctx, node.func.value)
                    op = arith_d.BitcastOp(
                        node.dtype.build(),
                        get_mlir_op_result(val),
                        ip=ctx.get_ip(),
                    )
                    if isinstance(node.func.value.dtype, UInt) or (node.dtype, UInt):
                        op.attributes["unsigned"] = UnitAttr.get()
                    return op

            if node.func.id in {"float", "int"}:
                # Python-Builtin functions
                stmts = build_stmts(ctx, node.args)
                return ASTTransformer.build_cast_op(
                    ctx,
                    stmts[0],
                    node.args[0].dtype,
                    Int(32) if node.func.id == "int" else float32,
                )

            if node.func.id in {"min", "max"}:
                stmts = build_stmts(ctx, node.args)
                if isinstance(node.dtype, Float):
                    opcls = {
                        "min": arith_d.MinimumFOp,
                        "max": arith_d.MaximumFOp,
                    }.get(node.func.id)
                elif isinstance(node.dtype, Int):
                    opcls = {
                        "min": arith_d.MinSIOp,
                        "max": arith_d.MaxSIOp,
                    }.get(node.func.id)
                elif isinstance(node.dtype, UInt):
                    opcls = {
                        "min": arith_d.MinUIOp,
                        "max": arith_d.MaxUIOp,
                    }.get(node.func.id)
                lhs = ASTTransformer.build_cast_op(
                    ctx, stmts[0], node.args[0].dtype, node.dtype
                )
                rhs = ASTTransformer.build_cast_op(
                    ctx, stmts[1], node.args[1].dtype, node.dtype
                )
                return opcls(
                    get_mlir_op_result(lhs), get_mlir_op_result(rhs), ip=ctx.get_ip()
                )
            raise RuntimeError(f"Cannot resolve function `{node.func.id}`")

        try:
            from ..backend.aie.vliw import VLIWKernelFunction
        except ImportError:

            class VLIWKernelFunction:
                pass

        if isinstance(obj, VLIWKernelFunction):
            # Get the external module from the VLIW function
            external_module = obj.get_external_module()
            # Add the external module to ext_libs for processing
            if external_module not in ctx.ext_libs:
                ctx.ext_libs.append(external_module)
                # Create function declaration similar to other external modules
                input_types = []
                for arg_type, shape in external_module.args:
                    ele_type = get_mlir_dtype_from_str(c2allo_type[arg_type])
                    if len(shape) != 0:
                        memref = MemRefType.get(shape, ele_type)
                    else:
                        memref = ele_type
                    input_types.append(memref)
                func_type = FunctionType.get(input_types, [])
                func_op = func_d.FuncOp(
                    name=external_module.top,
                    type=func_type,
                    ip=InsertionPoint(ctx.top_func),
                )
                func_op.attributes["sym_visibility"] = StringAttr.get("private")
            # Build arguments and create call
            new_args = build_stmts(ctx, node.args)
            call_op = func_d.CallOp(
                [],
                FlatSymbolRefAttr.get(external_module.top),
                [get_mlir_op_result(arg) for arg in new_args],
                ip=ctx.get_ip(),
            )
            return call_op

        if (
            obj.__module__.startswith("allo")
            and not obj.__module__.startswith("allo.library")
            and not obj.__module__.startswith("allo._mlir")
        ):
            # Local imports to avoid cyclic dependencies
            from ..backend.ip import IPModule

            try:
                from ..backend.aie.external_kernel import ExternalModule
            except ImportError:

                class ExternalModule:
                    pass

            fn_name = (
                obj.__name__
                if not isinstance(obj, (IPModule, ExternalModule))
                else None
            )
            if fn_name in {"gather", "scatter"}:
                fifo_list = node.args[0] if fn_name == "gather" else node.args[1]
                ip = ctx.get_stream_construct_ip()
                stream_type = allo_d.StreamType.get(
                    fifo_list.dtype.build(), depth=fifo_list.dtype.depth
                )
                if not ctx.unroll:
                    stream_op = allo_d.StreamConstructOp(stream_type, ip=ip)
                    fifo_name_attrs = []
                    fifo_symbolic_slice_attrs = []
                else:
                    stream_ops = []
                if isinstance(fifo_list, ast.List):
                    for elt in fifo_list.elts:
                        fifo_name, symbolic_slice, iterator_infos = (
                            ASTTransformer.get_stream_name(ctx, elt)
                        )
                        assert len(iterator_infos) == 0, "Not supported yet"
                        if not ctx.unroll:
                            fifo_name_attrs.append(StringAttr.get(fifo_name))
                            fifo_symbolic_slice_attrs.append(
                                StringAttr.get(symbolic_slice)
                            )
                        else:  # Unroll
                            stream_op = allo_d.StreamConstructOp(stream_type, ip=ip)
                            stream_op.attributes["name"] = StringAttr.get(fifo_name)
                            stream_ops.append(stream_op)
                elif isinstance(fifo_list, ast.Subscript):
                    # if len(static_sizes) may > 1, will be flattened
                    array_shape = ctx.get_symbol(fifo_list.value.id)
                    if isinstance(fifo_list.slice, ast.Tuple):
                        slices = list(fifo_list.slice.elts)
                    else:
                        slices = (
                            fifo_list.slice.dims
                            if len(fifo_list.shape) > 1
                            else [fifo_list.slice]
                        )
                    # offset, size, stride for accessing the stream array
                    offsets = []
                    static_sizes = []
                    static_strides = []
                    for index, size in zip(slices, array_shape):
                        if isinstance(index, ast.Slice):
                            # fixme: slice lower/upper/step is not constant
                            lower = (
                                0
                                if index.lower is None
                                else ASTResolver.resolve_constant(index.lower, ctx)
                            )
                            upper = (
                                size
                                if index.upper is None
                                else ASTResolver.resolve_constant(index.upper, ctx)
                            )
                            if index.step is None:
                                step = 1
                            elif isinstance(index.step, ast.Constant):
                                step = index.step.value
                            else:
                                raise RuntimeError("Unsupported step type")
                            assert lower is not None and upper is not None
                            assert (
                                0 <= lower < size and 0 <= upper <= size and step > 0
                            ), "Invalid index or step"
                            offsets.append(lower)
                            static_sizes.append((upper - lower) // step)
                            static_strides.append(step)
                        else:
                            ind = eval(ast.unparse(index), ctx.global_vars)
                            symbolic_ind, iterator_info = get_symbolic_expr(
                                copy.deepcopy(index),
                                ctx.symbolic,
                                ctx.global_vars,
                                ctx.get_alive_var_names(),
                            )
                            offsets.append((ind, symbolic_ind, iterator_info))
                            static_sizes.append(1)
                            static_strides.append(1)
                            raise RuntimeError("TODO")
                    assert all(
                        isinstance(offset, int) for offset in offsets
                    ), "Not supported yet"
                    # flattened here
                    for indices in itertools.product(
                        *[
                            range(o, o + s * st, st)
                            for o, s, st in zip(offsets, static_sizes, static_strides)
                        ]
                    ):
                        slice_str = "_".join([str(x) for x in indices])
                        fifo_name = f"{fifo_list.value.id}_{slice_str}"
                        if not ctx.unroll:
                            symbolic_slice = ",".join([str(x) for x in indices])
                            fifo_name_attrs.append(StringAttr.get(fifo_name))
                            fifo_symbolic_slice_attrs.append(
                                StringAttr.get(symbolic_slice)
                            )  # FIXME: unverfied
                        else:  # Unroll
                            stream_op = allo_d.StreamConstructOp(stream_type, ip=ip)
                            stream_op.attributes["name"] = StringAttr.get(fifo_name)
                            stream_ops.append(stream_op)
                else:
                    raise RuntimeError(f"Fail to resolve fifo_list for {fn_name}.")
                if not ctx.unroll:
                    stream_op.attributes["name"] = fifo_name_attrs[0]
                    if len(fifo_symbolic_slice_attrs) > 0:
                        stream_op.attributes["symbolic_slice"] = (
                            fifo_symbolic_slice_attrs[0]
                        )
                    stream_op.attributes["stream_list"] = ArrayAttr.get(fifo_name_attrs)
                    stream_op.attributes["stream_symbolic_slice_list"] = ArrayAttr.get(
                        fifo_symbolic_slice_attrs
                    )
                if fn_name == "gather":
                    # allocate local buffer for the gathered results
                    with ctx.get_ip():
                        alloc_op = ASTTransformer.build_array(
                            ctx, node.dtype, node.shape
                        )
                else:
                    buffer = build_stmt(ctx, node.args[0])
                # internally build gather loop
                with ctx.loop_scope_guard():
                    if not ctx.unroll:
                        lb_expr = arith_d.ConstantOp(
                            arith_d.IndexType.get(), 0, ip=ctx.get_ip()
                        )
                        ub_expr = arith_d.ConstantOp(
                            arith_d.IndexType.get(),
                            len(fifo_name_attrs),
                            ip=ctx.get_ip(),
                        )
                        step = arith_d.ConstantOp(
                            arith_d.IndexType.get(), 1, ip=ctx.get_ip()
                        )
                        for_op = scf_d.ForOp(
                            lb_expr.result, ub_expr.result, step.result, ip=ctx.get_ip()
                        )
                        op_name = f"S_{fn_name}_{str(ctx.loop_band_count)}"
                        for_op.attributes["op_name"] = StringAttr.get(op_name)
                        loop_iter_name = f"{fn_name}{str(ctx.loop_band_count)}"
                        for_op.attributes["loop_name"] = StringAttr.get(loop_iter_name)
                        stream_op.attributes["loop_name"] = StringAttr.get(op_name)
                        scf_d.YieldOp([], ip=InsertionPoint(for_op.body))
                        ip = InsertionPoint(for_op.body.operations[0])
                        num_iterations = 1
                    else:  # Unroll, create an scf.execute_region operation
                        indices = []  # Static indices
                        for i in range(len(stream_ops)):
                            index_const_expr = arith_d.ConstantOp(
                                arith_d.IndexType.get(), i, ip=ctx.get_ip()
                            )
                            indices.append(index_const_expr)
                        ip = ctx.get_ip()
                        num_iterations = len(stream_ops)
                    with ip:
                        static_offsets = []
                        size_values = []
                        stride_values = []
                        result_sizes = []
                        result_strides = []
                        tile_size = np.prod(node.shape)
                        for i, shape in enumerate(node.shape):
                            tile_size //= shape
                            stride_values.append(1)
                            if i == 0:
                                static_offsets.append(ShapedType.get_dynamic_size())
                                size_values.append(1)
                            else:
                                static_offsets.append(0)
                                size_values.append(shape)
                                result_sizes.append(shape)
                                result_strides.append(tile_size)
                        subview_result = MemRefType.get(
                            shape=result_sizes,
                            element_type=node.dtype.build(),
                            layout=StridedLayoutAttr.get(
                                offset=ShapedType.get_dynamic_stride_or_offset(),
                                strides=result_strides,
                            ),
                        )
                        for iteration in range(num_iterations):
                            offset_values = (
                                [indices[iteration]]
                                if ctx.unroll
                                else [for_op.induction_variable]
                            )
                            stream_op = (
                                stream_ops[iteration] if ctx.unroll else stream_op
                            )
                            if fn_name == "gather":
                                get_op = allo_d.StreamGetOp(
                                    fifo_list.dtype.build(), stream_op.result, []
                                )
                                subview = memref_d.SubViewOp(
                                    source=alloc_op.result,
                                    result=subview_result,
                                    static_offsets=static_offsets,
                                    static_sizes=size_values,
                                    static_strides=stride_values,
                                    offsets=offset_values,
                                    sizes=[],
                                    strides=[],
                                )
                                # copy to slice
                                memref_d.CopyOp(get_op.result, subview.result)
                            else:
                                subview = memref_d.SubViewOp(
                                    source=get_mlir_op_result(buffer),
                                    result=subview_result,
                                    static_offsets=static_offsets,
                                    static_sizes=size_values,
                                    static_strides=stride_values,
                                    offsets=offset_values,
                                    sizes=[],
                                    strides=[],
                                )
                                allo_d.StreamPutOp(stream_op.result, [], subview.result)
                    if fn_name == "gather":
                        return alloc_op
                    return for_op if not ctx.unroll else None
            # Allo library functions
            new_args = build_stmts(ctx, node.args)
            if isinstance(obj, (IPModule, ExternalModule)):
                input_idx = obj.input_idx if isinstance(obj, ExternalModule) else None
                input_types = []
                call_operands = []
                for idx, (arg_type, shape) in enumerate(obj.args):
                    ele_type = get_mlir_dtype_from_str(c2allo_type[arg_type])
                    if len(shape) != 0:
                        memref = MemRefType.get(shape, ele_type)
                    else:
                        memref = ele_type
                    input_types.append(memref)
                    operand = new_args[idx]
                    if get_mlir_op_result(operand).type != memref:
                        assert (
                            input_idx is not None
                        ), "IPModule is not well supported yet."
                        alloc_op = memref_d.AllocOp(memref, [], [], ip=ctx.get_ip())
                        call_operands.append(alloc_op.result)
                        if idx in input_idx:
                            memref_d.CopyOp(operand, alloc_op.result, ip=ctx.get_ip())
                    else:
                        call_operands.append(get_mlir_op_result(operand))
                # Add HLS IP as external library
                if obj not in ctx.ext_libs:
                    ctx.ext_libs.append(obj)
                    # Suppose it does not have any return values
                    func_type = FunctionType.get(input_types, [])
                    func_op = func_d.FuncOp(
                        name=obj.top, type=func_type, ip=InsertionPoint(ctx.top_func)
                    )
                    func_op.attributes["sym_visibility"] = StringAttr.get("private")
                call_op = func_d.CallOp(
                    [],
                    FlatSymbolRefAttr.get(obj.top),
                    call_operands,
                    ip=ctx.get_ip(),
                )
                for idx, (call_operand, operand_op) in enumerate(
                    zip(call_operands, new_args)
                ):
                    if input_idx is not None and idx not in input_idx:
                        operand_op_result = get_mlir_op_result(operand_op)
                        if call_operand != operand_op_result:
                            memref_d.CopyOp(
                                call_operand, operand_op_result, ip=ctx.get_ip()
                            )
                return
            if fn_name in {"zeros", "ones"}:
                shape = node.shape
                dtype = node.dtype
                with ctx.get_ip():
                    alloc_op = ASTTransformer.build_array(ctx, dtype, shape)
                    res = (
                        MockConstant(1, ctx)
                        if fn_name == "ones"
                        else MockConstant(0, ctx)
                    )
                    op = ASTTransformer.build_cast_op(ctx, res, Int(32), node.dtype)
                    op = linalg_d.fill(get_mlir_op_result(op), outs=[alloc_op.result])
                    return op.owner if ctx.enable_tensor else alloc_op
            if fn_name == "get_pid":
                res = []
                for i in range(3):
                    if f"df.p{i}" in ctx.global_vars:
                        res.append(
                            MockConstant(
                                ctx.global_vars[f"df.p{i}"], ctx, dtype=Index()
                            )
                        )
                return tuple(res)
            arg_types = []
            if isinstance(new_args[0].result, OpResultList):
                for arg in new_args:
                    if hasattr(arg, "result") and isinstance(arg.result, OpResultList):
                        for result in arg.result:
                            if hasattr(result, "type"):
                                arg_types.append(result.type)
            else:
                for arg in new_args:
                    if hasattr(arg, "result") and hasattr(arg.result, "type"):
                        arg_types.append(arg.result.type)
            if all(
                isinstance(arg_type, (F32Type, IntegerType)) for arg_type in arg_types
            ):
                opcls = {
                    "exp": math_d.ExpOp,
                    "log": math_d.LogOp,
                    "log2": math_d.Log2Op,
                    "log10": math_d.Log10Op,
                    "sqrt": math_d.SqrtOp,
                    "cos": math_d.CosOp,
                    "tan": math_d.TanOp,
                    "tanh": math_d.TanhOp,
                    "power": math_d.PowFOp,
                    "abs": math_d.AbsIOp,
                }.get(fn_name)
                return opcls(
                    *[get_mlir_op_result(x) for x in new_args], ip=ctx.get_ip()
                )
            if any(
                isinstance(arg_type, (MemRefType, RankedTensorType))
                for arg_type in arg_types
            ) and fn_name in {
                "matmul",
                "bmm",
                "softmax",
                "exp",
                "abs",
                "log",
                "add",
                "sub",
                "mul",
                "div",
                "relu",
                "conv2d",
                "maxpool",
                "sumpool",
                "copy",
                "transpose",
                "linear",
                "view",
                "concat",
            }:
                if fn_name in {"add", "sub", "mul", "div"}:
                    new_args[0] = ASTTransformer.build_broadcast_op(
                        ctx,
                        new_args[0],
                        node.dtype,
                        node.args[0].shape,
                        node.shape,
                        node.dims[0],
                    )
                    new_args[1] = ASTTransformer.build_broadcast_op(
                        ctx,
                        new_args[1],
                        node.dtype,
                        node.args[1].shape,
                        node.shape,
                        node.dims[1],
                    )
                return ASTTransformer.build_library_op(
                    ctx,
                    node=node,
                    attr=fn_name,
                    new_args=new_args,
                    out_buffer=out_buffer,
                )
            if fn_name in {"layernorm", "gelu", "tril"}:
                arg_results = [get_mlir_op_result(arg) for arg in new_args]
                input_types = [arg.type for arg in arg_results]
                output_types = [input_types[0]]
                func_op = func_d.FuncOp(
                    name=f"{fn_name}_{abs(hash(node))}",
                    type=FunctionType.get(input_types, output_types),
                    ip=InsertionPoint(ctx.top_func),
                )
                func_op.attributes["sym_visibility"] = StringAttr.get("private")
                call_op = func_d.CallOp(
                    [arg_results[0].type],
                    FlatSymbolRefAttr.get(f"{fn_name}_{abs(hash(node))}"),
                    arg_results,
                    ip=ctx.get_ip(),
                )
                return call_op
            raise RuntimeError(f"Unsupported function {fn_name} with type {arg_types}")

        # User-defined subfunction
        func = ctx.global_vars[obj_name]
        new_args = [get_mlir_op_result(stmt) for stmt in build_stmts(ctx, node.args)]
        func_name = obj_name if ctx.func_id is None else f"{obj_name}_{ctx.func_id}"
        if func_name not in ctx.global_vars or not isinstance(
            ctx.global_vars[func_name], func_d.FuncOp
        ):  # function not built yet
            # Create a new context to avoid name collision
            func_ctx = ctx.copy()
            func_ctx.call_args = new_args
            func_ctx.set_ip(ctx.top_func)
            stmts = build_stmts(func_ctx, node.tree.body)
            func_ctx.pop_ip()
            func_ctx.call_args = []
            for key, value in func_ctx.global_vars.items():
                if isinstance(value, func_d.FuncOp):
                    ctx.global_vars[key] = value
            # Attach buffers to function
            # FIXME: Should create subschedule
            for name, buffer in func_ctx.buffers.items():
                if isinstance(buffer, (memref_d.AllocOp, MockArg)):
                    # Intermediate buffers and function arguments
                    setattr(func, name, MockBuffer(func_name, name))
        elif isinstance(func, func_d.FuncOp):
            # Has already been defined in the top-level scope
            stmts = [func]
        else:
            stmts = [ctx.global_vars[func_name]]

        # Build call function in the top-level
        call_op = func_d.CallOp(
            stmts[-1].type.results,
            FlatSymbolRefAttr.get(func_name),
            new_args,
            ip=ctx.get_ip(),
        )
        ctx.func_id = original_func_id
        return call_op

    @staticmethod
    def build_library_op(
        ctx: ASTContext,
        node: ast.AST,
        attr: str,
        new_args: list[OpView],
        dtype: AlloType = None,
        shape: list[int] = None,
        out_buffer: OpView = None,
    ):
        assert attr is not None and attr != ""
        ip = ctx.get_ip()
        dtype = dtype if dtype is not None else node.dtype
        shape = shape if shape is not None else node.shape
        buf_op = (
            out_buffer
            if out_buffer is not None
            else ASTTransformer.build_array(ctx, dtype, shape)
        )
        with ip:
            if attr == "concat":
                axis = node.keywords[0].value.value
                strides = [1] * len(shape)
                offsets = [0] * len(shape)
                new_offsets = offsets.copy()
                new_offsets[axis] = node.args[0].shape[axis]
                if ctx.enable_tensor:
                    insert_op = tensor_d.InsertSliceOp(
                        source=get_mlir_op_result(new_args[0]),
                        dest=get_mlir_op_result(buf_op),
                        static_offsets=offsets,
                        static_sizes=list(node.args[0].shape),
                        static_strides=strides,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        ip=ctx.get_ip(),
                    )
                    # concanate the second tensor
                    concat_op = tensor_d.InsertSliceOp(
                        source=get_mlir_op_result(new_args[1]),
                        dest=insert_op.result,
                        static_offsets=new_offsets,
                        static_sizes=list(node.args[1].shape),
                        static_strides=strides,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        ip=ctx.get_ip(),
                    )
                    return concat_op

                concat_shape = [node.args[0].shape, node.args[1].shape]
                memref_strides = []
                product = 1
                memref_strides = []
                for size in reversed(shape):
                    memref_strides.append(product)
                    product *= size
                memref_strides.reverse()
                memref_offsets = 1
                for size in concat_shape[0][axis:]:
                    memref_offsets *= size
                result0 = MemRefType.get(
                    shape=concat_shape[0],
                    element_type=dtype.build(),
                    layout=StridedLayoutAttr.get(offset=0, strides=memref_strides),
                )
                op_ = memref_d.SubViewOp(
                    source=buf_op,
                    result=result0,
                    static_offsets=offsets,
                    static_sizes=list(node.args[0].shape),
                    static_strides=strides,
                    offsets=[],
                    sizes=[],
                    strides=[],
                    ip=ctx.get_ip(),
                )
                memref_d.CopyOp(
                    get_mlir_op_result(new_args[0]),
                    op_,
                    ip=ctx.get_ip(),
                )
                result1 = MemRefType.get(
                    shape=concat_shape[1],
                    element_type=dtype.build(),
                    layout=StridedLayoutAttr.get(
                        offset=memref_offsets, strides=memref_strides
                    ),
                )
                view_op = memref_d.SubViewOp(
                    source=buf_op,
                    result=result1,
                    static_offsets=new_offsets,
                    static_sizes=list(node.args[1].shape),
                    static_strides=strides,
                    offsets=[],
                    sizes=[],
                    strides=[],
                    ip=ctx.get_ip(),
                )
                memref_d.CopyOp(
                    get_mlir_op_result(new_args[1]),
                    view_op,
                    ip=ctx.get_ip(),
                )
                return buf_op
            if attr in {
                "matmul",
                "bmm",
                "conv2d",
                "maxpool",
                "sumpool",
                "relu",
            }:
                # init zero
                zero = MockConstant(0, ctx)
                zero = ASTTransformer.build_cast_op(ctx, zero, Int(32), node.dtype)
                linalg_fill = linalg_d.fill(
                    zero.result, outs=[get_mlir_op_result(buf_op)]
                )
                result_tensor = linalg_fill if ctx.enable_tensor else buf_op
                ASTTransformer.attach_op_name(
                    ctx,
                    node,
                    linalg_fill.owner,
                    f"{attr}_init_zero",
                    postfix="init_zero",
                )
            else:
                result_tensor = buf_op
            # build linalg op
            if attr in {
                "matmul",
                "bmm",
                "add",
                "sub",
                "mul",
                "div",
                "conv2d",
                "maxpool",
                "sumpool",
            }:
                # Since MLIR does not natively support matrix multiplication for matrices with more than 2 dimensions,
                # we must first flatten the leading dimensions, utilize bmm for computation, and subsequently restore the original shape.
                if len(shape) > 3 and attr == "matmul":
                    flattened_shapes = ASTTransformer.build_flattened_shapes(
                        node, new_args
                    )
                    for i, arg in enumerate(new_args):
                        new_args[i] = ASTTransformer.build_library_op(
                            ctx,
                            node,
                            "view",
                            [arg, flattened_shapes[i]],
                            shape=flattened_shapes[i],
                        )
                    inner_shape = flattened_shapes[0][:-1] + flattened_shapes[1][-1:]
                    op = ASTTransformer.build_library_op(
                        ctx,
                        node,
                        "bmm",
                        new_args,
                        dtype,
                        inner_shape,
                    )
                    if tuple(inner_shape) != shape:
                        op = ASTTransformer.build_library_op(
                            ctx,
                            node,
                            "view",
                            [op, shape],
                            shape=shape,
                        )
                    return op
                op = {
                    "matmul": linalg_d.matmul,
                    "bmm": linalg_d.batch_matmul,
                    "add": linalg_d.add,
                    "sub": linalg_d.sub,
                    "mul": linalg_d.mul,
                    "div": linalg_d.div,
                    "conv2d": linalg_d.conv_2d_nchw_fchw,
                    "maxpool": linalg_d.pooling_nchw_max,
                    "sumpool": linalg_d.pooling_nchw_sum,
                }.get(attr)(
                    get_mlir_op_result(new_args[0]),
                    get_mlir_op_result(new_args[1]),
                    outs=[
                        (
                            get_mlir_op_result(result_tensor)
                            if hasattr(result_tensor, "result")
                            else result_tensor
                        )
                    ],
                )
                op = op.owner
            elif attr in {"exp", "log", "abs", "copy"}:
                op = {
                    "exp": linalg_d.exp,
                    "log": linalg_d.log,
                    "abs": linalg_d.abs,
                    "copy": linalg_d.copy,
                }.get(attr)(
                    get_mlir_op_result(new_args[0]),
                    outs=[
                        (
                            get_mlir_op_result(result_tensor)
                            if hasattr(result_tensor, "result")
                            else result_tensor
                        )
                    ],
                )
                op = op.owner
            elif attr == "softmax":
                # TODO: Failed to lower to LLVM, see https://reviews.llvm.org/D153422
                # We temporarily replace SoftmaxOp with a predefined lowered function to enable LLVM execution
                op = linalg_d.SoftmaxOp(
                    input=get_mlir_op_result(new_args[0]),
                    dimension=0,
                    result=[],
                    output=(
                        result_tensor
                        if ctx.enable_tensor
                        else get_mlir_op_result(result_tensor)
                    ),
                )
            elif attr == "relu":
                if ctx.enable_tensor:
                    # TODO: Need to better manage library call
                    zero_op = ASTTransformer.build_array(ctx, dtype, shape)
                    # init zero
                    zero = MockConstant(0, ctx)
                    # TODO: support tensor
                    linalg_fill = linalg_d.fill(zero.result, outs=[zero_op.result])
                    op = linalg_d.max(
                        get_mlir_op_result(new_args[0]),
                        zero_op.result,
                        outs=[result_tensor],
                    )
                    op = op.owner
                else:
                    op = linalg_d.max(
                        get_mlir_op_result(new_args[0]),
                        (
                            result_tensor
                            if ctx.enable_tensor
                            else get_mlir_op_result(result_tensor)
                        ),
                        outs=[result_tensor],
                    )
                    op = op.owner
            elif attr == "transpose":
                with ctx.get_ip():
                    op = linalg_d.transpose(
                        input=get_mlir_op_result(new_args[0]),
                        outs=[get_mlir_op_result(result_tensor)],
                        permutation=tuple(x.val for x in new_args[1]),
                    )
            elif attr == "view":
                view_op = (
                    tensor_d.ReshapeOp if ctx.enable_tensor else memref_d.ReshapeOp
                )
                # When the input shape is a list of constants, we can get the values directly
                if MockConstant not in [type(x) for x in new_args[1]]:
                    value = np.array(new_args[1])
                else:
                    value = np.array(tuple(x.val for x in new_args[1]))
                shape_value = ASTTransformer.build_constant_tensor(
                    ctx,
                    node,
                    np_values=value,
                    dtype=Int(64),
                    shape=value.shape,
                    constant=True,
                )
                shaped_type = ASTTransformer.build_shaped_type(ctx, dtype, shape)
                op = view_op(
                    source=get_mlir_op_result(new_args[0]),
                    result=shaped_type,
                    shape=shape_value.result,
                    ip=ctx.get_ip(),
                )
                return op
            elif attr == "linear":  # X @ A.T + B
                permutation = [MockConstant(val, ctx) for val in (1, 0)]
                A_T = ASTTransformer.build_library_op(
                    ctx,
                    node,
                    "transpose",
                    [new_args[1], permutation],
                    shape=node.args[1].shape[::-1],
                )
                inner_attr = "matmul"
                if len(shape) >= 3:
                    flattened_shapes = ASTTransformer.build_flattened_shapes(
                        node, new_args
                    )
                    A_T = ASTTransformer.build_broadcast_op(
                        ctx,
                        A_T,
                        dtype,
                        list(node.args[1].shape[::-1]),
                        flattened_shapes[1],
                        [0],
                    )
                    inner_attr = "bmm"
                matmul = ASTTransformer.build_library_op(
                    ctx, node, inner_attr, [new_args[0], A_T]
                )

                # bias = True
                if len(new_args) == 3 and (
                    not isinstance(node.args[2], ast.Constant)
                    or node.args[2].value is not None
                ):
                    dims = list(range(len(node.shape) - 1))
                    bias = ASTTransformer.build_broadcast_op(
                        ctx, new_args[2], node.dtype, node.shape[-1:], node.shape, dims
                    )
                    add = ASTTransformer.build_library_op(
                        ctx, node, "add", [matmul, bias]
                    )
                    return add
                # bias = False
                return matmul
            else:
                raise RuntimeError("Unsupported operation")
            ASTTransformer.attach_op_name(ctx, node, op, attr)
        return op if (ctx.enable_tensor or out_buffer is not None) else result_tensor

    @staticmethod
    def build_flattened_shapes(node: ast.AST, new_args: list[OpView]):
        # Only support:
        # 1. flatten two matrices that
        #    the last two dimensions must conform to two-dimensional matrix multiplication,
        #    while the shapes of the remaining dimensions should be identical.
        # 2. flatten A @ B.T when A is 3D, B is 2D, and the last dimension of A and B is same.
        flattened_shapes = []
        if node.func.attr == "matmul":
            for i in range(len(new_args)):
                flattened_shapes.append(
                    [
                        np.prod(node.args[i].shape[:-2], dtype=np.int64).tolist(),
                        node.args[i].shape[-2],
                        node.args[i].shape[-1],
                    ]
                )
        if node.func.attr == "linear":
            flattened_shapes.append(node.args[0].shape)
            flattened_shapes.append(
                [flattened_shapes[0][0]] + list(node.args[1].shape[::-1])
            )
        return flattened_shapes

    @staticmethod
    def build_Return(ctx: ASTContext, node: ast.Return):
        ctx.has_return = True
        if node.value is None or (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            if ctx.top_func_tree.dtype is not None:
                raise RuntimeError("Mismatch in function signature")
            return func_d.ReturnOp([], ip=ctx.pop_ip())
        if ctx.top_func_tree.dtype is None:
            raise RuntimeError("Return value should have a dtype in function signature")
        if isinstance(node.value, ast.Tuple):
            # return multiple values
            rets = []
            for i, n in enumerate(node.value.elts):
                ret = build_stmt(ctx, n)
                ret = ASTTransformer.build_cast_op(
                    ctx,
                    ret,
                    n.dtype,
                    ctx.top_func_tree.dtype[i],
                    ctx.top_func_tree.shape[i],
                )
                rets.append(get_mlir_op_result(ret))
            return func_d.ReturnOp(rets, ip=ctx.pop_ip())
        # return a single value or none
        ret = build_stmt(ctx, node.value)
        if ret is None:
            return func_d.ReturnOp([], ip=ctx.pop_ip())
        ret = ASTTransformer.build_cast_op(
            ctx, ret, node.dtype, ctx.top_func_tree.dtype, ctx.top_func_tree.shape
        )
        res = get_mlir_op_result(ret)
        if type(res.type) is type(ctx.top_func.type.results[0]):
            if (
                isinstance(res.type, MemRefType)
                and res.type.layout != ctx.top_func.type.results[0].layout
            ):
                # memref.subview is involved, we need to copy the values from the original buffer
                alloc_op = ASTTransformer.build_array(ctx, node.dtype, node.shape)
                memref_d.CopyOp(
                    res,
                    alloc_op.result,
                    ip=ctx.get_ip(),
                )
                ret = alloc_op
                res = ret.result
        else:
            raise RuntimeError("Fail to resolve return value")
        return func_d.ReturnOp([res], ip=ctx.pop_ip())

    @staticmethod
    def build_With(ctx: ASTContext, node: ast.With):
        # Compile-time comparison
        if node.items[0].context_expr.func.attr in {"meta_if", "meta_elif"}:
            cond = ASTResolver.resolve_constant(node.items[0].context_expr.args[0], ctx)
            if node.items[0].context_expr.func.attr == "meta_if":
                final_cond = cond
                if len(ctx.meta_if_stack) > ctx.with_scope_level:
                    ctx.meta_if_stack[ctx.with_scope_level].append(final_cond)
                else:
                    ctx.meta_if_stack.append([final_cond])
            else:  # meta_elif
                assert (
                    len(ctx.meta_if_stack[ctx.with_scope_level]) > 0
                ), "Unmatched allo.meta_elif()"
                if ctx.meta_if_stack[ctx.with_scope_level][
                    -1
                ]:  # previous `if` has already satisfied
                    ctx.meta_if_stack[ctx.with_scope_level].pop()
                    ctx.meta_if_stack[ctx.with_scope_level].append(True)
                    final_cond = False
                else:
                    ctx.meta_if_stack[ctx.with_scope_level].pop()
                    ctx.meta_if_stack[ctx.with_scope_level].append(cond)
                    final_cond = cond
        elif node.items[0].context_expr.func.attr == "meta_else":
            assert (
                len(ctx.meta_if_stack[ctx.with_scope_level]) > 0
            ), "Unmatched allo.meta_else()"
            final_cond = not ctx.meta_if_stack[ctx.with_scope_level][-1]
            ctx.meta_if_stack[ctx.with_scope_level].pop()
        elif node.items[0].context_expr.func.attr == "meta_for":
            assert (
                len(node.items[0].context_expr.args) <= 3
            ), "Only support three arguments (lower, upper bound, and step) for `allo.meta_for()`"
            var = node.items[0].optional_vars.id
            rargs = [
                ASTResolver.resolve_constant(node.items[0].context_expr.args[0], ctx)
            ]
            if len(node.items[0].context_expr.args) > 1:
                rargs.append(
                    ASTResolver.resolve_constant(
                        node.items[0].context_expr.args[1], ctx
                    )
                )
            if len(node.items[0].context_expr.args) > 2:
                rargs.append(
                    ASTResolver.resolve_constant(
                        node.items[0].context_expr.args[2], ctx
                    )
                )
            if ctx.unroll or node in ctx.meta_fors_to_unroll:
                for i in range(*rargs):
                    with ctx.block_scope_guard():
                        ctx.global_vars[var] = i
                        build_stmts(ctx, node.body)
                        ctx.global_vars.pop(var)
            else:
                # replace with regular for loop to reduce codesize
                with ctx.loop_scope_guard():
                    op_name = f"S_{var}_{str(ctx.loop_band_count)}"
                    for_op = ASTTransformer.build_single_for(
                        ctx, node.items[0].context_expr.args, op_name, var
                    )
                    ctx.global_vars[var] = 0 if len(rargs) == 1 else rargs[0]
                    ctx.symbolic[var] = (op_name, (op_name, tuple(rargs)))
                    is_affine = isinstance(for_op, affine_d.AffineForOp)
                    with ctx.block_scope_guard():
                        ctx.put_symbol(
                            name=var,
                            val=MockArg(for_op.induction_variable, is_affine),
                            tag="placeholder",
                        )
                        ctx.set_ip(for_op.body.operations[0])
                        build_stmts(ctx, node.body)
                    gc.collect()
                    ctx.pop_ip()
                    ctx.global_vars.pop(var)
            return
        else:
            raise RuntimeError("Unsupported meta function")
        if final_cond:
            ctx.with_scope_level += 1
            with ctx.block_scope_guard():
                stmts = build_stmts(ctx, node.body)
            # clear inner context
            ctx.meta_if_stack = ctx.meta_if_stack[: ctx.with_scope_level]
            ctx.with_scope_level -= 1
            return stmts[-1]
        return "WithStatementSkipped"

    @staticmethod
    def build_Expr(ctx: ASTContext, node: ast.Expr):
        if isinstance(node.value, ast.Constant):
            # Python comments
            return
        if isinstance(node.value, ast.Call):
            return build_stmt(ctx, node.value)
        raise RuntimeError("Unsupported expression")

    @staticmethod
    def build_Pass(ctx: ASTContext, node: ast.Pass):
        return None


build_stmt = ASTTransformer()


def build_stmts(ctx: ASTContext, stmts: list[ast.stmt]):
    results = []
    for stmt in stmts:
        try:
            results.append(build_stmt(ctx, stmt))
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"{traceback.format_exc()}")
            print_error_message(str(e), stmt, ctx.top_func_tree)
            sys.exit(1)
    return results
