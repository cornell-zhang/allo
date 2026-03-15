# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import numpy
from .handler import BuiltinHandler, register_builtin_handler
import allo._mlir.extras.types as mlir_types
from allo._mlir.dialects import (
    arith as arith_d,
    allo as allo_d,
    memref as memref_d,
    linalg as linalg_d,
)
from allo._mlir.ir import (
    InsertionPoint,
    UnitAttr,
    AffineExpr,
    AffineMap,
    ArrayAttr,
    AffineMapAttr,
    Attribute,
    TypeAttr,
    StringAttr,
    DenseElementsAttr,
)
from allo.ir.types import (
    Index,
    Float,
    Int,
    UInt,
    Fixed,
    UFixed,
    float16,
    bfloat16,
)


@register_builtin_handler("constant")
class ConstantHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        assert isinstance(args_[0], ast.Constant)
        assert isinstance(args_[1], ast.Subscript)
        dtype, _ = self.builder.build_type(args_[1])
        const_op = arith_d.ConstantOp(dtype, args_[0].value, ip=self.builder.get_ip())
        return const_op

    def get_affine_expr(self, node: ast.Call, ivs: list, symbols: list):
        return self.builder.get_affine_expr(node.args[0], ivs, symbols)


@register_builtin_handler("constant_tensor")
class ConstantTensorHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        assert isinstance(args_[0], ast.Name) and isinstance(args_[1], ast.Name)
        assert isinstance(args_[2], ast.Subscript)
        target = args_[0].id
        value = self.builder.symbol_table.global_symbols[args_[1].id]
        dtype, _ = self.builder.build_type(args_[2])
        value_attr = DenseElementsAttr.get(numpy.array(value), type=dtype.element_type)
        op = memref_d.GlobalOp(
            sym_name=StringAttr.get(target),
            type_=TypeAttr.get(dtype),
            sym_visibility=StringAttr.get("private"),
            initial_value=value_attr,
            constant=True,
            alignment=None,
            ip=self.builder.get_ip(),
        )
        self.builder.global_symbols[target] = op


@register_builtin_handler("cast")
class CastHandler(BuiltinHandler):
    @staticmethod
    def infer(*args):
        cast_map = {
            # Index <-> UInt/Int
            (Int, Index): "cast_index",
            (UInt, Index): "cast_index",
            (Index, Int): "cast_index",
            (Index, UInt): "cast_index",
            # UInt/Int <-> Float
            (Int, Float): "cast_si_to_fp",
            (UInt, Float): "cast_ui_to_fp",
            (Float, Int): "cast_fp_to_si",
            (Float, UInt): "cast_fp_to_ui",
            # Float <-> Fixed/UFixed
            (Float, Fixed): "cast_float_to_fixed",
            (Float, UFixed): "cast_float_to_fixed",
            (Fixed, Float): "cast_fixed_to_float",
            (UFixed, Float): "cast_fixed_to_float",
            # Int/UInt <-> Fixed/UFixed
            (Fixed, Int): "cast_fixed_to_int",
            (Fixed, UInt): "cast_fixed_to_int",
            (UFixed, Int): "cast_fixed_to_int",
            (UFixed, UInt): "cast_fixed_to_int",
            (Int, Fixed): "cast_int_to_fixed",
            (Int, UFixed): "cast_int_to_fixed",
            (UInt, Fixed): "cast_int_to_fixed",
            (UInt, UFixed): "cast_int_to_fixed",
            # Fixed/UFixed <-> Fixed/UFixed
            (Fixed, Fixed): "cast_fixed_to_fixed",
            (Fixed, UFixed): "cast_fixed_to_fixed",
            (UFixed, Fixed): "cast_fixed_to_fixed",
            (UFixed, UFixed): "cast_fixed_to_fixed",
            # UInt/Int -> UInt/Int
            (Int, Int): "cast_int",
            (UInt, UInt): "cast_int",
            (Int, UInt): "cast_int",
            (UInt, Int): "cast_int",
            # Float -> Float
            (Float, Float): "cast_float",
            # Float -> Index
            (Float, Index): "cast_float_to_index",
            # Index -> Float
            (Index, Float): "cast_index_to_float",
            # Index -> Fixed/UFixed
            (Index, Fixed): "cast_index_to_fixed",
            (Index, UFixed): "cast_index_to_fixed",
            # Fixed/UFixed -> Index
            (Fixed, Index): "cast_fixed_to_index",
            (UFixed, Index): "cast_fixed_to_index",
        }
        src_type, res_type = args[0], args[1]
        # [NOTE]: float16 <-> bfloat16 not supported
        assert not (
            src_type == float16 and res_type == bfloat16
        ), "f16 -> bf16 not supported"
        assert not (
            src_type == bfloat16 and res_type == float16
        ), "bf16 -> f16 not supported"
        if (type(src_type), type(res_type)) in cast_map:
            handler_name = cast_map[(type(src_type), type(res_type))]
        else:
            raise TypeError(f"Invalid casting. src: {src_type}, dst: {res_type}")

        return res_type, src_type, handler_name

    def get_affine_expr(self, node: ast.Call, ivs: list, symbols: list):
        return self.builder.get_affine_expr(node.args[0], ivs, symbols)

    def get_operand(self, node: ast.Call):
        return (
            self.builder.get_op_result(self.builder.visit(node.args[0])),
            node.args[1].id,
        )

    def get_result_type(self, node: ast.Call):
        return self.builder.build_type(node.args[2])

    def get_generic_wrapper(self, val, dst_type, type_hint: str):
        buffer_op = self.builder.build_buffer(dst_type, type_hint)  # type hint tagged
        shape = val.type.shape
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
            ip=self.builder.get_ip(),
            inputs=[val],
            outputs=[buffer_op.result],
            result_tensors=[],
            iterator_types=iterator_types_attr,
        )
        cast_op.attributes[type_hint] = UnitAttr.get()
        block_arg_types = [val.type.element_type, dst_type.element_type]
        block = cast_op.regions[0].blocks.append(*block_arg_types)
        return buffer_op, block


@register_builtin_handler("cast_index")
class IndexCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, src_type = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        int_type = src_type if type_hint.startswith("_") else type_hint
        if len(getattr(mlir_type, "shape", [])) == 0:  # scalar
            op = arith_d.IndexCastOp(mlir_type, val, ip=self.builder.get_ip())
            op.attributes[int_type] = UnitAttr.get()
            return op
        else:  # tensor (memref)
            op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
            with InsertionPoint(generic_block):
                # add cast op to block
                yield_value = arith_d.IndexCastOp(
                    mlir_type.element_type, generic_block.arguments[0]
                )
                yield_value.attributes[int_type] = UnitAttr.get()
                linalg_d.YieldOp([yield_value])
            return op


@register_builtin_handler("cast_si_to_fp")
class SIToFPCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, src_type = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:  # scalar
            op = arith_d.SIToFPOp(mlir_type, val, ip=self.builder.get_ip())
            op.attributes[src_type] = UnitAttr.get()
            return op
        # tensor (memref)
        op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
        with InsertionPoint(generic_block):
            # add cast op to block
            yield_value = arith_d.SIToFPOp(
                mlir_type.element_type, generic_block.arguments[0]
            )
            yield_value.attributes[src_type] = UnitAttr.get()
            linalg_d.YieldOp([yield_value])
        return op


@register_builtin_handler("cast_ui_to_fp")
class UIToFPCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, src_type = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:  # scalar
            op = arith_d.UIToFPOp(mlir_type, val, ip=self.builder.get_ip())
            op.attributes[src_type] = UnitAttr.get()
            return op
        # tensor (memref)
        op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
        with InsertionPoint(generic_block):
            # add cast op to block
            yield_value = arith_d.UIToFPOp(
                mlir_type.element_type, generic_block.arguments[0]
            )
            yield_value.attributes[src_type] = UnitAttr.get()
            linalg_d.YieldOp([yield_value])
        return op


@register_builtin_handler("cast_fp_to_si")
class FPToSICastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, _ = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:  # scalar
            op = arith_d.FPToSIOp(mlir_type, val, ip=self.builder.get_ip())
            op.attributes[type_hint] = UnitAttr.get()
            return op
        # tensor (memref)
        op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
        with InsertionPoint(generic_block):
            # add cast op to block
            yield_value = arith_d.FPToSIOp(
                mlir_type.element_type, generic_block.arguments[0]
            )
            yield_value.attributes[type_hint] = UnitAttr.get()
            linalg_d.YieldOp([yield_value])
        return op


@register_builtin_handler("cast_fp_to_ui")
class FPToUICastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, _ = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:  # scalar
            op = arith_d.FPToUIOp(mlir_type, val, ip=self.builder.get_ip())
            op.attributes[type_hint] = UnitAttr.get()
            return op
        # tensor (memref)
        op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
        with InsertionPoint(generic_block):
            # add cast op to block
            yield_value = arith_d.FPToUIOp(
                mlir_type.element_type, generic_block.arguments[0]
            )
            yield_value.attributes[type_hint] = UnitAttr.get()
            linalg_d.YieldOp([yield_value])
        return op


@register_builtin_handler("cast_float_to_fixed")
class FloatToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, _ = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:  # scalar
            op = allo_d.FloatToFixedOp(mlir_type, val, ip=self.builder.get_ip())
            return op
        else:  # tensor (memref)
            op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
            with InsertionPoint(generic_block):
                # add cast op to block
                yield_value = allo_d.FloatToFixedOp(
                    mlir_type.element_type, generic_block.arguments[0]
                )
                linalg_d.YieldOp([yield_value])
            return op


@register_builtin_handler("cast_fixed_to_float")
class FixedToFloatCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, src_type = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:  # scalar
            op = allo_d.FixedToFloatOp(mlir_type, val, ip=self.builder.get_ip())
            op.attributes[src_type] = UnitAttr.get()
            return op
        # tensor (memref)
        op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
        with InsertionPoint(generic_block):
            # add cast op to block
            yield_value = allo_d.FixedToFloatOp(
                mlir_type.element_type, generic_block.arguments[0]
            )
            yield_value.attributes[src_type] = UnitAttr.get()
            linalg_d.YieldOp([yield_value])
        return op


@register_builtin_handler("cast_fixed_to_int")
class FixedToIntCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, src_type = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:  # scalar
            op = allo_d.FixedToIntOp(mlir_type, val, ip=self.builder.get_ip())
            op.attributes[src_type] = UnitAttr.get()
            return op
        else:  # tensor (memref)
            op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
            with InsertionPoint(generic_block):
                # add cast op to block
                yield_value = allo_d.FixedToIntOp(
                    mlir_type.element_type, generic_block.arguments[0]
                )
                yield_value.attributes[src_type] = UnitAttr.get()
                linalg_d.YieldOp([yield_value])
            return op


@register_builtin_handler("cast_int_to_fixed")
class IntToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, src_type = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:  # scalar
            op = allo_d.IntToFixedOp(mlir_type, val, ip=self.builder.get_ip())
            op.attributes[src_type] = UnitAttr.get()
            return op
        else:  # tensor (memref)
            op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
            with InsertionPoint(generic_block):
                # add cast op to block
                yield_value = allo_d.IntToFixedOp(
                    mlir_type.element_type, generic_block.arguments[0]
                )
                yield_value.attributes[src_type] = UnitAttr.get()
                linalg_d.YieldOp([yield_value])
            return op


@register_builtin_handler("cast_fixed_to_fixed")
class FixedToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, src_type = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:  # scalar
            op = allo_d.FixedToFixedOp(mlir_type, val, ip=self.builder.get_ip())
            op.attributes[src_type] = UnitAttr.get()
            return op
        else:  # tensor (memref)
            op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
            with InsertionPoint(generic_block):
                # add cast op to block
                yield_value = allo_d.FixedToFixedOp(
                    mlir_type.element_type, generic_block.arguments[0]
                )
                yield_value.attributes[src_type] = UnitAttr.get()
                linalg_d.YieldOp([yield_value])
            return op


@register_builtin_handler("cast_int")
class IntCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, src_type = self.get_operand(node)
        dst_type, type_hint = self.get_result_type(node)
        if len(getattr(dst_type, "shape", [])) == 0:
            is_scalar = True
            src_width = val.type.width
            dst_width = dst_type.width
        else:
            is_scalar = False
            src_width = val.type.element_type.width
            dst_width = dst_type.element_type.width
        if src_width > dst_width:
            opcls = arith_d.TruncIOp
        elif src_width < dst_width:
            if src_type.startswith("u"):
                opcls = arith_d.ExtUIOp
            else:
                opcls = arith_d.ExtSIOp
        else:
            return val
        if is_scalar:  # scalar
            op = opcls(dst_type, val, ip=self.builder.get_ip())
            op.attributes[src_type] = UnitAttr.get()
            return op
        else:  # tensor (memref)
            op, generic_block = self.get_generic_wrapper(val, dst_type, type_hint)
            with InsertionPoint(generic_block):
                # add cast op to block
                yield_value = opcls(dst_type.element_type, generic_block.arguments[0])
                yield_value.attributes[src_type] = UnitAttr.get()
                linalg_d.YieldOp([yield_value])
            return op


@register_builtin_handler("cast_float")
class FloatCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, _ = self.get_operand(node)
        dst_type, type_hint = self.get_result_type(node)
        if len(getattr(dst_type, "shape", [])) == 0:
            is_scalar = True
            src_width = val.type.width
            dst_width = dst_type.width
        else:
            is_scalar = False
            src_width = val.type.element_type.width
            dst_width = dst_type.element_type.width
        if src_width > dst_width:
            opcls = arith_d.TruncFOp
        elif src_width < dst_width:
            opcls = arith_d.ExtFOp
        else:
            return val
        if is_scalar:  # scalar
            op = opcls(dst_type, val, ip=self.builder.get_ip())
        else:  # tensor (memref)
            op, generic_block = self.get_generic_wrapper(val, dst_type, type_hint)
            with InsertionPoint(generic_block):
                # add cast op to block
                yield_value = opcls(dst_type.element_type, generic_block.arguments[0])
                linalg_d.YieldOp([yield_value])
        return op


@register_builtin_handler("cast_float_to_index")
class FloatToIndexCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, s_rc_type = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        # FP -> UI -> Index
        if len(getattr(mlir_type, "shape", [])) == 0:
            op = arith_d.FPToUIOp(mlir_types.i(32), val, ip=self.builder.get_ip())
            op = arith_d.IndexCastOp(mlir_type, op.result, ip=self.builder.get_ip())
            return op
        else:
            op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
            with InsertionPoint(generic_block):
                # add cast op to block
                value = arith_d.FPToUIOp(mlir_types.i(32), generic_block.arguments[0])
                yield_value = arith_d.IndexCastOp(mlir_type.element_type, value)
                linalg_d.YieldOp([yield_value])
            return op


@register_builtin_handler("cast_index_to_float")
class IndexToFloatCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, _ = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        # Index -> SI -> FP
        if len(getattr(mlir_type, "shape", [])) == 0:
            op = arith_d.IndexCastOp(mlir_types.i(32), val, ip=self.builder.get_ip())
            op = arith_d.SIToFPOp(mlir_type, op.result, ip=self.builder.get_ip())
            return op
        else:
            op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
            with InsertionPoint(generic_block):
                # add cast op to block
                value = arith_d.IndexCastOp(
                    mlir_types.i(32), generic_block.arguments[0]
                )
                yield_value = arith_d.SIToFPOp(mlir_type.element_type, value)
                linalg_d.YieldOp([yield_value])
            return op


@register_builtin_handler("cast_index_to_fixed")
class IndexToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, _ = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:
            op = arith_d.IndexCastOp(mlir_types.i(32), val, ip=self.builder.get_ip())
            op = allo_d.IntToFixedOp(mlir_type, op.result, ip=self.builder.get_ip())
            op.attributes[type_hint] = UnitAttr.get()
            return op
        else:
            op, generic_block = self.get_generic_wrapper(val, mlir_type)
            with InsertionPoint(generic_block):
                # add cast op to block
                value = arith_d.IndexCastOp(
                    mlir_types.i(32), generic_block.arguments[0]
                )
                yield_value = allo_d.IntToFixedOp(mlir_type.element_type, value)
                yield_value.attributes[type_hint] = UnitAttr.get()
                linalg_d.YieldOp([yield_value])
            return op


@register_builtin_handler("cast_fixed_to_index")
class FixedToIndexCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val, src_type = self.get_operand(node)
        mlir_type, type_hint = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:  # scalar
            op = allo_d.FixedToIntOp(mlir_types.i(32), val, ip=self.builder.get_ip())
            op.attributes[src_type] = UnitAttr.get()
            op = arith_d.IndexCastOp(mlir_type, op.result, ip=self.builder.get_ip())
            return op
        else:  # tensor (memref)
            op, generic_block = self.get_generic_wrapper(val, mlir_type, type_hint)
            with InsertionPoint(generic_block):
                # add cast op to block
                value = allo_d.FixedToIntOp(
                    mlir_types.i(32), generic_block.arguments[0]
                )
                value.attributes[src_type] = UnitAttr.get()
                yield_value = arith_d.IndexCastOp(mlir_type.element_type, value)
                linalg_d.YieldOp([yield_value])
            return op


@register_builtin_handler("broadcast")
class BroadcastHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        origianl = self.builder.get_op_result(self.builder.visit(args_[0]))
        assert isinstance(args_[1], ast.Tuple) and isinstance(args_[2], ast.Subscript)
        dims = [v.value for v in args_[1].elts]
        alloc_op = self.builder.build_buffer(*self.builder.build_type(args_[2]))
        with self.builder.get_ip():
            if len(getattr(origianl.type, "shape", [])) == 0:
                linalg_d.fill(origianl, outs=[alloc_op.result])
            else:
                linalg_d.broadcast(
                    input=origianl, outs=[alloc_op.result], dimensions=dims
                )
        return alloc_op
