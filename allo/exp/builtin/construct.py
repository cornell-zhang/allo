# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
from allo._mlir.dialects import allo as allo_d, memref as memref_d
from allo._mlir.ir import AffineMap, AffineMapAttr, UnitAttr, StringAttr, TypeAttr


##################################################
# Stream Operations
##################################################
@register_builtin_handler("constrcut_stream")
class StreamHandler(BuiltinHandler):

    def build(self, node, *args):
        assert isinstance(node.args[0], ast.Name)
        name = node.args[0].id
        dtype, shape, _, type_hint = self.builder.parse_type_ann(node.args[1])
        stream_type = allo_d.StreamType.get(dtype.build(), depth=dtype.depth)
        op = allo_d.stream_global(
            name, stream_type, shape, ip=self.builder.get_global_ip()
        )
        op.attributes[type_hint] = UnitAttr.get()
        return op


@register_builtin_handler("put")
class StreamPutHandler(BuiltinHandler):

    def build(self, node, *args):
        assert isinstance(node.args[0], ast.Name)
        assert isinstance(node.args[1], ast.Tuple)
        name = node.args[0].id
        indices, ivs, symbols = [], [], []
        for elt in node.args[1].elts:
            aff = self.builder.get_affine_expr(elt, ivs, symbols)
            assert aff is not None
            indices.append(aff)
        affine_map = AffineMap.get(
            dim_count=len(ivs), symbol_count=len(symbols), exprs=indices
        )
        value = self.builder.get_op_result(self.builder.visit(node.args[2]))
        allo_d.put_stream_global(
            name,
            ivs + symbols,
            value,
            AffineMapAttr.get(affine_map),
            ip=self.builder.get_ip(),
        )


@register_builtin_handler("get")
class StreamGetHandler(BuiltinHandler):

    def build(self, node, *args):
        assert isinstance(node.args[0], ast.Name)
        assert isinstance(node.args[1], ast.Tuple)
        name = node.args[0].id
        indices, ivs, symbols = [], [], []
        for elt in node.args[1].elts:
            aff = self.builder.get_affine_expr(elt, ivs, symbols)
            assert aff is not None
            indices.append(aff)
        affine_map = AffineMap.get(
            dim_count=len(ivs), symbol_count=len(symbols), exprs=indices
        )
        result, type_hint = self.builder.build_type(node.args[2])
        op = allo_d.GlobalStreamGetOp(
            result,
            name,
            ivs + symbols,
            AffineMapAttr.get(affine_map),
            ip=self.builder.get_ip(),
        )
        op.attributes[type_hint] = UnitAttr.get()
        return op


##################################################
# Bit Operations
##################################################
@register_builtin_handler("set_bits")
class SetBitsHandler(BuiltinHandler):
    def build(self, node, *args):
        result, type_hint = self.builder.build_type(node.args[2])
        value = self.builder.get_op_result(self.builder.visit(node.args[1]))
        assert isinstance(node.args[0], ast.Subscript)
        base = self.builder.get_op_result(self.builder.visit(node.args[0].value))
        bits = node.args[0].slice
        if isinstance(bits, ast.Slice):  # set slice
            op = allo_d.SetIntSliceOp(
                result,
                base,
                self.builder.get_op_result(self.builder.visit(bits.upper)),
                self.builder.get_op_result(self.builder.visit(bits.lower)),
                value,
                ip=self.builder.get_ip(),
            )
        else:
            op = allo_d.SetIntBitOp(
                result,
                base,
                self.builder.get_op_result(self.builder.visit(bits)),
                value,
                ip=self.builder.get_ip(),
            )
        op.attributes[type_hint] = UnitAttr.get()
        return op


@register_builtin_handler("get_bits")
class GetBitsHandler(BuiltinHandler):
    def build(self, node, *args):
        result, type_hint = self.builder.build_type(node.args[1])
        assert isinstance(node.args[0], ast.Subscript)
        base = self.builder.get_op_result(self.builder.visit(node.args[0].value))
        bits = node.args[0].slice
        if isinstance(bits, ast.Slice):  # get slice
            op = allo_d.GetIntSliceOp(
                result,
                base,
                self.builder.get_op_result(self.builder.visit(bits.upper)),
                self.builder.get_op_result(self.builder.visit(bits.lower)),
                ip=self.builder.get_ip(),
            )
        else:
            op = allo_d.GetIntBitOp(
                result,
                base,
                self.builder.get_op_result(self.builder.visit(bits)),
                ip=self.builder.get_ip(),
            )
        op.attributes[type_hint] = UnitAttr.get()
        return op
