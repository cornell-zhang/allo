# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Reference: taichi/python/taichi/lang/ast/transform.py
# pylint: disable=no-name-in-module, unused-argument

import gc
import ast
from hcl_mlir.ir import (
    Module,
    Location,
    InsertionPoint,
    FunctionType,
    MemRefType,
    RankedTensorType,
    IntegerType,
    IndexType,
    F32Type,
    UnitAttr,
    IntegerAttr,
    StringAttr,
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
)
from hcl_mlir.dialects import (
    hcl as hcl_d,
    func as func_d,
    memref as memref_d,
    tensor as tensor_d,
    affine as affine_d,
    scf as scf_d,
    arith as arith_d,
    math as math_d,
    linalg as linalg_d,
)
from hcl_mlir.exceptions import DTypeError
from .transform import build_for_loops
from .utils import (
    MockArg,
    MockScalar,
    MockConstant,
    MockBuffer,
    get_extra_type_hints,
    get_kwarg,
)
from .types import Int, UInt, Index, Float, Fixed, UFixed, Struct
from .visitor import ASTVisitor, ASTContext
from .symbol_resolver import ASTResolver


def build_shaped_type(dtype, shape, enable_tensor=False):
    if len(shape) == 0:
        return dtype.build()
    if not enable_tensor:
        return MemRefType.get(shape, dtype.build())
    return RankedTensorType.get(shape, dtype.build())


class ASTBuilder(ASTVisitor):
    def __call__(self, ctx, node):
        method = getattr(self, "build_" + node.__class__.__name__, None)
        if method is None:
            error_msg = f'Unsupported node "{node.__class__.__name__}"'
            raise RuntimeError(error_msg)
        with ctx.mlir_ctx, Location.unknown():
            res = method(ctx, node)
            return res


class ASTTransformer(ASTBuilder):
    @staticmethod
    def build_Name(ctx, node):
        if node.id in ctx.buffers:
            return ctx.buffers[node.id]
        if node.id in ctx.global_vars:
            return MockConstant(ctx.global_vars[node.id], ctx)
        raise RuntimeError("Unsupported Name")

    @staticmethod
    def build_Constant(ctx, node):
        return MockConstant(node.value, ctx)

    @staticmethod
    def build_Attribute(ctx, node):
        if node.attr == "T":
            ip = ctx.get_ip()
            shape = node.shape
            new_arg = build_stmt(ctx, node.value).result
            memref_type = MemRefType.get(shape, node.dtype.build())
            alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ip)
            index_exprs = []
            for dim in range(len(shape)):
                index_exprs.append(AffineExpr.get_dim(dim))
            affine_map_in = AffineMap.get(
                dim_count=len(shape), symbol_count=0, exprs=index_exprs
            )
            affine_map_out = AffineMap.get(
                dim_count=len(shape), symbol_count=0, exprs=index_exprs[::-1]
            )
            indexing_maps_attr = ArrayAttr.get(
                [AffineMapAttr.get(affine_map_in), AffineMapAttr.get(affine_map_out)]
            )
            iterator_types_attr = ArrayAttr.get(
                [Attribute.parse("#linalg.iterator_type<parallel>")] * len(shape)
            )
            op = linalg_d.GenericOp(
                indexing_maps=indexing_maps_attr,
                ip=ip,
                inputs=[new_arg],
                outputs=[alloc_op],
                result_tensors=[],
                iterator_types=iterator_types_attr,
            )
            # input and output types
            block_arg_types = [node.value.dtype.build(), node.dtype.build()]
            block = op.regions[0].blocks.append(*block_arg_types)
            ctx.set_ip(block)
            linalg_d.YieldOp([block.arguments[0]], ip=ctx.get_ip())
            ctx.pop_ip()
            if hasattr(node, "keywords") and len(node.keywords) > 0:
                op.result_tensors.owner.attributes["op_name"] = StringAttr.get(
                    node.keywords[0].value.value
                )
            else:
                op.result_tensors.owner.attributes["op_name"] = StringAttr.get(
                    f"transpose_{ctx.unnamed_linalg_op_count}"
                )
                ctx.unnamed_linalg_op_count += 1
            return alloc_op
        if node.attr == "reverse":
            value = build_stmt(ctx, node.value)
            bit_reverse = hcl_d.BitReverseOp(value.result, ip=ctx.get_ip())
            return bit_reverse
        raise RuntimeError("Unsupported Attribute")

    @staticmethod
    def build_all_for(ctx, node):
        grid = [
            x.value if isinstance(x, ast.Constant) else ctx.global_vars[x.id]
            for x in node.iter.args
        ]
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
        for_loops = build_for_loops(grid, ctx.get_ip(), names, stage_name)
        ivs = [loop.induction_variable for loop in for_loops]
        for name, iv in zip(names, ivs):
            ctx.buffers[name] = MockArg(iv)
        ctx.set_ip(for_loops[-1].body.operations[0])
        build_stmts(ctx, node.body)
        if (
            isinstance(node.iter.func, ast.Attribute)
            and node.iter.func.attr == "reduction"
        ):
            for loop in for_loops:
                loop.attributes["reduction"] = UnitAttr.get()
        # Remove loop variables
        for name, iv in zip(names, ivs):
            ctx.buffers.pop(name)
        for_loops = None
        # Not sure why the for loops will not be collected if we do not call gc.collect()
        gc.collect()
        ctx.pop_ip()

    @staticmethod
    def build_For(ctx, node):
        if node.orelse:
            raise RuntimeError("'else' clause for 'for' not supported in Allo kernels")
        with ctx.loop_scope_guard():
            if isinstance(node.iter, ast.Call):
                obj = ASTResolver.resolve(node.iter.func, ctx.global_vars)
                if (
                    obj is None
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == "range"
                ) or (obj is not None and obj.__name__ in {"grid", "reduction"}):
                    return ASTTransformer.build_all_for(ctx, node)
            raise RuntimeError("Unsupported for loop")

    # pylint: disable=too-many-branches, inconsistent-return-statements
    @staticmethod
    def build_cast_op(ctx, op, src_type, res_type):
        # determine cast op
        CastOpClass = None
        if type(res_type) is type(src_type) and res_type == src_type:
            return op
        if isinstance(src_type, (Int, UInt)) and isinstance(res_type, Index):
            CastOpClass = arith_d.IndexCastOp
        elif isinstance(src_type, Index) and isinstance(res_type, (Int, UInt)):
            CastOpClass = arith_d.IndexCastOp
        elif isinstance(src_type, Int) and isinstance(res_type, Float):
            CastOpClass = arith_d.SIToFPOp
        elif isinstance(src_type, UInt) and isinstance(res_type, Float):
            CastOpClass = arith_d.UIToFPOp
        elif isinstance(src_type, Float) and isinstance(res_type, Int):
            CastOpClass = arith_d.FPToSIOp
        elif isinstance(src_type, Float) and isinstance(res_type, Index):
            # FP to Index is not supported in MLIR
            # we need to cast to UInt first, then cast to Index
            op = arith_d.FPToUIOp(IndexType.get(), op.result, ip=ctx.get_ip())
            CastOpClass = arith_d.IndexCastOp  # proceed to build cast to index
        elif isinstance(src_type, Float) and isinstance(res_type, UInt):
            CastOpClass = arith_d.FPToUIOp
        elif isinstance(src_type, (Int, UInt)) and isinstance(res_type, (Int, UInt)):
            if src_type.bits > res_type.bits:
                CastOpClass = arith_d.TruncIOp
            elif src_type.bits == res_type.bits:
                return op
            else:  # src_type.bits < res_type.bits
                # pylint: disable=else-if-used
                if (
                    isinstance(
                        op, (hcl_d.GetIntBitOp, hcl_d.GetIntSliceOp, arith_d.ShLIOp)
                    )
                    or src_type.bits == 1
                ):
                    CastOpClass = arith_d.ExtUIOp
                elif isinstance(src_type, UInt):
                    CastOpClass = arith_d.ExtUIOp
                else:
                    CastOpClass = arith_d.ExtSIOp
        elif isinstance(src_type, Float) and isinstance(res_type, Float):
            if res_type.bits < src_type.bits:
                CastOpClass = arith_d.TruncFOp
            elif res_type.bits > src_type.bits:
                CastOpClass = arith_d.ExtFOp
            else:
                return op
        elif isinstance(src_type, Float) and isinstance(res_type, (Fixed, UFixed)):
            CastOpClass = hcl_d.FloatToFixedOp
        elif isinstance(src_type, (Fixed, UFixed)) and isinstance(res_type, Float):
            CastOpClass = hcl_d.FixedToFloatOp
        elif isinstance(src_type, (Fixed, UFixed)) and isinstance(
            res_type, (Int, UInt)
        ):
            CastOpClass = hcl_d.FixedToIntOp
        elif isinstance(src_type, (Int, UInt)) and isinstance(
            res_type, (Fixed, UFixed)
        ):
            CastOpClass = hcl_d.IntToFixedOp
        elif isinstance(src_type, (Fixed, UFixed)) and isinstance(
            res_type, (Fixed, UFixed)
        ):
            if src_type == res_type:
                return op
            CastOpClass = hcl_d.FixedToFixedOp
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
            op.result = op.expr.result
            op.ir_op = op.expr.ir_op
            return
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
            CastOpClass = hcl_d.IntToStructOp
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

        # build the cast op
        if isinstance(res_type, (Int, UInt, Struct)):
            mlir_type = res_type.build()
            cast_op = CastOpClass(mlir_type, op.result, ip=ctx.get_ip())
            if isinstance(res_type, (UInt, Struct)):
                cast_op.attributes["unsigned"] = UnitAttr.get()
        else:
            mlir_type = res_type.build()
            cast_op = CastOpClass(mlir_type, op.result, ip=ctx.get_ip())
        return cast_op

    @staticmethod
    def build_general_binop(ctx, node, lhs, rhs):
        opcls = {
            ast.Add: {
                Float: arith_d.AddFOp,
                Int: arith_d.AddIOp,
                UInt: arith_d.AddIOp,
                Fixed: hcl_d.AddFixedOp,
                UFixed: hcl_d.AddFixedOp,
            },
            ast.Sub: {
                Float: arith_d.SubFOp,
                Int: arith_d.SubIOp,
                UInt: arith_d.SubIOp,
                Fixed: hcl_d.SubFixedOp,
                UFixed: hcl_d.SubFixedOp,
            },
            ast.Mult: {
                Float: arith_d.MulFOp,
                Int: arith_d.MulIOp,
                UInt: arith_d.MulIOp,
                Fixed: hcl_d.MulFixedOp,
                UFixed: hcl_d.MulFixedOp,
            },
            ast.Div: {
                Float: arith_d.DivFOp,
                Int: arith_d.DivSIOp,
                UInt: arith_d.DivUIOp,
                Fixed: hcl_d.DivFixedOp,
                UFixed: hcl_d.DivFixedOp,
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
                Fixed: RuntimeError,
                UFixed: RuntimeError,
            },
            ast.RShift: {
                Float: RuntimeError,
                Int: arith_d.ShRSIOp,
                UInt: arith_d.ShRUIOp,
                Fixed: RuntimeError,
                UFixed: RuntimeError,
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
        if len(node.shape) > 0:
            new_args = [lhs.result, rhs.result]
            attr = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Div: "div",
            }.get(type(node.op))
            return ASTTransformer.build_linalg_op(
                ctx, node=node, op_name=attr, new_args=new_args
            )
        return opcls[type(node.dtype)](lhs.result, rhs.result, ip=ctx.get_ip())

    @staticmethod
    def build_UnaryOp(ctx, node):
        if isinstance(node.op, ast.USub):
            opcls = {
                "float": arith_d.NegFOp,
                "int": RuntimeError,
                "fixed": RuntimeError,
            }
        elif isinstance(node.op, ast.UAdd):
            opcls = {
                "float": RuntimeError,
                "int": RuntimeError,
                "fixed": RuntimeError,
            }
        else:
            raise RuntimeError("Unsupported unary op")
        if not isinstance(node.operand, ast.Constant):
            raise RuntimeError("Only support constant for unary op")
        if isinstance(node.operand.value, int):
            op = opcls["int"]
        elif isinstance(node.operand.value, float):
            op = opcls["float"]
        else:
            raise RuntimeError(
                f"Unsupported types for unary op: {type(node.operand.value)}"
            )
        return op(MockConstant(node.operand.value, ctx).result, ip=ctx.get_ip())

    @staticmethod
    def build_BinOp(ctx, node):
        lhs = build_stmt(ctx, node.left)
        rhs = build_stmt(ctx, node.right)
        # Cast lhs and rhs to the same type
        lhs = ASTTransformer.build_cast_op(ctx, lhs, node.left.dtype, node.dtype)
        rhs = ASTTransformer.build_cast_op(ctx, rhs, node.right.dtype, node.dtype)
        return ASTTransformer.build_general_binop(ctx, node, lhs, rhs)

    @staticmethod
    def build_store(ctx, node, val):
        if isinstance(node, ast.Subscript) and len(node.value.shape) > 0:
            # Note: Python 3.10 will generate different AST for Subscript compared to Python 3.8
            #       3.10 directly flattens the Index node and removes all the None attributes
            #       inside the node
            # pylint: disable=redefined-builtin
            slice = (
                node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            )
            elts = slice.elts if isinstance(slice, ast.Tuple) else [slice]
            ctx.dim_count = 0
            ctx.affine_vars = []
            index_exprs = []
            for index in elts:
                index_exprs.append(ASTTransformer.build_affine_expr(ctx, index))
            affine_map = AffineMap.get(
                dim_count=ctx.dim_count, symbol_count=0, exprs=index_exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            if isinstance(ctx.buffers[node.value.id], MockScalar):
                target = ctx.buffers[node.value.id].op.result
            else:
                target = ctx.buffers[node.value.id].result
            ivs = [ctx.buffers[x].result for x in ctx.affine_vars]
            store_op = affine_d.AffineStoreOp(
                val.result, target, ivs, affine_attr, ip=ctx.get_ip()
            )
            store_op.attributes["to"] = StringAttr.get(node.value.id)
            return store_op
        if isinstance(node, ast.Name):  # scalar
            affine_map = AffineMap.get(
                dim_count=0, symbol_count=0, exprs=[AffineConstantExpr.get(0)]
            )
            affine_attr = AffineMapAttr.get(affine_map)
            if isinstance(ctx.buffers[node.id], MockScalar):
                target = ctx.buffers[node.id].op.result
            else:
                target = ctx.buffers[node.id].result
            store_op = affine_d.AffineStoreOp(
                val.result, target, [], affine_attr, ip=ctx.get_ip()
            )
            store_op.attributes["to"] = StringAttr.get(node.id)
            return store_op
        if isinstance(node, ast.Subscript):  # bit operation
            slice = (
                node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            )
            elts = slice.elts if isinstance(slice, ast.Tuple) else [slice]
            if isinstance(node.slice, ast.Index):
                assert len(elts) == 1, "Only support single index for set_bit"
                index = build_stmt(ctx, node.slice.value)
                index = ASTTransformer.build_cast_op(
                    ctx, index, node.slice.value.dtype, Index()
                )
                value = build_stmt(ctx, node.value)
                # TODO: Test if rhs is uint1
                set_bit_op = hcl_d.SetIntBitOp(
                    node.value.dtype.build(),
                    value.result,
                    index.result,
                    val.result,
                    ip=ctx.get_ip(),
                )
                # write the updated integer back to the scalar
                store_op = ASTTransformer.build_store(ctx, node.value, set_bit_op)
                return store_op
            if isinstance(node.slice, ast.Slice):
                assert len(elts) == 1, "Only support a single slice for set_slice"
                # The backend implementation is different from the Python convention
                # The lower bound is inclusive and the upper bound is also inclusive
                node.slice.upper.value -= 1
                start = build_stmt(ctx, node.slice.lower)
                start = ASTTransformer.build_cast_op(
                    ctx, start, node.slice.lower.dtype, Index()
                )
                end = build_stmt(ctx, node.slice.upper)
                end = ASTTransformer.build_cast_op(
                    ctx, end, node.slice.upper.dtype, Index()
                )
                value = build_stmt(ctx, node.value)
                # TODO: Test if rhs has the correct bitwidth
                set_slice_op = hcl_d.SetIntSliceOp(
                    node.value.dtype.build(),
                    value.result,
                    end.result,
                    start.result,
                    val.result,
                    ip=ctx.get_ip(),
                )
                # write the updated integer back to the scalar
                store_op = ASTTransformer.build_store(ctx, node.value, set_slice_op)
                return store_op
        raise RuntimeError("Unsupported store")

    @staticmethod
    def build_Assign(ctx, node):
        # Compute RHS
        rhs = build_stmt(ctx, node.value)
        if len(node.targets) > 1:
            raise RuntimeError("Cannot assign to multiple targets")
        if isinstance(rhs, (func_d.CallOp, tensor_d.EmptyOp, memref_d.AllocOp)):
            if len(node.targets) > 1:
                raise RuntimeError("Cannot support multiple results yet")
            if isinstance(node.targets[0], ast.Name):
                if isinstance(rhs, func_d.CallOp):
                    rhs.attributes["name"] = StringAttr.get(node.targets[0].id)
                ctx.buffers[node.targets[0].id] = rhs
                return rhs
        # Store LHS
        rhs = ASTTransformer.build_cast_op(ctx, rhs, node.value.dtype, node.dtype)
        store_op = ASTTransformer.build_store(ctx, node.targets[0], rhs)
        return store_op

    @staticmethod
    def build_constant_tensor(ctx, node):
        np_values = node.np_values
        value_attr = DenseElementsAttr.get(np_values)
        sym_name = StringAttr.get(node.target.id)
        sym_visibility = StringAttr.get("private")
        memref_type = MemRefType.get(np_values.shape, node.dtype.build())
        type_attr = TypeAttr.get(memref_type)
        const_tensor = memref_d.GlobalOp(
            sym_name=sym_name,
            type_=type_attr,
            sym_visibility=sym_visibility,
            initial_value=value_attr,
            # TODO: Use dataflow analysis to determine whether some store ops
            #       are operated on this tensor
            constant=False,
            alignment=None,
            ip=InsertionPoint(ctx.top_func),
        )
        return const_tensor

    @staticmethod
    def build_AugAssign(ctx, node):
        # Compute RHS
        rhs = build_stmt(ctx, node.value)
        # Load LHS
        if isinstance(node.target, ast.Subscript):
            # pylint: disable=redefined-variable-type
            node.target.ctx = ast.Load()
            lhs = build_stmt(ctx, node.target)
            node.target.ctx = ast.Store()
            lhs.attributes["from"] = StringAttr.get(node.target.value.id)
        elif isinstance(node.target, ast.Name):  # scalar
            lhs = ctx.buffers[node.target.id]
        else:
            raise RuntimeError("Unsupported AugAssign")
        # Cast rhs to the target type
        rhs = ASTTransformer.build_cast_op(ctx, rhs, node.value.dtype, node.dtype)
        # Aug LHS
        res = ASTTransformer.build_general_binop(ctx, node, lhs, rhs)
        # Store LHS
        store_op = ASTTransformer.build_store(ctx, node.target, res)
        return store_op

    @staticmethod
    def build_affine_expr(ctx, node):
        if isinstance(node, ast.Name):
            if (
                node.id in ctx.buffers
                and isinstance(ctx.buffers[node.id], MockArg)
                and str(ctx.buffers[node.id].result.type) == "index"
            ):
                ctx.dim_count += 1
                ctx.affine_vars.append(node.id)
                return AffineExpr.get_dim(ctx.dim_count - 1)
            if (
                node.id in ctx.buffers
                and isinstance(ctx.buffers[node.id], MockScalar)
                and isinstance(ctx.buffers[node.id].dtype, Index)
            ):
                return ASTTransformer.build_affine_expr(ctx, ctx.buffers[node.id].value)
            return None
        if isinstance(node, ast.BinOp):
            lhs = ASTTransformer.build_affine_expr(ctx, node.left)
            rhs = ASTTransformer.build_affine_expr(ctx, node.right)
            op = {
                ast.Add: lambda l, r: l + r,
                ast.Sub: lambda l, r: l - r,
                ast.Mult: lambda l, r: l * r,
                ast.Div: lambda l, r: l / r,
                ast.FloorDiv: lambda l, r: l // r,
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
        raise RuntimeError("Unsupported affine expression")

    @staticmethod
    def build_Subscript(ctx, node):
        # pylint: disable=redefined-builtin
        slice = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
        elts = slice.elts if isinstance(slice, ast.Tuple) else [slice]
        if len(node.value.shape) > 0:
            # Load op
            ctx.dim_count = 0
            ctx.affine_vars = []
            index_exprs = []
            is_affine = True
            for index in elts:
                expr = ASTTransformer.build_affine_expr(ctx, index)
                if expr is None:
                    is_affine = False
                    break
                index_exprs.append(expr)
            # pylint: disable=no-else-return
            if is_affine:
                if isinstance(node.ctx, ast.Load):
                    affine_map = AffineMap.get(
                        dim_count=ctx.dim_count, symbol_count=0, exprs=index_exprs
                    )
                    affine_attr = AffineMapAttr.get(affine_map)
                    ivs = [ctx.buffers[x].result for x in ctx.affine_vars]
                    load_op = affine_d.AffineLoadOp(
                        ctx.buffers[node.value.id].result,
                        ivs,
                        affine_attr,
                        ip=ctx.get_ip(),
                    )
                    load_op.attributes["from"] = StringAttr.get(node.value.id)
                    return load_op
                else:
                    raise RuntimeError("Unsupported Subscript")
            else:  # Not affine
                new_indices = []
                for index in elts:
                    expr = build_stmt(ctx, index)
                    # cast to index type
                    expr_res = expr.result
                    if str(expr_res.type) == "i32":
                        expr = arith_d.IndexCastOp(
                            IndexType.get(), expr_res, ip=ctx.get_ip()
                        )
                    else:
                        raise RuntimeError(f"Unsupported index type, got {expr.type}")
                    new_indices.append(expr)
                # pylint: disable=redefined-variable-type
                load_op = memref_d.LoadOp(
                    ctx.buffers[node.value.id].result, new_indices, ip=ctx.get_ip()
                )
                load_op.attributes["from"] = StringAttr.get(node.value.id)
                return load_op
        else:  # bit operation
            value = build_stmt(ctx, node.value)
            if len(node.value.shape) == 0 and isinstance(node.value.dtype, (Int, UInt)):
                # Bit operations should follow the convention in
                # https://github.com/cornell-zhang/heterocl/issues/443
                # >>> a = 0xabcd0123
                # >>> a[28:32] # containing the bit of 28, 29, 30, 31
                # 0xa
                # >>> a[4:24]
                # 0xcd012
                # >>> a[28:32].reverse()
                # 0x5
                if isinstance(node.slice, ast.Index):
                    assert len(elts) == 1, "Only support single index for get_bit"
                    index = build_stmt(ctx, elts[0])
                    index = ASTTransformer.build_cast_op(
                        ctx, index, node.slice.dtype, Index()
                    )
                    return hcl_d.GetIntBitOp(
                        node.dtype.build(), value.result, index.result, ip=ctx.get_ip()
                    )
                elif isinstance(node.slice, ast.Slice):
                    # The backend implementation is different from the Python convention
                    # The lower bound is inclusive and the upper bound is also inclusive
                    node.slice.upper.value -= 1
                    lower = build_stmt(ctx, node.slice.lower)
                    upper = build_stmt(ctx, node.slice.upper)
                    lower = ASTTransformer.build_cast_op(
                        ctx, lower, node.slice.lower.dtype, Index()
                    )
                    upper = ASTTransformer.build_cast_op(
                        ctx, upper, node.slice.upper.dtype, Index()
                    )
                    return hcl_d.GetIntSliceOp(
                        node.dtype.build(),
                        value.result,
                        upper.result,
                        lower.result,
                        ip=ctx.get_ip(),
                    )
                else:
                    raise NotImplementedError
            else:
                raise RuntimeError("Can only access bit (slice) for integers")

    @staticmethod
    def build_AnnAssign(ctx, node):
        if node.value is not None:
            if isinstance(node.value, ast.List) or (
                isinstance(node.value, ast.Name) and node.value.id not in ctx.buffers
            ):
                rhs = ASTTransformer.build_constant_tensor(ctx, node)
            else:
                # Examples:
                # copied: int32 = a
                # init: int32 = 0
                # call: int32 = int(1)
                rhs = build_stmt(ctx, node.value)
        else:
            rhs = None
        shape, dtype = node.shape, node.dtype
        if len(shape) > 0:
            if not ctx.enable_tensor:
                memref_type = build_shaped_type(dtype, shape, False)
                if isinstance(node.value, ast.Name) and node.value.id in ctx.buffers:
                    if isinstance(rhs, (memref_d.AllocOp, MockArg)):
                        alloc_op = memref_d.AllocOp(
                            memref_type, [], [], ip=ctx.get_ip()
                        )
                        alloc_op.attributes["name"] = StringAttr.get(node.target.id)
                        ctx.buffers[node.target.id] = alloc_op
                        with ctx.get_ip():
                            # pylint: disable=unexpected-keyword-arg
                            linalg_d.copy(
                                rhs.result,
                                outs=[alloc_op],
                            )
                    else:
                        raise RuntimeError("Unsupported data type")
                elif isinstance(node.value, (ast.List, ast.Name)):
                    # pylint: disable=redefined-variable-type
                    rhs = memref_d.GetGlobalOp(
                        memref_type,
                        FlatSymbolRefAttr.get(node.target.id),
                        ip=ctx.get_ip(),
                    )
                    ctx.buffers[node.target.id] = rhs
                elif isinstance(node.value, ast.Constant) or (node.value is None):
                    alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ctx.get_ip())
                    alloc_op.attributes["name"] = StringAttr.get(node.target.id)
                    ctx.buffers[node.target.id] = alloc_op
                    if rhs is not None:
                        with ctx.get_ip():
                            rhs = ASTTransformer.build_cast_op(
                                ctx, rhs, node.value.dtype, node.dtype
                            )
                            # pylint: disable=unexpected-keyword-arg
                            linalg_d.fill(rhs.result, outs=[alloc_op.result])
                else:
                    raise RuntimeError("Unsupported data type")
            else:
                tensor_type = build_shaped_type(dtype, shape, True)
                tensorgen_op = tensor_d.GenerateOp(tensor_type, [], ip=ctx.get_ip())
                index_type = []
                for _ in range(len(shape)):
                    index_type.append(IndexType.get())
                ctx.set_ip(tensorgen_op.regions[0].blocks.append(*index_type))
                tensor_d.YieldOp(rhs.result, ip=ctx.get_ip())
                ctx.pop_ip()
                ctx.buffers[node.target.id] = tensorgen_op
        else:
            # TODO: figure out why zero-shape cannot work
            ctx.buffers[node.target.id] = MockScalar(
                node.target.id,
                node.dtype,
                ctx,
                value=node.value,
            )
            if rhs is not None:
                rhs = ASTTransformer.build_cast_op(
                    ctx, rhs, node.value.dtype, node.dtype
                )
                ASTTransformer.build_store(ctx, node.target, rhs)

    @staticmethod
    def build_FunctionDef(ctx, node):
        if ctx.top_func is not None:
            # Nested function def
            # Create a new context to avoid name collision
            old_ctx = ctx
            ctx = ASTContext(
                global_vars=ctx.global_vars,
                mlir_ctx=old_ctx.mlir_ctx,
                verbose=old_ctx.verbose,
            )
            ctx.set_ip(old_ctx.top_func)
            ctx.top_func_tree = node
        else:
            old_ctx = None

        arg_names = []

        # Build input types
        input_types = []
        input_typehints = []
        for arg in node.args.args:
            input_types.append(
                build_shaped_type(arg.dtype, arg.shape, ctx.enable_tensor)
            )
            input_typehints.append(get_extra_type_hints(arg.dtype))
            arg_names.append(arg.arg)

        # Build return type
        output_types = []
        output_typehints = []
        if not (
            (isinstance(node.returns, ast.Constant) and node.returns.value is None)
            or node.returns is None
        ):
            output_types.append(
                build_shaped_type(
                    node.returns.dtype, node.returns.shape, ctx.enable_tensor
                )
            )
            output_typehints.append(get_extra_type_hints(node.returns.dtype))

        # Build function
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        func_type = FunctionType.get(input_types, output_types)
        func_op = func_d.FuncOp(name=node.name, type=func_type, ip=ctx.get_ip())
        func_op.add_entry_block()
        # attach type hints
        func_op.attributes["otypes"] = StringAttr.get("".join(output_typehints))
        func_op.attributes["itypes"] = StringAttr.get("".join(input_typehints))
        # set context
        ctx.top_func = func_op
        ctx.top_func_tree = node
        for name, arg in zip(arg_names, func_op.arguments):
            ctx.buffers[name] = MockArg(arg)
        ctx.func_args[node.name] = arg_names
        ctx.set_ip(func_op.entry_block)
        stmts = build_stmts(ctx, node.body)
        if not isinstance(stmts[-1], func_d.ReturnOp):
            func_d.ReturnOp([], ip=ctx.pop_ip())
        # Recover the old context
        if old_ctx is not None:
            ctx = old_ctx
        # Add the built function to global variable for later reference
        ctx.global_vars[node.name] = func_op
        return func_op

    @staticmethod
    def build_Compare(ctx, node, is_affine=False):
        ATTR_MAP = {
            "int": {
                ast.Eq: 0,
                ast.NotEq: 1,
                ast.Lt: 2,
                ast.LtE: 3,
                ast.Gt: 4,
                ast.GtE: 5,
                "ult": 6,
                "ule": 7,
                "ugt": 8,
                "uge": 9,
            },
            "float": {
                "false": 0,
                ast.Eq: 1,
                ast.Gt: 2,
                ast.GtE: 3,
                ast.Lt: 4,
                ast.LtE: 5,
                "one": 6,
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
                "eq": 0,
                "ne": 1,
                "slt": 2,
                "sle": 3,
                "sgt": 4,
                "sge": 5,
                "ult": 6,
                "ule": 7,
                "ugt": 8,
                "uge": 9,
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
            attr = hcl_d.IntegerSetAttr.get(if_cond_set)
            return attr, ctx.buffers[node.left.id]
        else:
            lhs = build_stmt(ctx, node.left)
            rhs = build_stmt(ctx, node.comparators[0])
            # avoid rebuilding the same op
            rhs_res = rhs.result
            dtype = str(rhs_res.type)
            if dtype.startswith("i"):
                op = ATTR_MAP["int"][type(node.ops[0])]
                op = IntegerAttr.get(IntegerType.get_signless(64), op)
                return arith_d.CmpIOp(op, lhs.result, rhs_res, ip=ctx.get_ip())
            if dtype.startswith("fixed"):
                op = ATTR_MAP["fixed"][type(node.ops[0])]
                op = IntegerAttr.get(IntegerType.get_signless(64), op)
                return hcl_d.CmpFixedOp(op, lhs.result, rhs_res, ip=ctx.get_ip())
            if dtype.startswith("f"):
                op = ATTR_MAP["float"][type(node.ops[0])]
                op = IntegerAttr.get(IntegerType.get_signless(64), op)
                return arith_d.CmpFOp(op, lhs.result, rhs_res, ip=ctx.get_ip())
            raise RuntimeError(f"Unsupported types for binary op: {dtype}")

    @staticmethod
    def build_If(ctx, node, is_affine=False):
        if is_affine:
            # Should build the condition on-the-fly
            cond, var = build_stmt(ctx, node.test)
            if_op = affine_d.AffineIfOp(
                cond,
                [var.result],
                ip=ctx.get_ip(),
                hasElse=len(node.orelse),
                results_=[],
            )
        else:
            cond = build_stmt(ctx, node.test)
            if_op = scf_d.IfOp(
                cond.result, results_=[], ip=ctx.get_ip(), hasElse=len(node.orelse)
            )
        ctx.set_ip(if_op.then_block)
        build_stmts(ctx, node.body)
        if is_affine:
            affine_d.AffineYieldOp([], ip=ctx.get_ip())
        else:
            scf_d.YieldOp([], ip=ctx.get_ip())
        ctx.pop_ip()
        if len(node.orelse) > 0:
            ctx.set_ip(if_op.else_block)
            build_stmts(ctx, node.orelse)
            if is_affine:
                affine_d.AffineYieldOp([], ip=ctx.get_ip())
            else:
                scf_d.YieldOp([], ip=ctx.get_ip())
            ctx.pop_ip()

    @staticmethod
    def build_Module(ctx, node):
        with ctx.mlir_ctx:
            module = Module.create(loc=Location.unknown())
        ctx.set_ip(module.body)
        for stmt in node.body:
            build_stmt(ctx, stmt)
        ctx.pop_ip()
        return module

    @staticmethod
    def build_Call(ctx, node):
        obj = ASTResolver.resolve(node.func, ctx.global_vars)
        if obj is None:
            if isinstance(node.func, ast.Attribute):
                # x.T or x.reverse
                assert (
                    len(node.args) == 0
                ), "Only support zero argument for attribute methods"
                return build_stmt(ctx, node.func)
            if node.func.id in {"float", "int"}:
                # Python-Builtin functions
                assert (
                    len(node.args) == 1
                ), "Only support one argument for `float` and `int`"
                stmts = build_stmts(ctx, node.args)
                if node.func.id == "float":
                    return ASTTransformer.build_cast_op(
                        ctx, stmts[0], node.args[0].dtype, Float(32)
                    )
                if node.func.id == "int":
                    return ASTTransformer.build_cast_op(
                        ctx, stmts[0], node.args[0].dtype, Int(32)
                    )
            raise RuntimeError(f"Cannot resolve function `{node.func.id}`")

        if obj.__module__.startswith("allo"):
            # Allo library functions
            new_args = [stmt.result for stmt in build_stmts(ctx, node.args)]
            fn_name = obj.__name__
            if isinstance(new_args[0].type, (F32Type, IntegerType)):
                opcls = {
                    "exp": math_d.ExpOp,
                    "log": math_d.LogOp,
                    "log2": math_d.Log2Op,
                    "log10": math_d.Log10Op,
                    "sqrt": math_d.SqrtOp,
                    "sin": math_d.SinOp,
                    "cos": math_d.CosOp,
                    "tan": math_d.TanOp,
                    "tanh": math_d.TanhOp,
                    "power": math_d.PowFOp,
                }.get(fn_name)
                return opcls(*new_args, ip=ctx.get_ip())
            if isinstance(
                new_args[0].type, (MemRefType, RankedTensorType)
            ) and fn_name in {
                "matmul",
                "bmm",
                "softmax",
                "exp",
                "abs",
                "log",
                "add",
                "sub",
                "div",
            }:
                return ASTTransformer.build_linalg_op(
                    ctx, node=node, op_name=fn_name, new_args=new_args
                )
            raise RuntimeError(
                f"Unsupported function {fn_name} with type {new_args[0].type}"
            )

        # User-defined subfunction
        func = ctx.global_vars[node.func.id]
        if isinstance(func, func_d.FuncOp):
            # Has already been defined in the top-level scope
            stmts = [func]
        else:
            # Create a new context to avoid name collision
            func_ctx = ASTContext(
                global_vars=ctx.global_vars, mlir_ctx=ctx.mlir_ctx, verbose=ctx.verbose
            )
            func_ctx.set_ip(ctx.top_func)
            stmts = build_stmts(func_ctx, node.tree.body)
            func_ctx.pop_ip()
            # Attach buffers to function
            # FIXME: Should create subschedule
            for name, buffer in func_ctx.buffers.items():
                if isinstance(buffer, (memref_d.AllocOp, MockArg)):
                    # Intermediate buffers and function arguments
                    setattr(func, name, MockBuffer(f"{node.func.id}.{name}"))
        # Build call function in the top-level
        new_args = [stmt.result for stmt in build_stmts(ctx, node.args)]
        call_op = func_d.CallOp(
            stmts[-1].type.results,
            FlatSymbolRefAttr.get(node.func.id),
            new_args,
            ip=ctx.get_ip(),
        )
        return call_op

    @staticmethod
    def build_linalg_op(ctx, node, op_name, new_args):
        # +-/ and allo.add() are all supported
        assert op_name is not None and op_name != ""
        attr = op_name
        ip = ctx.get_ip()
        dtype, shape = node.dtype, node.shape
        with ip:
            if not ctx.enable_tensor:
                memref_type = MemRefType.get(shape, dtype.build())
                alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ip)
            else:
                alloc_op = tensor_d.EmptyOp(shape, dtype.build(), ip=ip)
            # init zero
            zero = MockConstant(0, ctx)
            zero = ASTTransformer.build_cast_op(ctx, zero, Int(32), node.dtype)
            # pylint: disable=unexpected-keyword-arg
            linalg_fill = linalg_d.fill(zero.result, outs=[alloc_op.result])
            # add op name for init_zero
            if hasattr(node, "keywords") and len(node.keywords) > 0:
                linalg_fill.owner.attributes["op_name"] = StringAttr.get(
                    f"{node.keywords[0].value.value}_init_zero"
                )
            else:
                linalg_fill.owner.attributes["op_name"] = StringAttr.get(
                    f"{op_name}_init_zero_{ctx.unnamed_linalg_op_count}"
                )
            # build linalg op
            if attr in {"matmul", "bmm", "add", "sub", "div"}:
                op = {
                    "matmul": linalg_d.matmul,
                    "bmm": linalg_d.batch_matmul,
                    "add": linalg_d.add,
                    "sub": linalg_d.sub,
                    "div": linalg_d.div,
                }.get(attr)(new_args[0], new_args[1], outs=[alloc_op])
            elif attr in {"exp", "log", "abs"}:
                op = {
                    "exp": linalg_d.exp,
                    "log": linalg_d.log,
                    "abs": linalg_d.abs,
                }.get(attr)(new_args[0], outs=[alloc_op])
            # TODO: only op.result has .owner and it failed to lower to LLVM, see https://reviews.llvm.org/D153422
            elif attr == "softmax":
                op = linalg_d.SoftmaxOp(
                    input=new_args[0],
                    dimension=1,
                    result=[],
                    output=alloc_op,
                ).result
            else:
                raise RuntimeError("Unsupported operation")
            if hasattr(node, "keywords") and len(node.keywords) > 0:
                op.owner.attributes["op_name"] = StringAttr.get(
                    node.keywords[0].value.value
                )
            else:
                op.owner.attributes["op_name"] = StringAttr.get(
                    f"{attr}_{ctx.unnamed_linalg_op_count}"
                )
                ctx.unnamed_linalg_op_count += 1
        return alloc_op

    @staticmethod
    def build_Return(ctx, node):
        ret = build_stmt(ctx, node.value)
        ret = ASTTransformer.build_cast_op(
            ctx, ret, node.dtype, ctx.top_func_tree.dtype
        )
        return func_d.ReturnOp([ret.result], ip=ctx.pop_ip())

    @staticmethod
    def build_Expr(ctx, node):
        return build_stmt(ctx, node.value)

    @staticmethod
    def build_Pass(ctx, node):
        return None


build_stmt = ASTTransformer()


def build_stmts(ctx, stmts):
    results = []
    for stmt in stmts:
        results.append(build_stmt(ctx, stmt))
    return results
