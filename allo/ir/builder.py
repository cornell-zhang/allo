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
)
import hcl_mlir
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


class ASTBuilder(ASTVisitor):
    def __call__(self, ctx, node, **kwargs):
        if node is None:
            return None
        method = getattr(self, "build_" + node.__class__.__name__, None)
        if method is None:
            error_msg = f'Unsupported node "{node.__class__.__name__}"'
            raise RuntimeError(error_msg)
        with ctx.mlir_ctx, Location.unknown():
            res = method(ctx, node, **kwargs)
            return res


# pylint: disable=too-many-public-methods
class ASTTransformer(ASTBuilder):
    @staticmethod
    def build_Name(ctx, node, val=None):
        if val is not None:
            affine_map = AffineMap.get(
                dim_count=0, symbol_count=0, exprs=[AffineConstantExpr.get(0)]
            )
            affine_attr = AffineMapAttr.get(affine_map)
            buffer = ctx.buffers[node.id]
            target = (
                buffer.op.result if isinstance(buffer, MockScalar) else buffer.result
            )
            store_op = affine_d.AffineStoreOp(
                val.result, target, [], affine_attr, ip=ctx.get_ip()
            )
            store_op.attributes["to"] = StringAttr.get(node.id)
            return store_op
        if node.id in ctx.buffers:
            return ctx.buffers[node.id]
        if node.id in ctx.global_vars:
            return MockConstant(ctx.global_vars[node.id], ctx)
        raise RuntimeError("Unsupported Name")

    @staticmethod
    def build_Constant(ctx, node):
        return MockConstant(node.value, ctx)

    @staticmethod
    def build_shaped_type(ctx, dtype, shape):
        if len(shape) == 0:
            return dtype.build()
        if not ctx.enable_tensor:
            return MemRefType.get(shape, dtype.build())
        return RankedTensorType.get(shape, dtype.build())

    @staticmethod
    def build_array(ctx, dtype, shape):
        if not ctx.enable_tensor:
            memref_type = MemRefType.get(shape, dtype.build())
            return memref_d.AllocOp(memref_type, [], [], ip=ctx.get_ip())
        return tensor_d.EmptyOp(shape, dtype.build(), ip=ctx.get_ip())

    @staticmethod
    def attach_op_name(ctx, node, op, name, postfix=""):
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
    def build_Attribute(ctx, node):
        value = build_stmt(ctx, node.value)

        if node.attr == "T":  # transpose
            shape = node.shape
            alloc_op = ASTTransformer.build_array(ctx, node.dtype, shape)
            transpose_op = linalg_d.TransposeOp(
                inputs=[value.result],
                outputs=[alloc_op.result],
                permutation=list(range(len(shape)))[::-1],
                ip=ctx.get_ip(),
            )
            ASTTransformer.attach_op_name(ctx, node, transpose_op, "transpose")
            return transpose_op if ctx.enable_tensor else alloc_op

        if node.attr == "reverse":
            return hcl_d.BitReverseOp(value.result, ip=ctx.get_ip())

        if node.attr == "copy":
            return ASTTransformer.build_library_op(
                ctx, node=node, attr="copy", new_args=[value]
            )
        raise RuntimeError("Unsupported Attribute")

    @staticmethod
    def build_Index(ctx, node):
        return build_stmt(ctx, node.value)

    @staticmethod
    def build_Tuple(ctx, node):
        return build_stmts(ctx, node.elts)

    @staticmethod
    def build_all_for(ctx, node, attr):
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
        iter_args = node.iter.args
        if attr in {"grid", "reduction"}:
            grid = [ASTResolver.resolve_constant(x, ctx) for x in iter_args]
            for_loops = build_for_loops(grid, ctx.get_ip(), names, stage_name)
        elif attr == "range":
            low = (
                0
                if len(iter_args) == 1
                else ASTResolver.resolve_constant(iter_args[0], ctx)
            )
            high = (
                ASTResolver.resolve_constant(iter_args[1], ctx)
                if len(iter_args) > 1
                else ASTResolver.resolve_constant(iter_args[0], ctx)
            )
            step = (
                ASTResolver.resolve_constant(iter_args[2], ctx)
                if len(iter_args) > 2
                else 1
            )
            if stage_name is None:
                stage_name = "S_" + "_".join(names)
            with ctx.get_ip():
                for_loops = [
                    hcl_mlir.make_for(
                        low, high, step=step, name=names[0], stage=stage_name
                    )
                ]
        ivs = [loop.induction_variable for loop in for_loops]
        for name, iv in zip(names, ivs):
            ctx.buffers[name] = MockArg(iv)
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
                ):
                    return ASTTransformer.build_all_for(ctx, node, "range")
                if obj is not None and obj.__name__ in {"grid", "reduction"}:
                    return ASTTransformer.build_all_for(ctx, node, obj.__name__)
            raise RuntimeError("Unsupported for loop")

    # pylint: disable=too-many-branches
    @staticmethod
    def build_cast_op(ctx, op, src_type, res_type):
        # No need to cast
        if type(res_type) is type(src_type) and res_type == src_type:
            return op

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
            # FP to Index is not supported in MLIR
            # (Float, Index): RuntimeError,
            # (Index, Float): RuntimeError,
            # Float <-> Fixed/UFixed
            (Float, Fixed): hcl_d.FloatToFixedOp,
            (Float, UFixed): hcl_d.FloatToFixedOp,
            (Fixed, Float): hcl_d.FixedToFloatOp,
            (UFixed, Float): hcl_d.FixedToFloatOp,
            # Int/UInt <-> Fixed/UFixed
            (Fixed, Int): hcl_d.FixedToIntOp,
            (Fixed, UInt): hcl_d.FixedToIntOp,
            (UFixed, Int): hcl_d.FixedToIntOp,
            (UFixed, UInt): hcl_d.FixedToIntOp,
            (Int, Fixed): hcl_d.IntToFixedOp,
            (Int, UFixed): hcl_d.IntToFixedOp,
            (UInt, Fixed): hcl_d.IntToFixedOp,
            (UInt, UFixed): hcl_d.IntToFixedOp,
            # Fixed/UFixed <-> Fixed/UFixed
            (Fixed, Fixed): hcl_d.FixedToFixedOp,
            (Fixed, UFixed): hcl_d.FixedToFixedOp,
            (UFixed, Fixed): hcl_d.FixedToFixedOp,
            (UFixed, UFixed): hcl_d.FixedToFixedOp,
        }
        if (type(src_type), type(res_type)) in cast_map:
            opcls = cast_map[(type(src_type), type(res_type))]
        elif isinstance(src_type, Float) and isinstance(res_type, Index):
            # FP to Index is not supported in MLIR
            # we need to cast to UInt first, then cast to Index
            op = arith_d.FPToUIOp(IndexType.get(), op.result, ip=ctx.get_ip())
            opcls = arith_d.IndexCastOp  # proceed to build cast to index
        elif isinstance(src_type, Index) and isinstance(res_type, Float):
            op = arith_d.IndexCastOp(
                IntegerType.get_signless(32), op.result, ip=ctx.get_ip()
            )
            opcls = arith_d.SIToFPOp  # proceed to build cast to float
        elif isinstance(src_type, (Int, UInt)) and isinstance(res_type, (Int, UInt)):
            if src_type.bits > res_type.bits:
                opcls = arith_d.TruncIOp
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
            opcls = hcl_d.IntToStructOp
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
            cast_op = opcls(mlir_type, op.result, ip=ctx.get_ip())
            if isinstance(res_type, (UInt, Struct)):
                cast_op.attributes["unsigned"] = UnitAttr.get()
        else:
            mlir_type = res_type.build()
            cast_op = opcls(mlir_type, op.result, ip=ctx.get_ip())
        return cast_op

    @staticmethod
    def build_broadcast_op(ctx, op, dtype, src_shape, dst_shape, dims):
        # No shape checking in this function, since it has been done in
        # type inference pass in infer.py
        if src_shape == dst_shape:
            return op
        if len(src_shape) == 0:
            # Get zero-rank memref for constant
            in_cst = ASTTransformer.build_array(ctx, dtype, tuple())
            with ctx.get_ip():
                # pylint: disable=unexpected-keyword-arg
                fill = linalg_d.fill(op.result, outs=[in_cst.result])
            op = fill.owner if ctx.enable_tensor else in_cst
        # target
        alloc_op = ASTTransformer.build_array(ctx, dtype, dst_shape)
        broadcast_op = linalg_d.BroadcastOp(
            inputs=[op.result],
            outputs=[alloc_op.result],
            dimensions=dims,
            ip=ctx.get_ip(),
        )
        return broadcast_op if ctx.enable_tensor else alloc_op

    @staticmethod
    def build_general_binop(ctx, node, lhs, rhs):
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
        ty_cls = Int if isinstance(node.dtype, Index) else type(node.dtype)
        return opcls[ty_cls](lhs.result, rhs.result, ip=ctx.get_ip())

    @staticmethod
    def build_UnaryOp(ctx, node):
        value = build_stmt(ctx, node.operand)
        if isinstance(node.op, ast.USub):
            value = ASTTransformer.build_cast_op(
                ctx, value, node.operand.dtype, node.dtype
            )
            return arith_d.NegFOp(value.result, ip=ctx.get_ip())
        # ast.UAdd
        return value

    @staticmethod
    def build_BinOp(ctx, node):
        lhs = build_stmt(ctx, node.left)
        rhs = build_stmt(ctx, node.right)
        # Cast lhs and rhs to the same type
        lhs = ASTTransformer.build_cast_op(ctx, lhs, node.left.dtype, node.dtype)
        rhs = ASTTransformer.build_cast_op(ctx, rhs, node.right.dtype, node.dtype)
        lhs = ASTTransformer.build_broadcast_op(
            ctx, lhs, node.dtype, node.left.shape, node.shape, node.dims[0]
        )
        rhs = ASTTransformer.build_broadcast_op(
            ctx, rhs, node.dtype, node.right.shape, node.shape, node.dims[1]
        )
        return ASTTransformer.build_general_binop(ctx, node, lhs, rhs)

    @staticmethod
    def build_indices(ctx, node):
        indices = node.value if isinstance(node, ast.Index) else node
        elts = indices.elts if isinstance(indices, ast.Tuple) else [indices]
        ctx.dim_count = 0
        ctx.affine_vars = []
        new_indices = []
        if not ctx.enable_tensor:
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
            new_indices.append(expr.result)
        return new_indices, False

    @staticmethod
    def build_Assign(ctx, node):
        # Compute RHS
        rhs = build_stmt(ctx, node.value)
        if (
            isinstance(node.value, ast.Call) or len(node.value.shape) > 0
        ) and isinstance(node.targets[0], ast.Name):
            rhs.attributes["name"] = StringAttr.get(node.targets[0].id)
            ctx.buffers[node.targets[0].id] = rhs
            return rhs
        # Store LHS
        rhs = ASTTransformer.build_cast_op(ctx, rhs, node.value.dtype, node.dtype)
        store_op = build_stmt(ctx, node.targets[0], val=rhs)
        # Since tensor operations returns a new tensor, we also need to update the buffer
        if (
            not isinstance(store_op, (MockScalar, MockConstant))
            and len(store_op.results) > 0
            and isinstance(store_op.result.type, RankedTensorType)
        ):
            ctx.buffers[node.targets[0].value.id] = store_op
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
        # pylint: disable=redefined-variable-type
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
                # Note: Variables may be changed inside a loop (or a non-top-down structure),
                # so we cannot safely take its value as a constant
                # e.g.,
                # x = 0
                # for i in range(10):
                #     A[x] = 1
                #     x = x + 1
                ctx.nested_loops == 0
                and node.id in ctx.buffers
                and isinstance(ctx.buffers[node.id], MockScalar)
                and isinstance(ctx.buffers[node.id].dtype, Index)
            ):
                return ASTTransformer.build_affine_expr(ctx, ctx.buffers[node.id].value)
            if node.id in ctx.global_vars and isinstance(ctx.global_vars[node.id], int):
                return ASTTransformer.build_affine_expr(
                    ctx, ast.Constant(ctx.global_vars[node.id])
                )
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
    def build_slices(ctx, node, in_shape):
        # caculate the static offsets, sizes, strides for ExtractSlice and InsertSlice
        slices = node.slice.dims
        static_offsets = []
        static_sizes = []
        static_strides = []
        for index, size in zip(slices, in_shape):
            if isinstance(index, ast.Slice):
                lower = 0 if index.lower is None else build_stmt(ctx, index.lower).val
                upper = (
                    size if index.upper is None else build_stmt(ctx, index.upper).val
                )
                if index.step is None:
                    step = 1
                elif isinstance(index.step, ast.Constant):
                    step = index.step.value
                else:
                    raise RuntimeError("Unsupported step type")
            elif isinstance(index, ast.Index):
                lower = index.value.value
                upper = lower + 1
                step = 1
            if lower < 0 or upper < 0:
                raise RuntimeError("Unsupported negative index")
            if lower > size or upper > size:
                raise RuntimeError("Index out of range")
            if step <= 0:
                raise RuntimeError("Unsupported negative step")
            if step > upper - lower:
                raise RuntimeError("Step larger than range")
            static_offsets.append(lower)
            static_sizes.append((upper - lower) // step)
            static_strides.append(step)
        return static_offsets, static_sizes, static_strides

    @staticmethod
    def build_tensor_access(ctx, node, val=None):
        value = build_stmt(ctx, node.value)
        dtype = RankedTensorType(value.result.type).element_type
        in_shape = RankedTensorType(value.result.type).shape
        if isinstance(node.slice, ast.ExtSlice):
            (
                static_offsets,
                static_sizes,
                static_strides,
            ) = ASTTransformer.build_slices(ctx, node, in_shape)
            result = RankedTensorType.get(static_sizes, dtype)
            # pylint: disable=no-else-return
            if isinstance(node.ctx, ast.Load):
                return tensor_d.ExtractSliceOp(
                    result=result,
                    source=value.result,
                    static_sizes=static_sizes,
                    static_strides=static_strides,
                    static_offsets=static_offsets,
                    offsets=[],
                    sizes=[],
                    strides=[],
                    ip=ctx.get_ip(),
                )
            else:  # ast.Store
                return tensor_d.InsertSliceOp(
                    source=val.result,
                    dest=value.result,
                    static_offsets=static_offsets,
                    static_sizes=static_sizes,
                    static_strides=static_strides,
                    offsets=[],
                    sizes=[],
                    strides=[],
                    ip=ctx.get_ip(),
                )
        if isinstance(node.slice, ast.Index):
            index_exprs, _ = ASTTransformer.build_indices(ctx, node.slice)
            # pylint: disable=no-else-return
            if isinstance(node.ctx, ast.Load):
                return tensor_d.ExtractOp(
                    tensor=value.result,
                    indices=index_exprs,
                    ip=ctx.get_ip(),
                )
            else:  # ast.Store
                return tensor_d.InsertOp(
                    scalar=val.result,
                    dest=value.result,
                    indices=index_exprs,
                    ip=ctx.get_ip(),
                )
        raise RuntimeError("Unsupported load subscript")

    @staticmethod
    def build_memory_access(ctx, node, val=None):
        new_indices, is_affine = ASTTransformer.build_indices(ctx, node.slice)
        value = build_stmt(ctx, node.value)
        if is_affine:
            affine_map = AffineMap.get(
                dim_count=ctx.dim_count, symbol_count=0, exprs=new_indices
            )
            affine_attr = AffineMapAttr.get(affine_map)
            ivs = [ctx.buffers[x].result for x in ctx.affine_vars]
            if isinstance(node.ctx, ast.Load):
                op = affine_d.AffineLoadOp(
                    value.result,
                    ivs,
                    affine_attr,
                    ip=ctx.get_ip(),
                )
            else:  # ast.Store
                op = affine_d.AffineStoreOp(
                    val.result, value.result, ivs, affine_attr, ip=ctx.get_ip()
                )
        else:  # Not affine
            # pylint: disable=else-if-used
            if isinstance(node.ctx, ast.Load):
                # pylint: disable=redefined-variable-type
                op = memref_d.LoadOp(value.result, new_indices, ip=ctx.get_ip())
            else:  # ast.Store
                op = memref_d.StoreOp(
                    val.result,
                    value.result,
                    new_indices,
                    ip=ctx.get_ip(),
                )
        attr = "from" if isinstance(node.ctx, ast.Load) else "to"
        op.attributes[attr] = StringAttr.get(node.value.id)
        return op

    # pylint: disable=inconsistent-return-statements
    @staticmethod
    def build_bit_operation(ctx, node, val=None):
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
        if isinstance(node.slice, ast.Index):
            index = build_stmt(ctx, node.slice)
            # pylint: disable=no-else-return
            if isinstance(node.ctx, ast.Load):
                index = ASTTransformer.build_cast_op(
                    ctx, index, node.slice.dtype, Index()
                )
                return hcl_d.GetIntBitOp(
                    node.dtype.build(),
                    value.result,
                    index.result,
                    ip=ctx.get_ip(),
                )
            else:
                index = ASTTransformer.build_cast_op(
                    ctx, index, node.slice.value.dtype, Index()
                )
                # TODO: Test if rhs is uint1
                set_bit_op = hcl_d.SetIntBitOp(
                    node.value.dtype.build(),
                    value.result,
                    index.result,
                    val.result,
                    ip=ctx.get_ip(),
                )
                # write the updated integer back to the scalar
                node.value.ctx = ast.Store()
                store_op = build_stmt(ctx, node.value, val=set_bit_op)
                return store_op

        if isinstance(node.slice, ast.Slice):
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
            # pylint: disable=no-else-return
            if isinstance(node.ctx, ast.Load):
                return hcl_d.GetIntSliceOp(
                    node.dtype.build(),
                    value.result,
                    upper.result,
                    lower.result,
                    ip=ctx.get_ip(),
                )
            else:  # ast.Store
                set_slice_op = hcl_d.SetIntSliceOp(
                    node.value.dtype.build(),
                    value.result,
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
    def build_Subscript(ctx, node, val=None):
        # pylint: disable=no-else-return
        if len(node.value.shape) > 0 and not ctx.enable_tensor:
            return ASTTransformer.build_memory_access(ctx, node, val=val)
        elif ctx.enable_tensor:
            return ASTTransformer.build_tensor_access(ctx, node, val=val)
        else:  # bit operation
            return ASTTransformer.build_bit_operation(ctx, node, val=val)

    @staticmethod
    def build_AnnAssign(ctx, node):
        shape, dtype = node.shape, node.dtype
        # Compute RHS
        if hasattr(node, "np_values"):
            memref_type = ASTTransformer.build_shaped_type(ctx, dtype, shape)
            rhs = ASTTransformer.build_constant_tensor(ctx, node)
            # pylint: disable=redefined-variable-type
            rhs = memref_d.GetGlobalOp(
                memref_type,
                FlatSymbolRefAttr.get(node.target.id),
                ip=ctx.get_ip(),
            )
            ctx.buffers[node.target.id] = rhs
            return
        # Not constant tensor
        rhs = build_stmt(ctx, node.value)
        if rhs is not None:
            rhs = ASTTransformer.build_cast_op(ctx, rhs, node.value.dtype, node.dtype)
        # Store LHS
        if len(shape) > 0:
            memref_type = ASTTransformer.build_shaped_type(ctx, dtype, shape)
            alloc_op = ASTTransformer.build_array(ctx, dtype, shape)
            alloc_op.attributes["name"] = StringAttr.get(node.target.id)
            with ctx.get_ip():
                if isinstance(rhs, (memref_d.AllocOp, MockArg)):
                    # pylint: disable=unexpected-keyword-arg
                    linalg_op = linalg_d.copy(rhs.result, outs=[alloc_op.result])
                elif rhs is not None:
                    # pylint: disable=unexpected-keyword-arg
                    linalg_op = linalg_d.fill(rhs.result, outs=[alloc_op.result])
                else:
                    linalg_op = alloc_op.result
            ctx.buffers[node.target.id] = (
                linalg_op.owner if ctx.enable_tensor else alloc_op
            )
        else:
            # TODO: figure out why zero-ranked cannot work
            ctx.buffers[node.target.id] = MockScalar(
                node.target.id,
                node.dtype,
                ctx,
                value=node.value,
            )
            if rhs is not None:
                build_stmt(ctx, node.target, val=rhs)

    @staticmethod
    def build_FunctionDef(ctx, node):
        if ctx.top_func is not None:
            # Nested function def
            # Create a new context to avoid name collision
            old_ctx = ctx
            ctx = ASTContext(
                global_vars=ctx.global_vars,
                mlir_ctx=old_ctx.mlir_ctx,
                enable_tensor=ctx.enable_tensor,
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
                ASTTransformer.build_shaped_type(ctx, arg.dtype, arg.shape)
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
                ASTTransformer.build_shaped_type(
                    ctx, node.returns.dtype, node.returns.shape
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
            attr = hcl_d.IntegerSetAttr.get(if_cond_set)
            return attr, ctx.buffers[node.left.id]
        else:
            lhs = build_stmt(ctx, node.left)
            rhs = build_stmt(ctx, node.comparators[0])
            # Cast lhs and rhs to the same type
            lhs = ASTTransformer.build_cast_op(ctx, lhs, node.left.dtype, node.dtype)
            rhs = ASTTransformer.build_cast_op(
                ctx, rhs, node.comparators[0].dtype, node.dtype
            )
            # avoid rebuilding the same op
            rhs_res = rhs.result
            dtype = str(rhs_res.type)
            if dtype.startswith("i") or dtype.startswith("ui"):
                op = ATTR_MAP["int" if dtype.startswith("i") else "uint"][
                    type(node.ops[0])
                ]
                predicate = IntegerAttr.get(IntegerType.get_signless(64), op)
                return arith_d.CmpIOp(predicate, lhs.result, rhs_res, ip=ctx.get_ip())
            if dtype.startswith("!hcl.Fixed") or dtype.startswith("!hcl.UFixed"):
                op = ATTR_MAP["fixed" if dtype.startswith("f") else "ufixed"][
                    type(node.ops[0])
                ]
                predicate = IntegerAttr.get(IntegerType.get_signless(64), op)
                return hcl_d.CmpFixedOp(predicate, lhs.result, rhs_res, ip=ctx.get_ip())
            if dtype.startswith("f"):
                op = ATTR_MAP["float"][type(node.ops[0])]
                predicate = IntegerAttr.get(IntegerType.get_signless(64), op)
                return arith_d.CmpFOp(predicate, lhs.result, rhs_res, ip=ctx.get_ip())
            raise RuntimeError(f"Unsupported types for binary op: {dtype}")

    @staticmethod
    def build_BoolOp(ctx, node):
        stmts = build_stmts(ctx, node.values)
        opcls = {
            ast.And: arith_d.AndIOp,
            ast.Or: arith_d.OrIOp,
        }.get(type(node.op))
        return opcls(stmts[0].result, stmts[1].result, ip=ctx.get_ip())

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
    def build_While(ctx, node):
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
            scf_d.ConditionOp(cond.result, [], ip=ctx.get_ip())
            ctx.pop_ip()
            ctx.set_ip(while_op.after.blocks[0])
            build_stmts(ctx, node.body)
            scf_d.YieldOp([], ip=ctx.get_ip())
            ctx.pop_ip()
        return while_op

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
                return build_stmt(ctx, node.func)
            if node.func.id in {"float", "int"}:
                # Python-Builtin functions
                stmts = build_stmts(ctx, node.args)
                return ASTTransformer.build_cast_op(
                    ctx,
                    stmts[0],
                    node.args[0].dtype,
                    Int(32) if node.func.id == "int" else Float(32),
                )
            raise RuntimeError(f"Cannot resolve function `{node.func.id}`")

        if obj.__module__.startswith("allo"):
            # Allo library functions
            new_args = build_stmts(ctx, node.args)
            fn_name = obj.__name__
            arg_type = new_args[0].result.type
            if isinstance(arg_type, (F32Type, IntegerType)):
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
                return opcls(*[x.result for x in new_args], ip=ctx.get_ip())
            if isinstance(arg_type, (MemRefType, RankedTensorType)) and fn_name in {
                "matmul",
                "bmm",
                "softmax",
                "exp",
                "abs",
                "log",
                "add",
                "sub",
                "div",
                "relu",
                "conv2d",
                "maxpool",
                "sumpool",
                "copy",
                "transpose",
                "linear",
            }:
                return ASTTransformer.build_library_op(
                    ctx, node=node, attr=fn_name, new_args=new_args
                )
            raise RuntimeError(f"Unsupported function {fn_name} with type {arg_type}")

        # User-defined subfunction
        func = ctx.global_vars[node.func.id]
        if isinstance(func, func_d.FuncOp):
            # Has already been defined in the top-level scope
            stmts = [func]
        else:
            # Create a new context to avoid name collision
            func_ctx = ASTContext(
                global_vars=ctx.global_vars,
                mlir_ctx=ctx.mlir_ctx,
                enable_tensor=ctx.enable_tensor,
                verbose=ctx.verbose,
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
    def build_library_op(ctx, node, attr, new_args, dtype=None, shape=None):
        assert attr is not None and attr != ""
        ip = ctx.get_ip()
        dtype = dtype if dtype is not None else node.dtype
        shape = shape if shape is not None else node.shape
        with ip:
            alloc_op = ASTTransformer.build_array(ctx, dtype, shape)
            # init zero
            zero = MockConstant(0, ctx)
            zero = ASTTransformer.build_cast_op(ctx, zero, Int(32), node.dtype)
            # pylint: disable=unexpected-keyword-arg
            linalg_fill = linalg_d.fill(zero.result, outs=[alloc_op.result])
            result_tensor = linalg_fill if ctx.enable_tensor else alloc_op
            ASTTransformer.attach_op_name(
                ctx,
                node,
                linalg_fill.owner,
                f"{attr}_init_zero",
                postfix="init_zero",
            )
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
                    new_args[0].result, new_args[1].result, outs=[result_tensor]
                )
                op = op.owner
            elif attr in {"exp", "log", "abs", "copy"}:
                op = {
                    "exp": linalg_d.exp,
                    "log": linalg_d.log,
                    "abs": linalg_d.abs,
                    "copy": linalg_d.copy,
                }.get(attr)(new_args[0].result, outs=[result_tensor])
                op = op.owner
            elif attr == "softmax":
                # TODO: Failed to lower to LLVM, see https://reviews.llvm.org/D153422
                op = linalg_d.SoftmaxOp(
                    input=new_args[0].result,
                    dimension=1,
                    result=[],
                    output=alloc_op,
                )
            elif attr == "relu":
                # TODO: Need to better manage library call
                zero_op = ASTTransformer.build_array(ctx, dtype, shape)
                # init zero
                zero = MockConstant(0, ctx)
                # TODO: support tensor
                # pylint: disable=unexpected-keyword-arg
                linalg_fill = linalg_d.fill(zero.result, outs=[zero_op.result])
                op = linalg_d.max(
                    new_args[0].result, zero_op.result, outs=[result_tensor]
                )
                op = op.owner
            elif attr == "transpose":
                op = linalg_d.TransposeOp(
                    inputs=[new_args[0].result],
                    outputs=[
                        result_tensor if ctx.enable_tensor else result_tensor.result
                    ],
                    permutation=tuple(x.val for x in new_args[1]),
                    ip=ctx.get_ip(),
                )
            elif attr == "linear":  # X @ A.T + B
                permutation = [MockConstant(val, ctx) for val in (1, 0)]
                A_T = ASTTransformer.build_library_op(
                    ctx,
                    node,
                    "transpose",
                    [new_args[1], permutation],
                    shape=node.args[1].shape[::-1],
                )
                matmul = ASTTransformer.build_library_op(
                    ctx, node, "matmul", [new_args[0], A_T]
                )
                bias = ASTTransformer.build_broadcast_op(
                    ctx, new_args[2], node.dtype, node.shape[-1:], node.shape, [0]
                )
                add = ASTTransformer.build_library_op(ctx, node, "add", [matmul, bias])
                return add
            else:
                raise RuntimeError("Unsupported operation")
            ASTTransformer.attach_op_name(ctx, node, op, attr)
        return op if ctx.enable_tensor else result_tensor

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
