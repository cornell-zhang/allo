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
    FloatAttr,
    StringAttr,
    AffineExpr,
    AffineConstantExpr,
    AffineMap,
    AffineMapAttr,
    IntegerSet,
    FlatSymbolRefAttr,
    DenseElementsAttr,
    TypeAttr,
    UnrankedTensorType,
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
from .transform import build_for_loops
from .utils import (
    MockArg,
    MockScalar,
    MockConstant,
    MockBuffer,
    get_extra_type_hints,
    get_kwarg,
)
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

    @staticmethod
    def build_general_binop(ctx, node, lhs, rhs):
        opcls = {
            ast.Add: {
                "float": arith_d.AddFOp,
                "int": arith_d.AddIOp,
                "fixed": hcl_d.AddFixedOp,
            },
            ast.Sub: {
                "float": arith_d.SubFOp,
                "int": arith_d.SubIOp,
                "fixed": hcl_d.SubFixedOp,
            },
            ast.Mult: {
                "float": arith_d.MulFOp,
                "int": arith_d.MulIOp,
                "fixed": hcl_d.MulFixedOp,
            },
            ast.Div: {
                "float": arith_d.DivFOp,
                "int": arith_d.DivSIOp,
                "uint": arith_d.DivUIOp,
                "fixed": hcl_d.DivFixedOp,
            },
            ast.FloorDiv: {
                "float": RuntimeError,
                "int": arith_d.FloorDivSIOp,
                "uint": RuntimeError,
            },
            ast.Mod: {
                "float": arith_d.RemFOp,
                "int": arith_d.RemSIOp,
                "uint": arith_d.RemUIOp,
            },
            ast.Pow: {
                "float": math_d.PowFOp,
                "int": RuntimeError,
                "uint": RuntimeError,
            },
            ast.LShift: {
                "float": RuntimeError,
                "int": arith_d.ShLIOp,
                "uint": RuntimeError,
            },
            ast.RShift: {
                "float": RuntimeError,
                "int": arith_d.ShRUIOp,
                "uint": RuntimeError,
            },
            ast.BitOr: {
                "float": RuntimeError,
                "int": arith_d.OrIOp,
                "uint": RuntimeError,
            },
            ast.BitXor: {
                "float": RuntimeError,
                "int": arith_d.XOrIOp,
                "uint": RuntimeError,
            },
            ast.BitAnd: {
                "float": RuntimeError,
                "int": arith_d.AndIOp,
                "uint": RuntimeError,
            },
        }.get(type(node.op))
        dtype = str(node.dtype.build())
        # FIXME: workaround to get the type
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
        if dtype.startswith("i"):
            op = opcls["int"]
        elif dtype.startswith("fixed"):
            op = opcls["fixed"]
        elif dtype.startswith("f"):
            op = opcls["float"]
        else:
            raise RuntimeError(f"Unsupported types for binary op: {dtype}")
        return op(lhs.result, rhs.result, ip=ctx.get_ip())

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
        return ASTTransformer.build_general_binop(ctx, node, lhs, rhs)

    @staticmethod
    def build_store(ctx, node, val):
        ip = ctx.get_ip()
        if isinstance(node, ast.Subscript):
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
                val.result, target, ivs, affine_attr, ip=ip
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
                val.result, target, [], affine_attr, ip=ip
            )
            store_op.attributes["to"] = StringAttr.get(node.id)
            return store_op
        raise RuntimeError("Unsupported store")

    @staticmethod
    def build_Assign(ctx, node):
        # Compute RHS
        if isinstance(node.value, ast.Name):  # scalar
            rhs = ctx.buffers[node.value.id]
        else:
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
        # Aug LHS
        res = ASTTransformer.build_general_binop(ctx, node, lhs, rhs)
        # Store LHS
        store_op = ASTTransformer.build_store(ctx, node.target, res)
        return store_op

    @staticmethod
    def build_affine_expr(ctx, node):
        # pylint: disable=no-else-return
        if isinstance(node, ast.Name):
            if (
                node.id in ctx.buffers
                and isinstance(ctx.buffers[node.id], MockArg)
                and str(ctx.buffers[node.id].result.type) == "index"
            ):
                ctx.dim_count += 1
                ctx.affine_vars.append(node.id)
                return AffineExpr.get_dim(ctx.dim_count - 1)
            else:
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
        # Load op
        ctx.dim_count = 0
        ctx.affine_vars = []
        index_exprs = []
        # pylint: disable=redefined-builtin
        slice = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
        elts = slice.elts if isinstance(slice, ast.Tuple) else [slice]
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
                    ctx.buffers[node.value.id].result, ivs, affine_attr, ip=ctx.get_ip()
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

    @staticmethod
    def build_AnnAssign(ctx, node):
        ip = ctx.get_ip()
        type_hint = node.annotation
        if node.value is not None:
            if isinstance(node.value, ast.Name) and node.value.id in ctx.buffers:
                rhs = ctx.buffers[node.value.id]
            elif isinstance(node.value, (ast.List, ast.Name)):
                rhs = ASTTransformer.build_constant_tensor(ctx, node)
            elif isinstance(node.value, ast.Constant):
                rhs = build_stmt(ctx, node.value)
            else:
                raise RuntimeError("Unsupported data type")
        else:
            rhs = None
        if isinstance(type_hint, ast.Subscript):
            type_str = type_hint.value.id
            if type_str in ctx.global_vars:
                type_str = str(ctx.global_vars[type_str])
            # pylint: disable=redefined-builtin
            slice = (
                type_hint.slice.value
                if isinstance(type_hint.slice, ast.Index)
                else type_hint.slice
            )
            elts = slice.elts if isinstance(slice, ast.Tuple) else [slice]
            shape, dtype = node.shape, node.dtype
            if not ctx.enable_tensor:
                memref_type = build_shaped_type(dtype, shape, False)
                if isinstance(node.value, ast.Name) and node.value.id in ctx.buffers:
                    if isinstance(rhs, (memref_d.AllocOp, MockArg)):
                        alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ip)
                        alloc_op.attributes["name"] = StringAttr.get(node.target.id)
                        ctx.buffers[node.target.id] = alloc_op
                        with ip:
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
                    alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ip)
                    alloc_op.attributes["name"] = StringAttr.get(node.target.id)
                    ctx.buffers[node.target.id] = alloc_op
                    if rhs is not None:
                        with ip:
                            # pylint: disable=unexpected-keyword-arg
                            linalg_d.fill(rhs.result, outs=[alloc_op.result])
                else:
                    raise RuntimeError("Unsupported data type")
            else:
                tensor_type = build_shaped_type(dtype, shape, True)
                tensorgen_op = tensor_d.GenerateOp(tensor_type, [], ip=ip)
                index_type = []
                for _ in elts:
                    index_type.append(IndexType.get())
                ctx.set_ip(tensorgen_op.regions[0].blocks.append(*index_type))
                ip = ctx.get_ip()
                tensor_d.YieldOp(rhs.result, ip=ip)
                ip = ctx.pop_ip()
                ctx.buffers[node.target.id] = tensorgen_op
        elif isinstance(type_hint, ast.Name):
            type_str = type_hint.id
            if type_str in ctx.global_vars:
                type_str = str(ctx.global_vars[type_str])
            # TODO: figure out why zero-shape cannot work
            ctx.buffers[node.target.id] = MockScalar(node.target.id, type_str, ctx)
            if rhs is not None:
                ASTTransformer.build_store(ctx, node.target, rhs)
        else:
            raise RuntimeError("Unsupported AnnAssign")

    @staticmethod
    def build_FunctionDef(ctx, node):
        if ctx.top_func is not None:
            # Nested function def
            # Create a new context to avoid name collision
            old_ctx = ctx
            ctx = ASTContext(global_vars=ctx.global_vars, mlir_ctx=old_ctx.mlir_ctx)
            ctx.set_ip(old_ctx.top_func)
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
        ctx.top_func = func_op
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
            # Python-Builtin functions
            assert (
                len(node.args) == 1
            ), "Only support one argument for `float` and `int`"
            if node.func.id == "float":
                if node.args[0].id in ctx.global_vars:
                    return MockConstant(float(ctx.global_vars[node.args[0].id]), ctx)
                # TODO: Support other types
                return arith_d.SIToFPOp(
                    F32Type.get(),
                    ctx.buffers[node.args[0].id].result,
                    ip=ctx.get_ip(),
                )
            if node.func.id == "int":
                return MockConstant(int(ctx.global_vars[node.args[0].id]), ctx)
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
                new_args[0].type, (MemRefType, UnrankedTensorType, RankedTensorType)
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
            func_ctx = ASTContext(global_vars=ctx.global_vars, mlir_ctx=ctx.mlir_ctx)
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
            ASTTransformer.build_init_zero(
                ctx, node=node, op_name=op_name, init_op=alloc_op, dtype=dtype
            )
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
            # TODO: failed to lower to LLVM, see https://reviews.llvm.org/D153422
            elif attr == "softmax":
                op = linalg_d.SoftmaxOp(
                    input=new_args[0],
                    dimension=1,
                    result=[],
                    output=alloc_op,
                )
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
    def build_init_zero(ctx, node, op_name, init_op, dtype):
        # initialize data op
        with ctx.get_ip():
            if str(dtype) == "int32":
                # pylint: disable=unexpected-keyword-arg
                zero = arith_d.ConstantOp(
                    value=IntegerAttr.get(dtype.build(), 0), result=dtype.build()
                ).result
            elif str(dtype) == "float32":
                # pylint: disable=unexpected-keyword-arg
                zero = arith_d.ConstantOp(
                    value=FloatAttr.get(dtype.build(), 0.0), result=dtype.build()
                ).result
            else:
                raise RuntimeError("Unsupported data type")
            # pylint: disable=unexpected-keyword-arg
            linalg_fill = linalg_d.fill(zero, outs=[init_op.result])
            # add op name for init_zero
            if hasattr(node, "keywords") and len(node.keywords) > 0:
                linalg_fill.owner.attributes["op_name"] = StringAttr.get(
                    f"{node.keywords[0].value.value}_init_zero"
                )
            else:
                linalg_fill.owner.attributes["op_name"] = StringAttr.get(
                    f"{op_name}_init_zero_{ctx.unnamed_linalg_op_count}"
                )

    @staticmethod
    def build_Return(ctx, node):
        ret = build_stmt(ctx, node.value)
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
