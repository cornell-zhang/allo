# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from contextlib import contextmanager
import allo._mlir.extras.types as mlir_types
from allo._mlir.extras.dialects.affine import AffExpr

from allo._mlir.extras.dialects import func as func
from allo._mlir.dialects import (
    allo as allo_d,
    func as func_d,
    memref as memref_d,
    affine as affine_d,
    scf as scf_d,
    arith as arith_d,
)
from allo._mlir.ir import (
    Context,
    Module,
    Location,
    InsertionPoint,
    OpView,
    Value,
    MemRefType,
    ShapedType,
    UnitAttr,
    StringAttr,
    AffineMap,
    AffineMapAttr,
    FlatSymbolRefAttr,
    OpResultList,
    StridedLayoutAttr,
)
from allo.spmw import FunctionType as FuncType
from allo.utils import register_dialect
from allo.memory import Layout
from allo.ir.utils import MockArg, MockCallResultTuple
from .utils import report_error, ErrorMsg, SymbolTable, Scope
from .builtin import BUILTIN_HANDLERS
from .config import _INTERFACE_CONFIG


class IRBuilder(ast.NodeVisitor):
    @contextmanager
    def block_scope(self):
        self.scopes.append(Scope())
        try:
            yield
        finally:
            self.scopes.pop()

    def __init__(self, symbol_table: SymbolTable):
        super().__init__()
        self.symbol_table: SymbolTable = symbol_table
        self.scopes: list[Scope] = []
        self.ctx: Context = Context()
        register_dialect(self.ctx)
        self.module: Module = None

        self.current_func: func_d.FuncOp = None  # the function under construction
        self.func_name: str = None
        self.reserved_bindings = {}

        self.ip_stack = []  # module insert pointes
        self.handler_cache = {}

        self.global_symbols = {}

        # error reporting
        self.err: ErrorMsg = None

    def get_builtin_handler(self, name):
        if name not in self.handler_cache:
            if name not in BUILTIN_HANDLERS:
                return None
            self.handler_cache[name] = BUILTIN_HANDLERS[name](self)
        return self.handler_cache[name]

    def visit(self, node):
        """
        Visit a node.

        [NOTE]: avoid missing any case
        """
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        assert visitor is not None, f"{method} not found"

        try:
            loc = Location.file(
                self.symbol_table.functions[self.func_name]._source,
                node.lineno,
                node.col_offset,
            )
        except:
            loc = Location.unknown()
        try:
            with loc:
                return visitor(node)
        except Exception as e:
            if not getattr(e, "_reported", False) and self.func_name is not None:
                source_file = self.symbol_table.functions[self.func_name]._source
                self.err = ErrorMsg(e, node, source_file=source_file)
                e._reported = True
            raise

    def set_ip(self, ip):
        if not isinstance(ip, InsertionPoint):
            ip = InsertionPoint(ip)
        self.ip_stack.append(ip)

    def get_ip(self):
        return self.ip_stack[-1]

    def get_global_ip(self):
        return self.ip_stack[0]

    def pop_ip(self):
        return self.ip_stack.pop()

    def put_var(self, name, val):
        self.scopes[-1].vars[name] = val

    def get_symbol(self, name, allow_missing=False):
        if name in self.reserved_bindings:
            return self.reserved_bindings[name]
        for scope in reversed(self.scopes):
            if name in scope.vars:
                return scope.vars[name]
            if name in scope.consts:
                return scope.consts[name]
        # global constant
        if name in self.symbol_table.constants:
            global_op = self.global_symbols[name]  # memref_d。GlobalOp
            const_tensor = memref_d.GetGlobalOp(
                global_op.type_.value,
                FlatSymbolRefAttr.get(name),
                ip=self.get_ip(),
            )
            self.put_var(name, const_tensor)
            return const_tensor
        if allow_missing:
            return None
        raise RuntimeError("unreachable")

    def get_op_result(self, val):
        if isinstance(val, OpView):
            if isinstance(val.result, OpResultList):
                assert len(val.result) == 1
                return val.result[0]
            return val.result
        if isinstance(val, MockArg):
            return val.result
        if isinstance(val, MockCallResultTuple):
            return val
        assert isinstance(val, Value), f"Fail to resolve op result: {val}"
        return val

    def build(self):
        try:
            with self.ctx, Location.unknown():
                self.module = Module.create()
                self.set_ip(self.module.body)
                # set up global operations
                for op in self.symbol_table.global_ops:
                    self.visit(op)
                for name, func_node in self.symbol_table.functions.items():
                    self.func_name = name
                    self.visit(func_node)
                    self.func_name = None
                self.pop_ip()
                return self.module
        except:
            if self.err is not None:
                report_error(self.err)
            raise

    def parse_type_ann(self, annotation: ast.Subscript):
        assert (
            isinstance(annotation.slice, ast.Tuple) and len(annotation.slice.elts) == 3
        )  # by construction
        dtype = annotation.slice.elts[0]
        shape = annotation.slice.elts[1]
        assert isinstance(dtype, ast.Name) and isinstance(shape, ast.Tuple)
        spec = annotation.slice.elts[2]
        assert isinstance(spec, ast.Name)
        allo_type = self.symbol_table.types[dtype.id]
        shape = [int(size.value) for size in shape.elts]
        spec = None if spec.id == "None" else self.symbol_table.types[spec.id]
        return allo_type, shape, spec, allo_type.type_hint()

    def build_type(self, annotation: ast.Subscript, force_memref: bool = False):
        """
        build type from annotation

        Args:
            annotation
            force_memref: if True, return memref type

        Returns:
            type, type_hint # FIXME: find a better way to handle unsigned
        """
        dtype, shape, _, type_hint = self.parse_type_ann(annotation)
        if len(shape) == 0 and not force_memref:
            return dtype.build(), type_hint
        return MemRefType.get(shape, dtype.build()), type_hint

    def build_buffer(self, memref_type: MemRefType, type_hint: str):
        buffer = memref_d.AllocOp(memref_type, [], [], ip=self.get_ip())
        buffer.attributes[type_hint] = UnitAttr.get()
        return buffer

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            var = self.get_op_result(self.get_symbol(node.id))
            if (
                isinstance(getattr(var, "type", None), MemRefType)
                and len(var.type.shape) == 0
            ):
                # load scalar from memref
                affine_map = AffineMap.get_identity(0)
                affine_attr = AffineMapAttr.get(affine_map)
                var = affine_d.AffineLoadOp(
                    var.type.element_type, var, [], affine_attr, ip=self.get_ip()
                )
            return var
        raise RuntimeError("unreachable")

    def visit_Constant(self, node: ast.Constant):
        if type(node.value) is int:
            return arith_d.ConstantOp(mlir_types.index(), node.value, ip=self.get_ip())
        if type(node.value) is bool:
            return arith_d.ConstantOp(mlir_types.i(1), node.value, ip=self.get_ip())
        raise NotImplementedError

    def get_affine_expr(self, node: ast.expr, ivs: list, symbols: list):
        """
        Parse an expression into an affine expression.

        [NOTE]: not suppose to build operations in the function, useless you think having some extra unused values are acceptable.
        """
        if isinstance(node, ast.Constant):
            return AffExpr.constant(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.reserved_bindings:
                var = self.reserved_bindings[node.id]
                symbols.append(var)
                return AffExpr.symbol(len(symbols) - 1)
            var = self.get_symbol(node.id)
            if isinstance(var, MockArg) and var.is_affine:
                ivs.append(var.result)
                return AffExpr.dim(len(ivs) - 1)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                # builtin
                if node.func.value.id == _INTERFACE_CONFIG.builtin:
                    handler = self.get_builtin_handler(node.func.attr)
                    if handler:
                        return handler.get_affine_expr(node, ivs, symbols)
        # TODO: other cases
        return None

    def get_affine_attr(self, node: ast.expr):
        ivs, symbols = [], []
        expr = self.get_affine_expr(node, ivs, symbols)
        if expr is None:
            return None, None
        return (
            AffineMap.get(dim_count=len(ivs), symbol_count=len(symbols), exprs=[expr]),
            ivs + symbols,
        )

    def visit_Subscript(self, node: ast.Subscript, val=None):
        base = self.get_op_result(self.visit(node.value))
        if isinstance(base, MockCallResultTuple):
            # [NOTE] special case handling for function call with multiple return
            return base.results[node.slice.value]
        assert isinstance(base.type, MemRefType)
        shape: list[int] = base.type.shape  # tensor shape
        layout = base.type.layout
        elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        offsets, sizes, strides = [], [], []
        indices, ivs, symbols = [], [], []
        # try to parse elts to affine expressions (https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
        use_affine = True
        for elt in elts:
            aff = self.get_affine_expr(elt, ivs, symbols)
            if aff is not None:
                indices.append(aff)
                if isinstance(elt, ast.Constant):  # constant value
                    offsets.append(int(elt.value))
                else:
                    offsets.append(ShapedType.get_dynamic_stride_or_offset())
                sizes.append(1)
                strides.append(1)
            elif isinstance(elt, ast.Slice):  # getting (static) slice
                lower, upper, step = elt.lower.value, elt.upper.value, elt.step.value
                offsets.append(lower)
                sizes.append((upper - lower) // step)
                strides.append(step)
            else:
                use_affine = False
                # placeholder, so we can use len(indices) to check if is element access
                indices.append(None)
                offsets.append(ShapedType.get_dynamic_stride_or_offset())
                sizes.append(1)
                strides.append(1)
        if len(indices) == len(shape):  # access element
            if use_affine:  # affine operations
                affine_map = AffineMap.get(
                    dim_count=len(ivs), symbol_count=len(symbols), exprs=indices
                )
                affine_attr = AffineMapAttr.get(affine_map)
                if isinstance(node.ctx, ast.Load):
                    op = affine_d.AffineLoadOp(
                        base.type.element_type,
                        base,
                        ivs + symbols,
                        affine_attr,
                        ip=self.get_ip(),
                    )
                    return op
                else:  # ast.Store
                    op = affine_d.AffineStoreOp(
                        val,
                        base,
                        ivs + symbols,
                        affine_attr,
                        ip=self.get_ip(),
                    )
                    return None
            else:  # memref operaitons
                indices = [self.get_op_result(self.visit(elt)) for elt in elts]
                if isinstance(node.ctx, ast.Load):
                    op = memref_d.LoadOp(base, indices, ip=self.get_ip())
                    return op
                else:  # ast.Store
                    op = memref_d.StoreOp(val, base, indices, ip=self.get_ip())
                    return None
        else:  # access slice
            # TODO: support hybrid slice
            dynamic_offset = []
            for elt, offset_ in zip(elts, offsets):
                if offset_ < 0:
                    dynamic_offset.append(self.get_op_result(self.visit(elt)))
            sizes.extend(shape[len(offsets) :])
            strides.extend([1] * (len(shape) - len(offsets)))
            offsets.extend([0] * (len(shape) - len(offsets)))
            if isinstance(layout, StridedLayoutAttr):
                orig_offset = layout.offset
                orig_strides = layout.strides
            elif isinstance(layout, AffineMapAttr):
                orig_offset = 0
                orig_strides = [1]
                for i in reversed(shape[1:]):
                    orig_strides.insert(0, orig_strides[0] * i)
            else:
                raise RuntimeError(f"Unsupported layout type {type(layout)}")
            result_sizes = []
            stride_attr = []
            for idx_, size in enumerate(sizes):
                if size > 1:
                    result_sizes.append(size)
                    stride_attr.append(strides[idx_] * orig_strides[idx_])
            if len(dynamic_offset) > 0 or orig_offset < 0:
                offset_attr = ShapedType.get_dynamic_stride_or_offset()
            else:
                offset_attr = orig_offset + sum(
                    o * s for o, s in zip(offsets, orig_strides)
                )
            result = MemRefType.get(
                shape=result_sizes,
                element_type=base.type.element_type,
                # relative to the base memref
                layout=StridedLayoutAttr.get(offset=offset_attr, strides=stride_attr),
            )
            subview = memref_d.SubViewOp(
                source=base,
                result=result,
                static_offsets=offsets,
                static_sizes=sizes,
                static_strides=strides,
                offsets=dynamic_offset,
                sizes=[],
                strides=[],
                ip=self.get_ip(),
            )
            if isinstance(node.ctx, ast.Load):
                return subview
            else:
                return memref_d.CopyOp(val, subview.result, ip=self.get_ip())

    def visit_BoolOp(self, node: ast.BoolOp):
        opcls = {
            ast.And: arith_d.AndIOp,
            ast.Or: arith_d.OrIOp,
        }.get(type(node.op))
        result = opcls(
            self.get_op_result(self.visit(node.values[0])),
            self.get_op_result(self.visit(node.values[1])),
            ip=self.get_ip(),
        )
        for i in range(2, len(node.values)):
            result = opcls(
                result.result,
                self.get_op_result(self.visit(node.values[i])),
                ip=self.get_ip(),
            )
        return result

    def visit_Assign(self, node: ast.Assign):
        # [NOTE]: only used for special case
        assert isinstance(node.value, ast.Call)
        # - some special builtin: get_wid
        if (
            isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == _INTERFACE_CONFIG.builtin
        ):
            name = node.value.func.attr
            assert name in BUILTIN_HANDLERS
            handler = self.get_builtin_handler(name)
            handler.build(node.value, node.targets)
            return
        # - call a function with multiple returns
        assert len(node.targets) == 1
        call_op = self.visit(node.value)
        assert self.get_symbol(name=node.targets[0].id, allow_missing=True) is None
        self.put_var(node.targets[0].id, val=MockCallResultTuple(call_op.results))

    def visit_AnnAssign(self, node: ast.AnnAssign):
        value = (
            None if node.value is None else self.get_op_result(self.visit(node.value))
        )
        if isinstance(node.target, ast.Name):
            target = self.get_symbol(name=node.target.id, allow_missing=True)
            if target is None:
                # declare new variable
                alloc_op = self.build_buffer(
                    *self.build_type(node.annotation, force_memref=True)
                )
                alloc_op.attributes["name"] = StringAttr.get(node.target.id)
                self.put_var(node.target.id, val=alloc_op)
                target = alloc_op
        elif isinstance(node.target, ast.Subscript):
            self.visit_Subscript(node.target, val=value)
            return
        else:
            # FIXME: unreachable?
            target = self.visit(node.target)
        if value is None:
            return
        target = self.get_op_result(target)
        if isinstance(value.type, MemRefType):
            # tensor
            memref_d.CopyOp(value, target, ip=self.get_ip())
        else:
            # scalar
            affine_map = AffineMap.get(dim_count=0, symbol_count=0, exprs=[])
            affine_d.AffineStoreOp(
                value, target, [], AffineMapAttr.get(affine_map), ip=self.get_ip()
            )

    def visit_Expr(self, node: ast.Expr):
        return self.visit(node.value)

    def visit_For(self, node: ast.For):
        # TODO: should use higher-level affine loop if possible
        # TODO: handle `type_comment`
        args = node.iter.args
        lb, lb_bound_ivs = self.get_affine_attr(args[0])
        ub, ub_bound_ivs = self.get_affine_attr(args[1])
        use_affine_loop = (
            lb is not None and ub is not None and isinstance(args[2], ast.Constant)
        )
        if use_affine_loop:
            step = int(args[2].value)
            for_op = affine_d.AffineForOp(
                lower_bound=lb,
                upper_bound=ub,
                step=step,
                iter_args=[],
                lower_bound_operands=lb_bound_ivs,
                upper_bound_operands=ub_bound_ivs,
                ip=self.get_ip(),
            )
            if node.type_comment is not None:
                for_op.attributes["loop_type"] = StringAttr.get(node.type_comment)
            affine_d.AffineYieldOp([], ip=InsertionPoint(for_op.body))
        else:
            assert node.type_comment != "unroll"
            lb = self.get_op_result(self.visit(args[0]))
            rb = self.get_op_result(self.visit(args[1]))
            step = self.get_op_result(self.visit(args[2]))
            for_op = scf_d.ForOp(lb, rb, step, ip=self.get_ip())
            scf_d.YieldOp([], ip=InsertionPoint(for_op.body))

        with self.block_scope():
            self.put_var(
                name=node.target.id,
                val=MockArg(for_op.induction_variable, is_affine=use_affine_loop),
            )
            self.set_ip(for_op.body.operations[0])
            for stmt in node.body:
                self.visit(stmt)
            self.pop_ip()
        return

    def visit_While(self, node: ast.While):
        while_op = scf_d.WhileOp([], [], ip=self.get_ip())
        while_op.before.blocks.append(*[])
        while_op.after.blocks.append(*[])
        self.set_ip(while_op.before.blocks[0])
        cond = self.get_op_result(self.visit(node.test))
        scf_d.ConditionOp(cond, [], ip=self.get_ip())
        self.pop_ip()
        self.set_ip(while_op.after.blocks[0])
        with self.block_scope():
            for stmt in node.body:
                self.visit(stmt)
            scf_d.YieldOp([], ip=self.get_ip())
        self.pop_ip()
        return while_op

    def visit_If(self, node: ast.If):
        # TODO: should use higher-level affine operation if possible
        if isinstance(node.test, ast.Constant):  # simple DCE
            # [NOTE]: do not eliminate the branch on AST, so we can keep the original scoping
            if node.test.value:
                with self.block_scope():
                    for stmt in node.body:
                        self.visit(stmt)
            else:
                with self.block_scope():
                    for stmt in node.orelse:
                        self.visit(stmt)
            return
        if_op = scf_d.IfOp(
            self.get_op_result(self.visit(node.test)),
            ip=self.get_ip(),
            has_else=len(node.orelse),
        )
        self.set_ip(if_op.then_block)
        with self.block_scope():
            for stmt in node.body:
                self.visit(stmt)
            scf_d.YieldOp([], ip=self.get_ip())
        self.pop_ip()
        if len(node.orelse) > 0:
            else_block = if_op.elseRegion.blocks[0]
            self.set_ip(else_block)
            with self.block_scope():
                for stmt in node.orelse:
                    self.visit(stmt)
                scf_d.YieldOp([], ip=self.get_ip())
            self.pop_ip()

    def visit_IfExp(self, node: ast.IfExp):
        raise NotImplementedError

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            func_d.ReturnOp([], ip=self.get_ip())
            return
        values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
        rets = []
        for idx, value in enumerate(values):
            ret = self.get_op_result(self.visit(value))
            if (
                isinstance(ret.type, MemRefType)
                and ret.type != self.current_func.type.results[idx]
            ):  # mlir has strict type checking, `memref<32xi32, strided<[1]>>` != `memref<32xi32>`
                # FIXME: return unsigned?
                alloc_op = self.build_buffer(
                    self.current_func.type.results[idx], "signed"
                )
                memref_d.CopyOp(ret, alloc_op.result, ip=self.get_ip())
                ret = alloc_op.result
            rets.append(ret)
        func_d.ReturnOp(rets, ip=self.get_ip())

    def visit_Pass(self, node: ast.Pass):
        return None

    def visit_With(self, node: ast.With):
        raise NotImplementedError

    def visit_work(self, callee: ast.FunctionDef):
        callee_name = callee.name
        grid, top_args = None, None
        for kw in callee.decorator_list[0].keywords:
            if kw.arg == "grid":
                grid = [c.value for c in kw.value.elts]
            elif kw.arg == "args":
                top_args = [(self.get_op_result(self.visit(e))) for e in kw.value.elts]
        shardings, grid_args = [], []
        call_args = []
        assert len(top_args) == len(callee.args.args)
        for arg, top_arg in zip(callee.args.args, top_args):
            dtype, shape, spec, type_hint = self.parse_type_ann(arg.annotation)
            if isinstance(spec, Layout):
                call_args.append(len(grid_args))
                grid_args.append(top_arg)
                shardings.append(
                    [
                        p.axis if isinstance(p, Layout.Shard) else -1
                        for p in spec.partitions
                    ]
                )
            else:
                call_args.append(top_arg)
        with self.get_ip():
            op = allo_d.GridMapOp(grid_args, shardings, grid)
            block = op.block
        block_args = list(block.arguments)
        for i in range(len(call_args)):
            if isinstance(call_args[i], int):
                call_args[i] = block_args[call_args[i]]
        func_d.CallOp(
            [],
            FlatSymbolRefAttr.get(callee_name),
            call_args,
            ip=InsertionPoint.at_block_begin(block),
        )

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            if node.func.value.id == _INTERFACE_CONFIG.builtin:
                # handling for builtins
                name = node.func.attr
                assert name in BUILTIN_HANDLERS
                handler = self.get_builtin_handler(name)
                return handler.build(node)
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
            callee = self.symbol_table.functions[callee_name]
            if callee._type in {FuncType.WORK}:
                self.visit_work(callee)
                return None
            rets = (
                callee.returns.elts
                if isinstance(callee.returns, ast.Tuple)
                else [callee.returns] if callee.returns is not None else []
            )
            call_op = func_d.CallOp(
                [self.build_type(ret)[0] for ret in rets],
                FlatSymbolRefAttr.get(callee_name),
                [self.get_op_result(self.visit(arg)) for arg in node.args],
                ip=self.get_ip(),
            )
            return call_op
        raise NotImplementedError

    def visit_FunctionDef(self, node: ast.FunctionDef):
        input_types, input_hints = [], []
        for arg in node.args.args:
            in_type, hint = self.build_type(arg.annotation)
            input_types.append(in_type)
            input_hints.append(hint[0])
        output_types, output_hints = [], []
        if node.returns:
            rets = (
                node.returns.elts
                if isinstance(node.returns, ast.Tuple)
                else [node.returns]
            )

            for ret in rets:
                out_type, hint = self.build_type(ret)
                output_types.append(out_type)
                output_hints.append(hint[0])
        # Build function
        with self.get_ip():
            func_op = func.function(
                node.name,
                input_types,
                output_types,
                itype_hints=input_hints,
                otype_hints=output_hints,
            )
        self.current_func = func_op
        self.reserved_bindings.clear()
        with self.block_scope():
            # function arguments
            for i, (ast_arg, arg) in enumerate(zip(node.args.args, func_op.arguments)):
                mock_arg = MockArg(arg, is_affine=False, idx=i)
                self.put_var(name=ast_arg.arg, val=mock_arg)
            self.set_ip(func_op.entry_block)
            for stmt in node.body:
                self.visit(stmt)
            self.pop_ip()
