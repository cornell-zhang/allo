# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import copy
from typing import Union
from collections import deque, ChainMap
from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import sympy
from .utils import get_ast, report_error, ErrorMsg, SymbolTable, Scope, ErrorValue
from .config import _INTERFACE_CONFIG
from allo.spmw import FunctionType
from allo.ir.types import (
    AlloType,
    Float,
    Stream,
    Stateful,
    ConstExpr,
    Index,
    UInt,
    bool as allo_bool,
)
from allo.memory import Layout
from .builtin import BUILTIN_HANDLERS


@dataclass(frozen=True)
class Axes:
    idx: int
    wid: ast.Name


class ASTPreProcessor(ast.NodeTransformer):
    @contextmanager
    def namespace(self, node: ast.FunctionDef):
        name = node.name
        symbols = getattr(node, "template_bindings", {})

        self.current_namespace = name
        self.scopes.append(Scope())
        self.symbols = ChainMap(symbols, self.global_symbols)

        try:
            yield
        finally:
            self.scopes.pop()
            self.current_namespace = None
            self.symbols = self.global_symbols

    @contextmanager
    def function_scope(self, node: ast.FunctionDef):
        symbols_ckpt = self.symbols

        self.symbols = ChainMap(getattr(node, "template_bindings", {}), self.symbols)
        self.current_func = node.name
        self.meta_ops = []
        self.scopes.append(Scope())

        try:
            yield
        finally:
            self.scopes.pop()
            self.symbols = symbols_ckpt
            self.current_func = None
            self.meta_ops = None
            self.meta_cond.clear()

    @contextmanager
    def block_scope(self):
        self.scopes.append(Scope())
        try:
            yield
        finally:
            self.scopes.pop()

    def __init__(
        self,
        symbol_table: SymbolTable,
        global_symbols: dict,
        typing_rule: str = "default",
    ):
        super().__init__()
        self.symbol_table: SymbolTable = symbol_table
        self.global_symbols: dict = global_symbols
        self.typing_rule = typing_rule

        self.worklist: deque[ast.FunctionDef] = deque([])

        self.current_namespace: str = None

        self.current_func: str = None
        self.symbols = global_symbols
        self.work_meta = {}  # per work instance's meta data
        self.meta_ops = None  # operations for meta setup
        # keep track of metaprogramming conditions, allow nesting
        self.meta_cond: list[bool] = []

        self.scopes: list[Scope] = []

        # error reporting
        self.err: ErrorMsg = None

    @staticmethod
    def _copy_loc(new_node: ast.AST, old_node: ast.AST) -> ast.AST:
        """Copy source location from old_node to new_node and fill missing locations in children."""
        if isinstance(new_node, ast.AST) and hasattr(old_node, "lineno"):
            ast.copy_location(new_node, old_node)
            ast.fix_missing_locations(new_node)
        return new_node

    def visit(self, node):
        """
        Visit a node.

        [NOTE]: avoid missing any case
        """
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        assert visitor is not None, f"{method} not found"
        try:
            result = visitor(node)
        except Exception as e:
            if not getattr(e, "_reported", False):
                name = self.current_func or self.current_namespace
                if name is not None:
                    source_file = self.symbol_table.functions[name]._source
                    self.err = ErrorMsg(e, node, source_file=source_file)
                    e._reported = True
            raise
        # propagate source location
        if isinstance(result, list):
            for r in result:
                self._copy_loc(r, node)
        else:
            self._copy_loc(result, node)
        return result

    def put_var(self, name, val):
        assert (
            name not in self.symbol_table.functions
            and name not in self.symbol_table.variables
            and name not in self.symbol_table.constants
        )
        self.scopes[-1].vars[name] = val

    def put_const(self, name, const, is_global: bool = False):
        assert (
            name not in self.symbol_table.functions
            and name not in self.symbol_table.variables
            and name not in self.symbol_table.constants
        )
        if is_global:
            self.symbol_table.constants[name] = const
        else:
            self.scopes[-1].consts[name] = const

    def get_symbol(self, name, allow_missing=False):
        """
        Get the value of a symbol from the current scope chain.

        Args:
            - name (str): The variable name to look up.
            - allow_missing (bool): If True, return None when the symbol
                does not exist. Otherwise, raise an error.
        """
        for i in reversed(range(len(self.scopes))):
            scope = self.scopes[i]
            if name in scope.vars:
                var = scope.vars[name]
                if self.current_namespace is not None and isinstance(var, ast.arg):
                    # self.current_func -> not directly under UNIT (namespace)
                    if self.current_func is not None:
                        name = var.arg
                        work_def = self.symbol_table.functions[self.current_func]
                        assert hasattr(work_def, "arg_kw")  # shared resources
                        for idx, v in enumerate(work_def.arg_kw.value.elts):
                            if v.id == name:
                                return work_def.args.args[idx]
                        # add as work's argument
                        work_def.arg_kw.value.elts.append(
                            ast.Name(id=name, ctx=ast.Load())
                        )
                        arg = copy.deepcopy(var)
                        work_def.args.args.append(arg)
                        self.scopes[i + 1].vars[name] = arg
                        return arg
                return var
            if name in scope.consts:
                return scope.consts[name]
        if name in self.symbol_table.variables:
            return self.symbol_table.variables[name]
        if name in self.symbol_table.constants:
            return self.symbol_table.constants[name]
        if allow_missing:
            return None
        raise ValueError(f"Variable {name} not defined in current scope.")

    def get_consts(self):
        consts = {}
        for scope in self.scopes:
            for k, v in scope.consts.items():
                consts[k] = v.value  # v is ast.Constant
        for k, v in self.symbol_table.constants.items():
            consts[k] = v.value
        return consts

    def process(self, fn: Union[Callable, str], instantiate: list = None):
        """
        Process the input function.

        Args:
            fn: The function to process.
            instantiate: The arguments to instantiate the function. default to None.
        """
        try:
            func: ast.FunctionDef = get_ast(fn)
            # if instantiate is not None, we need to use the args to instantiate the unique function
            node, top_name = self.visit_function_signature(func, instantiate)
            while self.worklist:
                n = self.visit_function_body(
                    self.symbol_table.functions[self.worklist.popleft()]
                )
            return func, top_name
        except:
            if self.err is not None:
                report_error(self.err)
            raise

    def eval_constant(self, node, consts=None):
        """
        Evaluate the constant expression.

        Args:
            node: The node to evaluate.
            consts: The constants to use. default to None. Used to avoid reanalyzing the constants.
        """
        if consts is None:
            consts = self.get_consts()
            consts.update(self.symbols)
        return eval(
            compile(ast.fix_missing_locations(ast.Expression(node)), "", "eval"), consts
        )

    def resolve_node(self, node: ast.AST):
        if isinstance(node, ast.Name):
            return self.symbols[node.id]  # limited to single-level symbol lookup
        if isinstance(node, ast.Call):
            ty_cls = self.symbols[node.func.id]
            consts = self.get_consts()
            consts.update(self.symbols)
            args = [self.eval_constant(arg, consts=consts) for arg in node.args]
            kwargs = {
                kw.arg: self.eval_constant(kw.value, consts=consts)
                for kw in node.keywords
            }
            return ty_cls(*args, **kwargs)
        if isinstance(node, ast.Attribute):
            v = node.value
            chain = [node.attr]
            while isinstance(v, ast.Attribute):
                chain.append(v.attr)
                v = v.value
            if not isinstance(v, ast.Name):
                # Example cases that fall under this branch:
                #   - x[i].attr: ast.Subscript
                #   - (a + b).attr: ast.BinOp
                return None
            chain.append(v.id)
            if chain[-1] not in self.global_symbols:
                return None
            mod = self.global_symbols[chain[-1]]
            for attr in reversed(chain[:-1]):
                try:
                    mod = getattr(mod, attr)
                except AttributeError:
                    return None
            return mod

    def visit_type_annotation(self, annotation: ast.AST):
        """
        Visit the type annotation.

        Returns:
            dtype: data type.
            shape: The shape of the type.
            refinement: type refinement. Stateful, Memory, etc.
        """
        if isinstance(annotation, ast.Name):
            # e.g., A: int32
            return self.resolve_node(annotation), tuple(), None
        if isinstance(annotation, ast.Call):
            # e.g., A: Int(32)
            return self.resolve_node(annotation), tuple(), None
        if isinstance(annotation, ast.Subscript):
            # e.g., a: int32[32], a: Int(32)[32], pipe: Stream[Ty, 4][4]
            base_type, base_shape, _ = self.visit_type_annotation(annotation.value)
            assert len(base_shape) == 0
            if base_type is Stream:
                # e.g., Stream[Ty, 4]
                ty, depth = None, None
                if isinstance(annotation.slice, ast.Tuple):
                    assert len(annotation.slice.elts) == 2
                    ty, depth = annotation.slice.elts[0], annotation.slice.elts[1]
                else:
                    ty = annotation.slice
                    depth = ast.Constant(value=2)  # default depth
                ele_type, ele_shape, _ = self.visit_type_annotation(ty)
                depth = self.eval_constant(depth)
                return (
                    Stream(dtype=ele_type, shape=ele_shape, depth=depth),
                    tuple(),
                    None,
                )
            if base_type is ConstExpr:
                # e.g., a: ConstExpr[int32]
                ele_type, ele_shape, _ = self.visit_type_annotation(annotation.slice)
                assert len(ele_shape) == 0, "ConstExpr only supports scalar types"
                const_dtype = copy.deepcopy(ele_type)
                const_dtype.constexpr = True
                return const_dtype, tuple(), None
            size = annotation.slice
            elts = size.elts if isinstance(size, ast.Tuple) else [size]
            return base_type, tuple(self.eval_constant(x) for x in elts), None
        if isinstance(annotation, ast.BinOp):
            # e.g., B: Int(32) @ Stateful = 0, a: int32[32] @ Memory(resource="URAM")
            assert isinstance(annotation.op, ast.MatMult)
            dtype, shape, spec = self.visit_type_annotation(annotation.left)
            if isinstance(annotation.right, (ast.Name, ast.Call)):
                # 1.    B: Int(32) @ Stateful = 0
                # 2.    mm = Memory(resource="URAM") # defined in 'global' scope
                #       a: int32[32] @ mm
                # a: int32[32] @ Memory(resource="URAM")
                refinement_type = self.resolve_node(annotation.right)
            elif isinstance(annotation.right, ast.List):
                # a: int32[32] @ [S(0)]
                refinement_type = [self.resolve_node(v) for v in annotation.right.elts]
            else:
                raise NotImplementedError
            if refinement_type is Stateful:
                stateful_dtype = copy.deepcopy(dtype)
                stateful_dtype.stateful = True
                return stateful_dtype, shape, spec
            return dtype, shape, refinement_type
        if isinstance(annotation, ast.Constant):  # template
            assert isinstance(annotation.value, str)
            tree = ast.parse(annotation.value)
            return self.visit_type_annotation(tree.body[0].value)
        raise NotImplementedError

    def get_ast_annotaiton(
        self, dtype: AlloType, shape: tuple[int], spec
    ) -> ast.Subscript:
        # TODO: may collect spec in the same way
        dtype_name = str(dtype)
        if dtype_name not in self.symbol_table.types:
            self.symbol_table.types[dtype_name] = dtype
        spec_name = str(spec)
        if spec and spec_name not in self.symbol_table.types:
            self.symbol_table.types[spec_name] = spec
        return ast.Subscript(
            value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
            slice=ast.Tuple(
                elts=[
                    ast.Name(id=dtype_name, ctx=ast.Load()),
                    ast.Tuple(
                        elts=[ast.Constant(value=d) for d in shape],
                        ctx=ast.Load(),
                    ),
                    ast.Name(id=spec_name, ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            ),
            ctx=ast.Load(),
        )

    def reset_type(self, arg: ast.arg, shape=None, spec=None):
        """
        Reset ast argument's shape to a new one. Usually used in sharding.
        """
        if arg.shape != shape:
            arg.shape = shape
            arg.annotation.slice.elts[1] = ast.Tuple(
                elts=[ast.Constant(value=d) for d in shape],
                ctx=ast.Load(),
            )
        if getattr(arg, "spec", None) != spec:
            arg.spec = spec
            spec_name = str(spec)
            if spec and spec_name not in self.symbol_table.types:
                self.symbol_table.types[spec_name] = spec
            arg.annotation.slice.elts[2] = ast.Name(id=spec_name, ctx=ast.Load())

    def finalize_dtype(self, node: ast.AST, target_dtype: AlloType = None):
        """
        Dtypes can be deferred until a concrete type is required.
        This method resolves the final dtype for the given node.
        """
        from allo.utils import np_supported_types, np_dtype_to_allo_dtype

        node_value = getattr(node, "value", None)
        assert node_value is not None, "only support deferred dtype for constants"
        try:
            if target_dtype is None:
                target_dtype = np_dtype_to_allo_dtype[node_value.dtype]
            else:
                node_value = node.value.astype(np_supported_types[str(target_dtype)])
        except:
            raise RuntimeError(f"Fail to finalize dtype for {ast.dump(node)}")
        # untyped global constant
        node.dtype = target_dtype
        # add a new global op
        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
                attr="constant_tensor",
                ctx=ast.Load(),
            ),
            args=[
                node,  # tensor name
                node,  # init value symbol
                self.get_ast_annotaiton(target_dtype, node.shape, None),
            ],
            keywords=[],
        )
        self.symbol_table.global_symbols[node.id] = node_value
        self.symbol_table.global_ops.append(call)
        return node

    def visit_broadcast(
        self, node: ast.AST, dtype: AlloType, target_shape: tuple[int]
    ) -> ast.AST:
        """
        Broadcast an expression to a specific shape. Return the broadcasted expression if broadcast is needed, otherwise return the original expression.
        """
        shape = getattr(node, "shape", None)
        assert shape is not None and len(shape) <= len(target_shape)
        if shape == target_shape:
            return node
        # FIXME: tentative, using -1 as placeholder to distinguish from a dim with size is 1
        padded_shape = [-1] * (len(target_shape) - len(shape)) + list(shape)
        dims = []
        for idx, (s, t) in enumerate(zip(padded_shape, target_shape)):
            if s != t:
                if s != 1 and s != -1:
                    raise ValueError(f"shape mismatch: {shape} vs {target_shape}")
                dims.append(idx)
        # FIXME: currently use linalg.broadcast for lowering, can only 'insert' dim
        assert len(target_shape) - len(shape) == len(dims), "not a semantic constraint"
        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
                attr="broadcast",
                ctx=ast.Load(),
            ),
            args=[
                node,  # original node
                ast.Tuple(
                    elts=[ast.Constant(value=d) for d in dims],
                    ctx=ast.Load(),
                ),  # dims
                self.get_ast_annotaiton(dtype, target_shape, None),  # target type
            ],
            keywords=[],
        )
        call_node.dtype, call_node.shape = dtype, target_shape
        return call_node

    def visit_cast(
        self, node: ast.AST, target_dtype: AlloType, skip_const: bool = False
    ) -> ast.AST:
        assert isinstance(node, ast.expr), "Invalid casting."
        if isinstance(node, ast.Constant):
            if skip_const:  # special case for some const index (e.g., loop args)
                return node
            # constant should be explicitly 'typed', replace the node with builtin constant construction function call
            shape = node.shape
            assert len(shape) == 0, "buildin `constant` op is only for scalalr constant"
            if isinstance(target_dtype, Float):
                node.value = float(node.value)
            else:
                node.value = int(node.value)
            node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
                    attr="constant",
                    ctx=ast.Load(),
                ),
                args=[
                    node,
                    self.get_ast_annotaiton(
                        target_dtype, shape, getattr(node, "spec", None)
                    ),
                ],
                keywords=[],
            )
            node.dtype, node.shape = target_dtype, shape
        if not hasattr(node, "dtype"):
            node = self.finalize_dtype(node, target_dtype)
        if node.dtype == target_dtype:
            return node
        # infer specific handler using CastHandler (abstract class for all cast handlers)
        try:
            # [NOTE] the first two return value is not useful here, we keep them to make `infer`'s interface consistent
            _, _, handler = BUILTIN_HANDLERS["cast"].infer(node.dtype, target_dtype)
        except TypeError as e:
            raise TypeError(f"Cast inference failed: {e}")

        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
                attr=handler,  # Dispatch to specific handler
                ctx=ast.Load(),
            ),
            args=[
                node,
                ast.Name(id=node.dtype.type_hint(), ctx=ast.Load()),
                self.get_ast_annotaiton(
                    target_dtype, node.shape, getattr(node, "spec", None)
                ),
            ],
            keywords=[],
        )
        call_node.dtype = target_dtype
        call_node.shape = node.shape
        return call_node

    def visit_constant(self, node):
        value = self.eval_constant(node)
        if isinstance(value, list):
            value = np.array(value)
        if isinstance(value, np.ndarray):
            name = f"np_arr_{SymbolTable.get_hash(value)}"
            node = self.get_symbol(name, allow_missing=True)
            if node is not None:
                return node
            node = ast.Name(id=name, ctx=ast.Load())  # refer to the global const tensor
            node.shape = value.shape
            node.value = value
            self.put_const(name, node, is_global=True)
            return node
        node = ast.Constant(value)
        node.shape = tuple()
        return node

    def visit_symbol(self, node: ast.expr):
        """
        symbol must be a index
        """
        try:
            const_node = self.visit_constant(node)
            assert isinstance(const_node, ast.Constant) and len(const_node.shape) == 0
            return self.visit_cast(const_node, Index()), sympy.Integer(
                int(const_node.value)
            )
        except:
            pass
        if isinstance(node, ast.Name):
            new_node = self.visit_cast(self.visit(node), Index())
            assert len(new_node.shape) == 0, "Invalid index"
            return new_node, sympy.symbols(node.id)
        if isinstance(node, ast.BinOp):
            lhs, lhs_symbol = self.visit_symbol(node.left)
            rhs, rhs_symbol = self.visit_symbol(node.right)
            new_node = self.visit_binary_op_operands(lhs, rhs, node.op)
            new_node = self.visit_cast(new_node, Index())
            assert len(new_node.shape) == 0, "Invalid index"
            if lhs_symbol and rhs_symbol:
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
                return new_node, op(lhs_symbol, rhs_symbol)
            return new_node, None
        node = self.visit_cast(self.visit(node), Index())
        assert len(node.shape) == 0, "Invalid index"
        return node, None

    def visit_Name(self, node: ast.Name):
        var = self.get_symbol(node.id, allow_missing=True)
        if var is not None:
            if isinstance(var, ast.Constant):
                return var
            if isinstance(var, ast.Name):
                node.id = var.id  # redirect symbol
            if isinstance(var, ast.arg):
                node.id = var.arg  # redirect symbol
            node.dtype, node.shape = var.dtype, var.shape
            return node
        # compile time constant
        return self.visit_constant(node)

    def visit_List(self, node: ast.List):  # constant tensor
        try:
            return self.visit_constant(node)
        except:
            raise RuntimeError("List constant evaluation failed")

    def visit_Constant(self, node: ast.Constant):
        # e.g., 1, 1.0, True, False
        node.shape = tuple()  # dtype unknown
        return node

    def visit_Dict(self, node: ast.Dict):
        raise NotImplementedError

    def visit_Attribute(self, node: ast.Attribute):
        # get work id
        if node.attr == "id":
            assert isinstance(node.value, ast.Name)
            var = self.get_symbol(node.value.id)
            assert isinstance(var, Axes)
            return var.wid
        raise NotImplementedError

    def visit_Subscript(self, node: ast.Subscript):
        # e.g., A[i], A[i, j]
        # slice: A[0:10], A[::1]
        try:
            return self.visit_constant(node)  # parse as compile time constant
        except:
            pass
        value = self.visit(node.value)
        node.value = value
        if len(value.shape) > 0:
            # tensor subscript
            elts = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            new_elts = []
            assert len(elts) <= len(value.shape)
            shape = []
            for idx, elt in enumerate(elts):
                elt_ = self.visit(elt)
                if isinstance(elt_, ast.Slice):  # TODO: use visit_symbol?
                    if elt_.upper is None:
                        elt_.upper = ast.Constant(value.shape[idx])
                    assert isinstance(elt_.lower, ast.Constant)
                    assert isinstance(elt_.upper, ast.Constant)
                    assert isinstance(elt_.step, ast.Constant)
                    size = (elt_.upper.value - elt_.lower.value) // elt_.step.value
                    if size > 0:
                        shape.append(size)
                else:
                    elt_ = self.visit_cast(elt_, Index(), skip_const=True)
                new_elts.append(elt_)
            shape.extend(value.shape[len(elts) :])
            if not hasattr(value, "dtype"):
                value = self.finalize_dtype(value)
            node.dtype, node.shape = value.dtype, tuple(shape)
            if isinstance(node.slice, ast.Tuple):
                node.slice.elts = new_elts
            else:
                node.slice = new_elts[0]
            return node
        # bit operation
        if isinstance(node.slice, ast.Slice):  # slice
            assert node.slice.lower and node.slice.upper
            node.slice.lower, l_symbol = self.visit_symbol(node.slice.lower)
            node.slice.upper, u_symbol = self.visit_symbol(
                ast.BinOp(node.slice.upper, ast.Sub(), ast.Constant(value=1))
            )  # FIXME: this actually break python's convention
            if node.slice.step:
                node.slice.step, s_symbol = self.visit_symbol(node.slice.step)
                assert isinstance(s_symbol, sympy.Integer) and s_symbol == 1
            if (
                l_symbol is not None
                and u_symbol is not None
                and isinstance(u_symbol - l_symbol, sympy.Integer)
            ):
                size = int(u_symbol - l_symbol) + 1
                assert size > 0, "upper bound must be greater than lower bound"
                node.dtype = UInt(size)
            node.shape = tuple()
        else:  # single bit
            node.slice = self.visit_cast(self.visit(node.slice), Index())
            assert len(node.slice.shape) == 0, "Invalid index"
            node.dtype, node.shape = UInt(1), tuple()
        if isinstance(node.ctx, ast.Load):  # get bit or get slice
            call_op = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
                    attr="get_bits",
                    ctx=ast.Load(),
                ),
                args=[node, self.get_ast_annotaiton(node.dtype, node.shape, None)],
                keywords=[],
            )
            call_op.dtype, call_op.shape = node.dtype, node.shape
            return call_op
        return node

    def visit_Slice(self, node: ast.Slice):
        # e.g., A[0:10], A[::-1]
        if node.lower is not None:
            node.lower = self.visit(node.lower)
        else:
            node.lower = ast.Constant(value=0)
        if node.upper is not None:
            node.upper = self.visit(node.upper)
        if node.step is not None:
            node.step = self.visit(node.step)
        else:
            node.step = ast.Constant(value=1)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        # e.g., +x, -x, not x
        node.operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            # +x -> x
            return node.operand
        if isinstance(node.op, ast.USub):
            # -x -> 0 - x
            if isinstance(node.operand, ast.Constant):
                assert isinstance(node.operand.value, (int, float))
                node.operand.value = -node.operand.value
                return node.operand
            return self.visit(ast.BinOp(ast.Constant(value=0), ast.Sub(), node.operand))
        if isinstance(node.op, ast.Not):
            # not x
            if isinstance(node.operand, ast.Constant):
                assert isinstance(node.operand.value, bool)
                node.operand.value = not node.operand.value
                return node.operand
            return self.visit(
                ast.Compare(ast.Constant(value=False), [ast.Eq()], [node.operand])
            )
        raise TypeError(f"Unsupported unary operator: {type(node.op).__name__}")

    def resolve_broadcast_shape(self, shape_a, shape_b):
        """
        Compute the compatible shape specifically for broadcasting from shape_a and shape_b.

        See the broadcasting rules in NumPy
        https://numpy.org/doc/stable/user/basics.broadcasting.html
        When operating on two arrays, NumPy compares their shapes element-wise.
        It starts with the trailing (i.e. rightmost) dimension and works its way left.
        Two dimensions are compatible when
        1. they are equal, or
        2. one of them is 1.
        """
        # Align shapes by prefixing with 1s
        ndim_a, ndim_b = len(shape_a), len(shape_b)
        ndim_res = max(ndim_a, ndim_b)
        aligned_a = (1,) * (ndim_res - ndim_a) + tuple(shape_a)
        aligned_b = (1,) * (ndim_res - ndim_b) + tuple(shape_b)

        res_shape = []
        for da, db in zip(aligned_a, aligned_b):
            if da == db:
                res_shape.append(da)
            elif da == 1:
                res_shape.append(db)
            elif db == 1:
                res_shape.append(da)
            else:
                raise ValueError(
                    f"Operands could not be broadcast together with shapes {shape_a} {shape_b}"
                )
        return tuple(res_shape)

    def visit_binary_op_operands(
        self, left: ast.expr, right: ast.expr, op: ast.operator
    ):
        arg1 = getattr(left, "dtype", getattr(left, "value", None))
        arg2 = getattr(right, "dtype", getattr(right, "value", None))
        try:
            result_type, l_type, r_type, *others = BUILTIN_HANDLERS[
                str(type(op).__name__)
            ].infer(arg1, arg2)
        except TypeError as e:
            raise TypeError(f"Type error in binary operation ({op}): {e}")
        left = self.visit_cast(left, l_type)
        right = self.visit_cast(right, r_type)
        # Broadcasting
        lhs_shape = getattr(left, "shape", tuple())
        rhs_shape = getattr(right, "shape", tuple())
        if lhs_shape != rhs_shape:
            try:
                result_shape = self.resolve_broadcast_shape(lhs_shape, rhs_shape)
            except ValueError as e:
                raise ValueError(f"Broadcasting error in binary operation {op}: {e}")
            left = self.visit_broadcast(left, left.dtype, result_shape)
            right = self.visit_broadcast(right, right.dtype, result_shape)
        else:
            result_shape = lhs_shape
        args = [left, right, self.get_ast_annotaiton(result_type, result_shape, None)]
        for extra in others:
            if isinstance(extra, str):
                args.append(ast.Name(id=extra, ctx=ast.Load()))
            else:
                raise NotImplementedError
        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
                attr=str(type(op).__name__),
                ctx=ast.Load(),
            ),
            args=args,
            keywords=[],
        )
        call_node.dtype, call_node.shape = result_type, result_shape
        return call_node

    def make_assignment(self, target: ast.AST, value: ast.AST) -> ast.AnnAssign:
        target_dtype = getattr(target, "dtype", None)
        target_shape = getattr(target, "shape", None)
        if target_dtype is not None:
            value = self.visit_cast(value, target_dtype)
            value = self.visit_broadcast(value, target_dtype, target_shape)
        target.dtype, target.shape = value.dtype, value.shape
        if isinstance(target, ast.Subscript) and len(target.value.shape) == 0:
            # set bit or set slice
            annotation = self.get_ast_annotaiton(
                target.value.dtype, target.value.shape, None
            )
            set_bits_op = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
                    attr="set_bits",
                    ctx=ast.Load(),
                ),
                args=[target, value, annotation],
                keywords=[],
            )
            target = copy.deepcopy(target.value)
            target.ctx = ast.Store()
            value = set_bits_op
            value.dtype, value.shape = target.dtype, target.shape
        else:
            annotation = self.get_ast_annotaiton(
                target.dtype, target.shape, getattr(target, "spec", None)
            )
        assign_node = ast.AnnAssign(
            target=target,
            annotation=annotation,
            value=value,
            simple=isinstance(target, ast.Name),
        )
        assign_node.dtype, assign_node.shape = value.dtype, value.shape
        return assign_node

    def visit_BinOp(self, node: ast.BinOp):
        # e.g., x + y, x - y, x * y, x / y, x // y, x % y,
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        # costant folding
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            new_node = ast.Constant(value=self.eval_constant(node))
            new_node.shape = tuple()
            return new_node
        return self.visit_binary_op_operands(node.left, node.right, node.op)

    def visit_BoolOp(self, node: ast.BoolOp):
        # e.g., x and y, x or y
        arg_dtypes = []
        new_value = []
        for value in node.values:
            val = self.visit(value)
            arg_dtype = getattr(val, "dtype", getattr(val, "value", None))
            arg_dtypes.append(arg_dtype)
            new_value.append(val)
        node.values = new_value
        for val in node.values:
            val.dtype, val.shape = allo_bool, tuple()
        node.dtype, node.shape = allo_bool, tuple()
        return node

    def visit_Compare(self, node: ast.Compare):
        # e.g., x < y, x <= y, x > y, x >= y, x == y, x != y
        assert len(node.comparators) == 1, "Only support one comparator for now"
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        # costant folding
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            new_node = ast.Constant(value=self.eval_constant(node))
            new_node.shape = tuple()
            return new_node
        return self.visit_binary_op_operands(left, right, node.ops[0])

    def visit_assign_symbol(self, targets: list[ast.Name], symbols: list[ast.Name]):
        assert len(targets) == len(symbols), "Invalid assignment, number mismatch"
        for target, symbol in zip(targets, symbols):
            target_ = self.get_symbol(target.id, allow_missing=True)
            assert target_ is None, "symbol cannot be reassigned."
            self.put_var(target.id, val=symbol)

    def visit_Assign(self, node: ast.Assign):
        # e.g., A[i] = 1
        #       v = 1
        #       i, j = 1, 2
        assert len(node.targets) == 1, "chained assignment not supported"
        targets = (
            node.targets[0].elts
            if isinstance(node.targets[0], ast.Tuple)
            else [node.targets[0]]
        )
        node_list = []
        if isinstance(node.value, ast.Call):
            callee = self.visit(node.value)
            if getattr(callee, "is_symbol", False):  # used to redirect
                callee = [callee]
            if isinstance(callee, list):  # list of symbols
                self.visit_assign_symbol(targets, callee)
                return None
            if len(targets) > 1:
                # function call with multiple returns
                result_name = f"_res_{id(callee)}"  # unique name
                call_node = ast.Assign(
                    targets=[ast.Name(id=result_name, ctx=ast.Store())], value=callee
                )
                self._copy_loc(call_node, node)
                node_list.append(call_node)
                values = []
                res = ast.Name(id=result_name, ctx=ast.Load())
                for i, (dtype, shape) in enumerate(zip(callee.dtype, callee.shape)):
                    value = ast.Subscript(
                        value=res, slice=ast.Constant(i), ctx=ast.Load()
                    )
                    value.dtype, value.shape = dtype, shape  # TODO: spec
                    values.append(value)
            else:
                values = [callee]
        else:
            values = (
                node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
            )
            values = [self.visit(value) for value in values]
        assert len(targets) == len(values)
        # FIXME: this has the potential issue of serializing simultaneous assignment
        for target, rhs in zip(targets, values):
            if getattr(rhs, "is_symbol", False):  # used to redirect
                self.visit_assign_symbol([target], [rhs])
                continue
            if isinstance(target, ast.Name):
                target_ = self.get_symbol(target.id, allow_missing=True)
                if target_ is None:
                    assert getattr(rhs, "dtype", None) is not None
                    assert getattr(rhs, "shape", None) is not None
                    self.put_var(name=target.id, val=target)
                else:
                    assert not getattr(
                        target_.dtype, "constexpr", False
                    ), "Cannot reassign constants."
                    assert not getattr(
                        target_, "immutable", False
                    ), "Cannot reassign scalar arguments"
                    target.dtype, target.shape = target_.dtype, target_.shape
            else:
                # e.g., A[i] = 1
                self.visit(target)
            node_list.append(self.make_assignment(target, rhs))
        # replace with a list of AnnAssign for normalization
        return node_list

    def visit_AugAssign(self, node: ast.AugAssign):
        # e.g., A[i] += 1
        rhs = self.visit(node.value)
        lhs = self.visit(node.target)
        assert not getattr(lhs.dtype, "constexpr", False), "Cannot reassign constants."
        assert not getattr(lhs, "immutable", False), "Cannot reassign scalar arguments"
        left = copy.deepcopy(lhs)
        for n in ast.walk(left):
            if isinstance(n, (ast.Name, ast.Attribute, ast.Subscript)):
                n.ctx = ast.Load()
        value = self.visit_binary_op_operands(left, rhs, node.op)
        return self.make_assignment(lhs, value)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # e.g., C: float32[32, 32] = 0.0
        #       B: int32 = 0
        #       acc: Int(4) @ Stateful = 0
        dtype, shape, spec = self.visit_type_annotation(node.annotation)
        assert isinstance(
            node.target, ast.Name
        ), "target of AnnAssign must be Name, other type not supported."
        target_ = self.get_symbol(node.target.id, allow_missing=True)
        if isinstance(dtype, Stream):
            assert node.value is None, "Invalid stream declaration."
            assert self.current_namespace is not None, "stream must be defined in unit"
            assert self.current_func is None, spec is None
            # replace with a special call
            stream_name = self.symbol_table.mangle_with_namespace(
                node.target.id, self.current_namespace
            )
            stream = ast.Name(id=stream_name, ctx=ast.Store())
            call_op = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
                    attr="constrcut_stream",
                    ctx=ast.Load(),
                ),
                args=[stream, self.get_ast_annotaiton(dtype, shape, None)],
                keywords=[],
            )
            stream.dtype, stream.shape = dtype, shape
            self.put_var(node.target.id, val=stream)
            return ast.Expr(value=call_op)
        if target_ is not None:
            assert (
                node.value is not None
            ), "Unsupported annotated assignment without a value."
            # assignment
            assert (
                target_.dtype == dtype and target_.shape == shape
            ), f"Invalid assignment to {node.target.id}, type mismatch."
            assert not getattr(
                target_.dtype, "constexpr", False
            ), "Cannot reassign constants."
        if getattr(dtype, "constexpr", False):  # TODO: scalar only?
            try:
                node.value = ast.Constant(self.eval_constant(node.value))
                self.put_const(node.target.id, node.value)
                node.value.dtype, node.value.shape = dtype, shape
            except:
                if isinstance(node.value, ast.Attribute):  # is symbol
                    assert isinstance(dtype, Index)
                    self.visit_assign_symbol([node.target], [self.visit(node.value)])
                else:
                    raise RuntimeError("unreachable")
            return None
        else:
            if node.value is not None:
                value = self.visit(node.value)
                if isinstance(value, ast.arg):
                    self.put_var(node.target.id, value)
                    return None
                value = self.visit_cast(value, dtype)
                node.value = self.visit_broadcast(value, dtype, shape)
            self.put_var(node.target.id, node.target)
        node.target.dtype = node.dtype = dtype
        node.target.shape = node.shape = shape
        node.target.spec = node.spec = spec
        node.annotation = self.get_ast_annotaiton(dtype, shape, spec)
        return node

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call):
            node.value = self.visit(node.value)
            return node
        if isinstance(node.value, ast.Constant):
            # comments
            return None
        raise RuntimeError(f"Unsupported expression: {node.value}")

    def set_loop_iter(self, target: ast.Name):
        iter_ = self.get_symbol(target.id, allow_missing=True)
        assert iter_ is None, "Please choose a different name for the loop iterator."
        target.shape, target.dtype = tuple(), Index()
        self.put_var(target.id, target)

    def visit_body(self, stmts):
        new_body = []
        for stmt in stmts:
            res = self.visit(stmt)
            if isinstance(res, list):
                new_body.extend(res)
            elif res is not None:
                new_body.append(res)
        return new_body

    def visit_For(self, node: ast.For):
        # e.g., for i in range(10):
        #       for i in range(0, 10, 2):
        if node.orelse:
            raise RuntimeError("'else' clause for 'for' not supported in Allo kernels")
        if isinstance(node.iter, ast.Call):
            # naive for loop
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                with self.block_scope():
                    self.set_loop_iter(node.target)
                    ivs_ = [
                        self.visit_cast(self.visit(iv), Index(), skip_const=True)
                        for iv in node.iter.args
                    ]
                    if len(ivs_) == 1:
                        ivs_.insert(0, ast.Constant(value=0))
                    if len(ivs_) == 2:
                        ivs_.append(ast.Constant(value=1))
                    node.iter.args = ivs_
                    node.body = self.visit_body(node.body)
                return node
            # builtin for loops
            module = self.resolve_node(node.iter.func)
            assert module.__module__.startswith(
                _INTERFACE_CONFIG.lib
            ), "Invalid for statement"
            attr = module.__name__
            assert attr == "grid" or attr == "reduction", "Unsupported loop type"
            targets = (
                node.target.elts
                if isinstance(node.target, ast.Tuple)
                else [node.target]
            )
            ranges = node.iter.args
            assert len(targets) == len(ranges)
            loops: list[ast.For] = []
            with self.block_scope():
                for target, range_ in zip(targets, ranges):
                    self.set_loop_iter(target)
                    iv_ = self.visit_cast(self.visit(range_), Index(), skip_const=True)
                    ivs_ = [ast.Constant(value=0), iv_, ast.Constant(value=1)]
                    for_node = ast.For(
                        target=target,
                        iter=ast.Call(
                            func=ast.Name("range", ctx=ast.Load()),
                            args=ivs_,
                            keywords=[],
                        ),
                        body=[],
                        orelse=[],
                        type_comment=attr,
                    )
                    self._copy_loc(for_node, node)
                    if len(loops) > 0:
                        loops[-1].body.append(for_node)
                    loops.append(for_node)
                loops[-1].body = self.visit_body(node.body)
            return loops[0]
        raise RuntimeError("Unsupported for loop")

    def visit_While(self, node: ast.While):
        # e.g., while i < 10:
        if node.orelse:
            raise RuntimeError(
                "'else' clause for 'while' not supported in Allo kernels"
            )
        node.test = self.visit_cast(self.visit(node.test), allo_bool)
        assert len(node.test.shape) == 0, "while condition should be a scalar."
        with self.block_scope():
            node.body = self.visit_body(node.body)
        return node

    def visit_If(self, node: ast.If):
        # e.g., if i < 10: ... else: ...
        node.test = self.visit_cast(self.visit(node.test), allo_bool)
        assert len(node.test.shape) == 0, "if condition should be a scalar."
        with self.block_scope():
            node.body = self.visit_body(node.body)
        if len(node.orelse) > 0:
            with self.block_scope():
                node.orelse = self.visit_body(node.orelse)
        return node

    def visit_IfExp(self, node: ast.IfExp):
        # e.g., x if cond else y
        raise NotImplementedError

    def visit_Return(self, node: ast.Return):
        values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
        func_node = self.symbol_table.functions[self.current_func]
        dtypes = (
            func_node.dtype if isinstance(func_node.dtype, list) else [func_node.dtype]
        )
        shapes = (
            func_node.shape if isinstance(func_node.shape, list) else [func_node.shape]
        )
        assert len(values) == len(dtypes) == len(shapes), "Invalid return statement"
        new_values = []
        for value, dtype, shape in zip(values, dtypes, shapes):
            value = self.visit_cast(self.visit(value), dtype)
            value = self.visit_broadcast(value, dtype, shape)
            new_values.append(value)
        if isinstance(node.value, ast.Tuple):
            node.value.elts = new_values
        else:
            node.value = new_values[0]
        return node

    def visit_Pass(self, node: ast.Pass):
        return node

    def visit_With(self, node: ast.With):  # for meta programming in `allo.template`
        assert len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call)
        func = node.items[0].context_expr.func
        module = self.resolve_node(func)
        assert module.__module__.startswith(
            _INTERFACE_CONFIG.meta
        ), "Invalide with statement"
        attr = module.__name__
        # compile time unrolled loop
        if attr == "meta_for":
            with self.block_scope():
                target = node.items[0].optional_vars
                self.set_loop_iter(target)
                try:
                    ivs_ = [
                        self.visit_constant(iv)
                        for iv in node.items[0].context_expr.args
                    ]
                except:
                    raise RuntimeError(
                        "meta_for loop args must be compile time constants"
                    )
                if len(ivs_) == 1:
                    ivs_.insert(0, ast.Constant(value=0))
                if len(ivs_) == 2:
                    ivs_.append(ast.Constant(value=1))
                for_node = ast.For(
                    target=target,
                    iter=ast.Call(
                        func=ast.Name("range", ctx=ast.Load()),
                        args=ivs_,
                        keywords=[],
                    ),
                    body=self.visit_body(node.body),
                    orelse=[],
                    type_comment="unroll",
                )
                return for_node
        raise NotImplementedError

    def visit_call_kernel(self, orig_node: ast.Call, func, instantiate: list = None):
        func: ast.FunctionDef = get_ast(func)
        callee, callee_name = self.visit_function_signature(func, instantiate)
        # arguments TODO: support kwargs and others?
        assert len(orig_node.args) == len(
            callee.args.args
        ), f"Invalid call to {callee_name}, argument number mismatch."
        new_args = []
        for arg, callee_arg in zip(orig_node.args, callee.args.args):
            # TODO: spec?
            arg = self.visit_cast(self.visit(arg), callee_arg.dtype)
            arg = self.visit_broadcast(arg, arg.dtype, callee_arg.shape)
            new_args.append(arg)
        orig_node.args = new_args
        # return value
        if hasattr(callee, "dtype") and hasattr(callee, "shape"):
            orig_node.dtype, orig_node.shape = callee.dtype, callee.shape
        orig_node.func = ast.Name(id=callee_name, ctx=ast.Load())
        return orig_node

    def visit_stream_method_call(self, node: ast.Call, stream: ast.expr):
        attr = node.func.attr
        if isinstance(stream, ast.Name):
            name, indices = stream, []
        elif isinstance(stream, ast.Subscript):
            assert isinstance(stream.value, ast.Name)
            name = stream.value
            indices = (
                stream.slice.elts
                if isinstance(stream.slice, ast.Tuple)
                else [stream.slice]
            )
        else:
            raise RuntimeError("unreachable")
        if attr == "put":
            assert len(node.args) == 1, "invalid stream put"
            # arg type checking
            arg = self.visit_cast(self.visit(node.args[0]), stream.dtype.dtype)
            arg = self.visit_broadcast(arg, arg.dtype, stream.dtype.shape)
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
                    attr=attr,
                    ctx=ast.Load(),
                ),
                args=[name, ast.Tuple(elts=indices, ctx=ast.Load()), arg],
                keywords=[],
            )
        if attr == "get":
            assert len(node.args) == 0, "invalid stream get"
            call_op = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
                    attr=attr,
                    ctx=ast.Load(),
                ),
                args=[
                    name,
                    ast.Tuple(elts=indices, ctx=ast.Load()),
                    self.get_ast_annotaiton(
                        stream.dtype.dtype, stream.dtype.shape, None
                    ),
                ],
                keywords=[],
            )
            call_op.dtype, call_op.shape = stream.dtype.dtype, stream.dtype.shape
            return call_op
        raise NotImplementedError

    def visit_method(self, node: ast.Call):
        obj = node.func.value
        attr: str = node.func.attr
        var = self.visit(node.func.value)
        # stream's methods
        if attr in {"put", "get"}:
            assert isinstance(var.dtype, Stream) and len(var.shape) == 0
            return self.visit_stream_method_call(node, var)
        # data sharding
        if attr == "shard":
            assert isinstance(obj, ast.Name) and self.current_namespace is not None
            arg_name = obj.id
            arg = self.get_symbol(arg_name)
            if isinstance(getattr(arg, "spec", None), Layout):
                raise RuntimeError("Cannot shard the same tensor multiple times.")
            assert isinstance(arg, ast.arg)
            work_def = self.symbol_table.functions[self.current_func]
            assert len(node.args) == 1 and isinstance(node.args[0], ast.List)
            axes = node.args[0].elts
            assert len(axes) == len(arg.shape), "Invalid sharding"
            partitions = []
            for axe in axes:
                if isinstance(axe, ast.Constant) and axe.value is None:
                    partitions.append(Layout.Replicate)
                elif isinstance(axe, ast.Name):
                    axe_value = self.get_symbol(axe.id)
                    assert isinstance(axe_value, Axes)
                    partitions.append(Layout.Shard(axe_value.idx))
                else:
                    raise NotImplementedError
            spec = Layout(partitions)
            # find the argument and update type
            for work_arg in work_def.args.args:
                if work_arg.arg == arg_name:
                    self.reset_type(
                        work_arg,
                        shape=tuple(spec.shard(work_arg.shape, work_def.grid)),
                        spec=spec,
                    )
                    self.put_var(
                        arg_name,
                        ErrorValue(arg_name, "The variable is invalid after sharding."),
                    )
                    # unique symbol, should be used to replace 'references'
                    work_arg.is_symbol = True
                    return work_arg
            raise RuntimeError("unreachable")
        return None

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            # FIXME: tentative!!!
            func = self.resolve_node(node.func)
            if hasattr(func, "__allo_handler__"):
                name = func.__allo_handler__
                # infer type
                arg_types, new_args = [], []
                for arg in node.args:
                    arg_ = self.visit(arg)
                    # FIXME: pass shape and spec
                    arg_types.append(getattr(arg_, "dtype", None))
                    new_args.append(arg_)
                # FIXME: assuming no kwargs for now
                try:
                    result_type, *other_types = BUILTIN_HANDLERS[name].infer(*arg_types)
                except NotImplementedError:
                    raise RuntimeError(f"Custom handler {name} must implement `infer`")

                # FIXME: should support casting and broadcasting
                call_node = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()),
                        attr=name,
                        ctx=ast.Load(),
                    ),
                    args=new_args,
                    keywords=[],
                )
                # FIXME: result dtype, shape?
                return call_node
            else:  # call another source kernel
                return self.visit_call_kernel(node, func)
        if isinstance(node.func, ast.Subscript):  # call an instantiated kernel
            elts = (
                node.func.slice.elts
                if isinstance(node.func.slice, ast.Tuple)
                else [node.func.slice]
            )
            instantiate = [self.eval_constant(e) for e in elts]
            func = self.resolve_node(node.func.value)
            return self.visit_call_kernel(node, func, instantiate)
        if isinstance(node.func, ast.Attribute):
            module = self.resolve_node(node.func)
            if module and module.__module__.startswith(_INTERFACE_CONFIG.spmw):
                attr = module.__name__
                assert attr in self.work_meta, f"{attr} must be in work."
                values = self.work_meta[attr]
                return values  # list of symbols
            # builtin methods
            try:
                ret = self.visit_method(node)
                if ret is not None:
                    return ret
            except:
                pass
        # TODO
        return node

    def visit_function_signature(self, node: ast.FunctionDef, instantiate: list = None):
        # instantiate an instance from template
        if instantiate is not None:
            assert node._type in {FunctionType.KERNEL, FunctionType.UNIT}
            func_name = self.symbol_table.mangle_template_name(node.name, instantiate)
            assert len(getattr(node, "type_params", [])) == len(instantiate)
            node.template_bindings = {}
            for type_var, call_val in zip(node.type_params, instantiate):
                if type_var.bound is not None:
                    raise NotImplementedError
                node.template_bindings[type_var.name] = call_val
        elif node._type in {FunctionType.WORK}:
            assert self.current_namespace is not None, "work must be defined in unit"
            func_name = self.symbol_table.mangle_with_namespace(
                node.name, self.current_namespace
            )
        else:
            func_name = node.name
        if func_name in self.symbol_table.functions:  # function instance visited
            return self.symbol_table.functions[func_name], func_name
        self.symbol_table.functions[func_name] = node  # record function
        if node._type in {FunctionType.KERNEL, FunctionType.UNIT}:
            self.worklist.append(func_name)  # declare only
        node.name = func_name
        symbols_ckpt = self.symbols
        self.symbols = ChainMap(getattr(node, "template_bindings", {}), self.symbols)
        # arguments
        for arg in node.args.args:
            arg.dtype, arg.shape, arg.spec = self.visit_type_annotation(arg.annotation)
            if len(arg.shape) == 0:
                arg.immutable = True  # [NOTE] scalar argument is defined as immutable, we don't allocate buffer for them
            arg.annotation = self.get_ast_annotaiton(arg.dtype, arg.shape, arg.spec)
            assert not getattr(
                arg.dtype, "stateful", False
            ), f"Function parameter '{arg.arg}' cannot be Stateful."
            # FIXME: this assumes functions are under global scope
            assert arg.arg not in self.symbols, (
                f"Argument name '{arg.arg}' conflicts with an existing symbol. "
                "Please choose a different name to avoid the conflict."
            )
        # return type
        if node.returns is not None:
            assert not node._type in {FunctionType.WORK, FunctionType.UNIT}
            if isinstance(node.returns, ast.Tuple):
                # Multiple return values
                node.returns.shape = []
                node.returns.dtype = []
                node.returns.spec = []
                new_elts = []
                for elt in node.returns.elts:
                    elt.dtype, elt.shape, elt.spec = self.visit_type_annotation(elt)
                    node.returns.dtype.append(elt.dtype)
                    node.returns.shape.append(elt.shape)
                    node.returns.spec.append(elt.spec)
                    new_elts.append(
                        self.get_ast_annotaiton(elt.dtype, elt.shape, elt.spec)
                    )
                node.returns.elts = new_elts
            else:
                # Single return value
                dtype, shape, spec = self.visit_type_annotation(node.returns)
                node.returns = self.get_ast_annotaiton(dtype, shape, spec)
                node.returns.dtype = dtype
                node.returns.shape = shape
                node.returns.spec = spec
            node.dtype, node.shape = node.returns.dtype, node.returns.shape
        self.symbols = symbols_ckpt
        return node, func_name

    def visit_function_body(self, node: ast.FunctionDef):
        if node._type in {FunctionType.UNIT}:
            with self.namespace(node):
                # arguments
                for arg in node.args.args:
                    self.put_var(name=arg.arg, val=arg)
                # unit region
                node.body = self.visit_body(node.body)
                node.body.append(ast.Return())
                meta_ops = []
                node.body = meta_ops + node.body
        else:
            with self.function_scope(node):
                # arguments
                for arg in node.args.args:
                    self.put_var(name=arg.arg, val=arg)
                # function body
                new_body = self.visit_body(node.body)
                if node.returns is None and not isinstance(new_body[-1], ast.Return):
                    new_body.append(ast.Return())
                node.body = new_body
        ast.fix_missing_locations(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef, instantiate: list = None):
        assert len(node.decorator_list) == 1 and isinstance(
            node.decorator_list[0], ast.Call
        )
        module = self.resolve_node(node.decorator_list[0].func)
        assert (
            module.__module__.startswith(_INTERFACE_CONFIG.spmw)
            and module.__name__ == "work"
        )
        node._type = FunctionType.WORK
        node._source = self.symbol_table.functions[self.current_namespace]._source
        with self.block_scope():  # to support args shadowing
            # keywords
            grid = []
            for kw in node.decorator_list[0].keywords:
                assert isinstance(kw.value, ast.List)
                if kw.arg == "grid":
                    elts_ = []
                    for c in kw.value.elts:
                        dim = self.visit_constant(c)
                        grid.append(dim.value)
                        elts_.append(dim)
                    kw.value.elts = elts_
                else:
                    raise RuntimeError("Invalid work declaration")
            node.grid = grid
            # 'arguments' from unit
            node.arg_kw = ast.keyword("args", ast.List([]))
            node.decorator_list[0].keywords.append(node.arg_kw)
            node, _ = self.visit_function_signature(node, instantiate=instantiate)
            assert (
                node.returns is None and len(node.args.args) == 0
            ), "Invalid work. work should not have return value or arguments."
            # check arg mapping
            with self.function_scope(node):
                # arguments
                for arg in node.args.args:
                    self.put_var(name=arg.arg, val=arg)
                # internally insert `get_wid` as the first statement (meta data setup)
                wids, axes = [], []
                for i in range(len(grid)):
                    target = ast.Name(id=f"__wid_{i}", ctx=ast.Store())  # reserved word
                    target_ = copy.deepcopy(target)
                    wids.append(target)
                    target_.dtype, target_.shape = Index(), tuple()
                    target_.dtype.constexpr = True  # immutable
                    target_.is_symbol = True
                    target_.ctx = ast.Load()
                    axe = Axes(idx=i, wid=target_)
                    axes.append(axe)
                self.work_meta["axes"] = axes
                op = ast.Assign(
                    targets=[ast.Tuple(elts=wids, ctx=ast.Load())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(
                                id=_INTERFACE_CONFIG.builtin, ctx=ast.Load()
                            ),
                            attr="get_wid",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                )
                self._copy_loc(op, node.body[0])  # function entry
                self.meta_ops.append(op)
                new_body = self.visit_body(node.body)
                new_body.append(ast.Return())
                node.body = self.meta_ops + new_body
                self.work_meta.clear()
        call_node = ast.Call(
            func=ast.Name(id=node.name, ctx=ast.Load()),
            args=node.arg_kw.value.elts,
            keywords=[],
        )
        return ast.Expr(value=call_node)

    # ----- invalid syntax -----
    def visit_Break(self, node: ast.Break):
        raise RuntimeError("Break statement is not supported")

    def visit_Continue(self, node: ast.Continue):
        raise RuntimeError("Continue statement is not supported")
