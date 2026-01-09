"""DSLX Proc AST node classes for representing DSLX proc structures.

This module provides an AST (Abstract Syntax Tree) representation for DSLX procs,
allowing generic construction and serialization of proc definitions.
"""


class DslxProcNode:
    """Base class for DSLX proc AST nodes."""
    pass


class DslxType(DslxProcNode):
    """Represents a DSLX type."""

    def __init__(self, name, params=None):
        self.name = name  # e.g., "F32", "u32", "chan"
        self.params = params or []  # e.g., ["F32", "in"] for chan<F32> in


class DslxImport(DslxProcNode):
    """Represents an import statement."""

    def __init__(self, module_name):
        self.module_name = module_name


class DslxTypeAlias(DslxProcNode):
    """Represents a type alias (e.g., type F32 = float32::F32)."""

    def __init__(self, alias_name, target_type):
        self.alias_name = alias_name
        self.target_type = target_type


class DslxChannelDecl(DslxProcNode):
    """Represents a channel declaration in a proc.

    Examples:
        a_in: chan<F32> in
        activations: chan<F32>[ROWS] in
        from_wests: chan<F32>[COLS + u32:1][ROWS] in
    """

    def __init__(self, name, chan_type, direction, array_dims=None):
        self.name = name  # e.g., "a_in"
        self.chan_type = chan_type  # e.g., "F32"
        self.direction = direction  # "in" or "out"
        self.array_dims = array_dims or []  # e.g., ["ROWS"], ["COLS", "ROWS"]


class DslxParam(DslxProcNode):
    """Represents a parametric value (e.g., K: u32)."""

    def __init__(self, name, param_type):
        self.name = name
        self.param_type = param_type


class DslxExpr(DslxProcNode):
    """Base class for expressions."""
    pass


class DslxLiteral(DslxExpr):
    """Represents a literal value."""

    def __init__(self, value, lit_type=None):
        self.value = value
        self.lit_type = lit_type  # e.g., "u32", "F32"


class DslxVar(DslxExpr):
    """Represents a variable reference."""

    def __init__(self, name):
        self.name = name


class DslxFuncCall(DslxExpr):
    """Represents a function call."""

    def __init__(self, func_name, args):
        self.func_name = func_name
        self.args = args  # List of DslxExpr


class DslxBinOp(DslxExpr):
    """Represents a binary operation."""

    def __init__(self, op, lhs, rhs):
        self.op = op  # "+", "*", "==", etc.
        self.lhs = lhs
        self.rhs = rhs


class DslxTuple(DslxExpr):
    """Represents a tuple expression."""

    def __init__(self, elements):
        self.elements = elements  # List of DslxExpr


class DslxArrayLiteral(DslxExpr):
    """Represents an array literal."""

    def __init__(self, elem_type, elements):
        self.elem_type = elem_type
        self.elements = elements


class DslxArrayIndex(DslxExpr):
    """Represents array indexing: arr[i] or arr[i][j]."""

    def __init__(self, array, indices):
        self.array = array  # DslxVar or DslxExpr
        self.indices = indices  # List of DslxExpr (can be nested)


class DslxStmt(DslxProcNode):
    """Base class for statements."""
    pass


class DslxLet(DslxStmt):
    """Represents a let binding."""

    def __init__(self, pattern, expr):
        self.pattern = pattern  # Can be DslxVar or DslxTuple
        self.expr = expr


class DslxIf(DslxExpr):
    """Represents an if-else expression."""

    def __init__(self, condition, then_expr, else_expr):
        self.condition = condition
        self.then_expr = then_expr
        self.else_expr = else_expr


class DslxChannelOp(DslxExpr):
    """Represents channel operations (recv, send)."""

    def __init__(self, op_type, args):
        self.op_type = op_type  # "recv", "send", "join"
        self.args = args  # List of DslxExpr


class DslxChannelCreate(DslxStmt):
    """Represents channel creation.

    Examples:
        let (s, r) = chan<T>("name")
        let (s, r) = chan<T, u32:1>("name")  # with FIFO depth
        let (to_easts, from_wests) = chan<F32>[COLS + u32:1][ROWS]("east_west")
        let (to_easts, from_wests) = chan<F32, u32:1>[COLS + u32:1][ROWS]("east_west")  # with FIFO
    """

    def __init__(self, sender_name, receiver_name, chan_type, label, array_dims=None, fifo_depth=None):
        self.sender_name = sender_name
        self.receiver_name = receiver_name
        self.chan_type = chan_type
        self.label = label
        self.array_dims = array_dims or []  # e.g., ["COLS + u32:1", "ROWS"]
        self.fifo_depth = fifo_depth  # e.g., "u32:1" for FIFO depth of 1


class DslxSpawn(DslxStmt):
    """Represents a spawn statement."""

    def __init__(self, proc_name, type_params, args):
        self.proc_name = proc_name
        self.type_params = type_params  # List of DslxExpr
        self.args = args  # List of DslxExpr


class DslxUnrollFor(DslxStmt):
    """Represents an unroll_for! loop.

    Example:
        unroll_for! (row, tok): (u32, token) in u32:0..ROWS {
            ...
        }(tok)
    """

    def __init__(self, loop_vars, range_expr, body, init_expr):
        self.loop_vars = loop_vars  # List of (name, type) tuples
        self.range_expr = range_expr  # e.g., "u32:0..ROWS"
        self.body = body  # DslxBlock or list of statements
        self.init_expr = init_expr  # Initial value expression


class DslxConst(DslxStmt):
    """Represents a const declaration.

    Example:
        const ACTIVATIONS_COL = u32:0;
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value  # DslxExpr


class DslxBlock(DslxProcNode):
    """Represents a block of statements."""

    def __init__(self, stmts):
        self.stmts = stmts  # List of DslxStmt


class DslxConfigFunc(DslxProcNode):
    """Represents a proc config function."""

    def __init__(self, params, body):
        self.params = params  # List of (name, type, direction) or (name, type, direction, array_dims) tuples
        self.body = body  # DslxBlock or DslxExpr (typically tuple return)


class DslxInitFunc(DslxProcNode):
    """Represents a proc init function."""

    def __init__(self, init_expr):
        self.init_expr = init_expr  # DslxExpr


class DslxNextFunc(DslxProcNode):
    """Represents a proc next function."""

    def __init__(self, state_type, body):
        self.state_type = state_type  # DslxType or string
        self.body = body  # DslxBlock


class DslxProc(DslxProcNode):
    """Represents a complete proc definition."""

    def __init__(self, name, type_params=None, channels=None, config=None,
                 init=None, next_func=None, is_public=False):
        self.name = name
        self.type_params = type_params or []  # List of DslxParam
        self.channels = channels or []  # List of DslxChannelDecl
        self.config = config  # DslxConfigFunc
        self.init = init  # DslxInitFunc
        self.next_func = next_func  # DslxNextFunc
        self.is_public = is_public


class DslxModule(DslxProcNode):
    """Represents a complete DSLX module."""

    def __init__(self, imports=None, type_aliases=None, procs=None):
        self.imports = imports or []  # List of DslxImport
        self.type_aliases = type_aliases or []  # List of DslxTypeAlias
        self.procs = procs or []  # List of DslxProc
