"""DSLX AST node classes for representing DSLX code structure."""


class DslxNode:
    pass


class DslxVar(DslxNode):
    def __init__(self, name):
        self.name = name


class DslxConst(DslxNode):
    def __init__(self, value, bits=32):
        self.value = value
        self.bits = bits


class DslxBinOp(DslxNode):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs


class DslxLoad(DslxNode):
    def __init__(self, buffer_name, index_expr):
        self.buffer_name = buffer_name
        self.index_expr = index_expr


class DslxStore(DslxNode):
    def __init__(self, buffer_name, index_expr, value_expr):
        self.buffer_name = buffer_name
        self.index_expr = index_expr
        self.value_expr = value_expr


class DslxFor(DslxNode):
    def __init__(self, iter_name, lb, ub, body, accum_vars=None):
        self.iter_name = iter_name
        self.lb = lb
        self.ub = ub
        self.body = body
        self.accum_vars = accum_vars or []


class DslxLet(DslxNode):
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr


class DslxArrayInit(DslxNode):
    def __init__(self, elem_expr, shape):
        self.elem_expr = elem_expr
        self.shape = shape


class DslxFunction(DslxNode):
    def __init__(self, name, params, return_type, body):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
