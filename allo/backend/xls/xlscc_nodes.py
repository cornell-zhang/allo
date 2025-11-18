"""XLSCC AST node classes for representing DSLX code structure."""

class XLSCCNode:
    pass


class XLSCCTranslationUnit(XLSCCNode):
    def __init__(self, includes=None, decls=None):
        self.includes = includes or []
        self.decls = decls or []


class XLSCCInclude(XLSCCNode):
    def __init__(self, header, angled=True):
        self.header = header
        self.angled = angled


class XLSCCType(XLSCCNode):
    def __init__(self, name, template_args=None):
        self.name = name
        self.template_args = template_args or []


class XLSCCTemplateParam(XLSCCNode):
    def __init__(self, kind, name, default=None):
        self.kind = kind
        self.name = name
        self.default = default


class XLSCCTemplateAlias(XLSCCNode):
    def __init__(self, template_params, alias_name, aliased_type):
        self.template_params = template_params
        self.alias_name = alias_name
        self.aliased_type = aliased_type


class XLSCCClass(XLSCCNode):
    def __init__(self, name, members=None):
        self.name = name
        self.members = members or []


class XLSCCAccessSpec(XLSCCNode):
    def __init__(self, access):
        self.access = access


class XLSCCParam(XLSCCNode):
    def __init__(self, type_, name):
        self.type = type_
        self.name = name


class XLSCCPragma(XLSCCNode):
    def __init__(self, text):
        self.text = text


class XLSCCMethod(XLSCCNode):
    def __init__(self, name, return_type, params=None, body=None,
                 is_static=False, is_const=False, pragmas=None):
        self.name = name
        self.return_type = return_type
        self.params = params or []
        self.body = body or []
        self.is_static = is_static
        self.is_const = is_const
        self.pragmas = pragmas or []


class XLSCCExpr(XLSCCNode):
    pass


class XLSCCVar(XLSCCExpr):
    def __init__(self, name):
        self.name = name


class XLSCCLiteral(XLSCCExpr):
    def __init__(self, value):
        self.value = value


class XLSCCBinOp(XLSCCExpr):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs


class XLSCCCall(XLSCCExpr):
    def __init__(self, callee, args=None):
        self.callee = callee
        self.args = args or []


class XLSCCStmt(XLSCCNode):
    pass


class XLSCCExprStmt(XLSCCStmt):
    def __init__(self, expr):
        self.expr = expr


class XLSCCReturnStmt(XLSCCStmt):
    def __init__(self, expr):
        self.expr = expr