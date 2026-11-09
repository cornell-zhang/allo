# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend function symbols, independent of specialization identity."""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, replace
from contextlib import contextmanager


@dataclass(frozen=True)
class FunctionName:
    name: str
    definition: int = 0
    instance: int | str | None = None
    coordinates: tuple[int, ...] | None = None
    scope: tuple = ()
    clone: int | None = None


def validate_name(value, identifier=False):
    if not isinstance(value, str) or "__" in value:
        raise ValueError(f"Reserved separator '__' in Allo name {value!r}")
    if identifier and not value.isidentifier():
        raise ValueError(f"Invalid function name {value!r}")
    if not identifier and not re.fullmatch(r"[A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid instance label {value!r}")


def format_function_name(value: FunctionName):
    """Format source, definition, instance, coordinates, and clone, in that order.

    ``__`` separates fields; ``__fn_`` separates enclosing dataflow scopes.
    Automatic IDs are decimal; string IDs use ``id_`` and ASCII alphanumerics
    or single underscores. Empty string IDs are allowed.
    """
    validate_name(value.name, identifier=True)
    result = value.name

    def decimal(number):
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"Invalid function-name counter {number!r}")
        return str(number)

    definition = decimal(value.definition)
    if value.definition:
        result += "__def_" + definition
    if isinstance(value.instance, str):
        validate_name(value.instance)
        result += "__id_" + value.instance
    elif value.instance is not None:
        result += "__" + decimal(value.instance)
    if value.coordinates is not None:
        if not value.coordinates:
            raise ValueError("A kernel needs at least one coordinate")
        result += "__pe_" + "_".join(decimal(number) for number in value.coordinates)
    if value.clone is not None:
        result += "__clone_" + decimal(value.clone)
    if value.scope:
        result = "__fn_".join(
            [*(format_function_name(scope) for scope in value.scope), result]
        )
    return result


def parse_function_name(symbol: str):
    """Parse a canonical frontend symbol, preserving single underscores."""
    # Match markers only after consuming the current field's prefix. In
    # particular, the label in foo__id__fn_x is '_fn_x', not an inner scope.
    marker = re.compile(r"__(?:def_|id_|pe_|clone_|fn_|[0-9])")
    fields = {}
    scopes = []
    end = marker.search(symbol)
    fields["name"] = symbol[: end.start() if end else len(symbol)]
    while end:
        start = end.start() + 2
        if symbol[start].isdigit():
            key, prefix = "instance", ""
        else:
            tag = symbol[start : symbol.index("_", start)]
            key = {"def": "definition", "id": "instance"}.get(tag, tag)
            prefix = tag + "_"
        start += len(prefix)
        end = marker.search(symbol, start)
        text = symbol[start : end.start() if end else len(symbol)]
        if key == "fn":
            scopes.append(FunctionName(**fields))
            fields = {"name": text}
            continue
        if key in fields:
            raise ValueError(f"Duplicate field in function name {symbol!r}")
        if prefix == "id_":
            fields[key] = text
        elif key == "pe":
            fields["coordinates"] = tuple(int(number) for number in text.split("_"))
        else:
            fields[key] = int(text)
    value = FunctionName(**fields, scope=tuple(scopes))
    if format_function_name(value) != symbol:
        raise ValueError(f"Noncanonical function name {symbol!r}")
    return value


def definition_origin(node):
    return getattr(node, "allo_origin", (node.name, node.lineno, node.col_offset))


def fresh_function_name(occupied, name):
    """Allocate a new lowering definition in an existing module namespace."""
    definitions = set()
    for symbol in occupied:
        try:
            value = parse_function_name(symbol)
        except ValueError:
            continue  # Backend/external symbols have their own contracts.
        if value.name == name:
            definitions.add(value.definition)
    definition = 0
    while definition in definitions:
        definition += 1
    return format_function_name(FunctionName(name, definition, 0))


def origin_text(origin):
    if inspect.isfunction(origin):
        code = origin.__code__
        return f"{code.co_filename}:{code.co_firstlineno}:{origin.__qualname__}:{id(origin)}"
    return repr(origin)


class NameRegistry:
    def __init__(self):
        self.definitions = {}
        self.instances = {}
        self.inferred = {}
        self.built = {}

    def base(self, name, origin):
        validate_name(name, identifier=True)
        definitions = self.definitions.setdefault(name, [])
        if origin not in definitions:
            definitions.append(origin)
        return FunctionName(name, definitions.index(origin))

    def instance(self, base, parameters, parent, label=None):
        origin = self.definitions[base.name][base.definition]
        parameters = parameter_key(parameters)
        records = self.instances.setdefault(origin, [])
        scope = parent.scope if parent else ()
        if parent and parent.coordinates is not None:
            scope = (*scope, replace(parent, scope=()))
        if label is None and parent:
            for part in reversed((*parent.scope, parent)):
                if isinstance(part.instance, str):
                    label = part.instance
                    break
        if label is not None:
            validate_name(label)
            key = ("label", label, scope)
        else:
            key = ("auto", parameters, parent)
        for existing_key, args, value in records:
            if key == existing_key:
                if parameters != args:
                    raise ValueError(
                        f"Instance {format_function_name(value)} has conflicting parameters"
                    )
                return value
        index = sum(key[0] == "auto" for key, _, _ in records)
        value = replace(base, instance=index if label is None else label, scope=scope)
        records.append((key, parameters, value))
        return value


def parameter_key(value):
    """Specializations distinguish Python values with different types."""
    if isinstance(value, (list, tuple)):
        return type(value), tuple(parameter_key(item) for item in value)
    if isinstance(value, dict):
        return type(value), tuple(
            (parameter_key(k), parameter_key(v)) for k, v in value.items()
        )
    return type(value), value


def defer_definition(ctx, node):
    """Bind lexical definitions now; instantiate their bodies only when called."""
    if (
        ctx.top_func is None
        or ctx.callee_name is not None
        or getattr(node, "allo_generated", False)
    ):
        return False
    if any(is_kernel(d) for d in node.decorator_list):
        return False
    for name in (node.name, *(arg.arg for arg in node.args.args)):
        if ctx.get_symbol(name, allow_missing=True) is not None:
            raise ValueError(
                f"Nested function symbol {name!r} conflicts with an existing symbol"
            )
    node.allo_captured = ctx.lexical_parameters
    node.allo_bindings = ctx.lexical_bindings.copy()
    node.allo_environment = SourceEnvironment(ctx.global_vars)
    ctx.global_vars[node.name] = node
    ctx.put_symbol(name=node.name, val=node)
    return True


class SourceEnvironment:
    """Lexical bindings survive AST copies without copying modules/callables."""

    def __init__(self, values):
        self.values = values

    def __deepcopy__(self, _memo):
        return self


def function_environment(ctx, obj):
    values = (ctx if isinstance(ctx, dict) else ctx.global_vars).copy()
    if inspect.isfunction(obj):
        if not isinstance(ctx, dict):
            for name in ctx.lexical_bindings:
                values.pop(name, None)
        obj = inspect.unwrap(obj)
        values.update(obj.__globals__)
        if obj.__closure__:
            values.update(inspect.getclosurevars(obj).nonlocals)
    elif hasattr(obj, "allo_environment"):
        values.update(obj.allo_environment.values)
        values.update(obj.allo_bindings)
    return values


def prepare_function(ctx, node):
    """Resolve a definition; generated kernel symbols already carry their scope."""
    if getattr(node, "allo_generated", False):
        value = parse_function_name(node.name)
    else:
        origin = definition_origin(node)
        value = ctx.names.base(node.name, origin)
        if ctx.callee_name is not None:
            value = ctx.callee_name
    ctx.function = value
    ctx.origin = definition_origin(node)
    ctx.callee_name = None
    ctx.lexical_parameters = (*getattr(node, "allo_captured", ()), *(ctx.inst or []))
    ctx.lexical_bindings = getattr(node, "allo_bindings", {}).copy()
    ctx.lexical_bindings.update(
        (param.name, value) for param, value in zip(node.type_params, ctx.inst or [])
    )
    return format_function_name(value)


def resolve_instance(ctx, obj, node, parameters=None):
    if isinstance(obj, ast.FunctionDef):
        origin, name = definition_origin(obj), obj.name
    else:
        origin, name = obj, obj.__name__
    base = ctx.names.base(name, origin)
    if parameters is None:
        type_params = getattr(obj, "__type_params__", getattr(obj, "type_params", ()))
        if type_params:
            try:
                parameters = [
                    ctx.global_vars[getattr(param, "name", None) or param.__name__]
                    for param in type_params
                ]
            except KeyError:
                if len(ctx.inst or []) != len(type_params):
                    raise ValueError(f"Missing generic arguments for {name}") from None
                parameters = ctx.inst
    args = list(parameters or [])
    labels = [arg for arg in args if isinstance(arg, str)]
    if len(labels) > 1:
        raise ValueError("A function call accepts at most one instance label")
    label = labels[0] if labels else None
    args = [arg for arg in args if not isinstance(arg, str)]
    parent = ctx.function
    contextual = parent and (
        parent.instance is not None or parent.coordinates is not None or parent.scope
    )
    if parameters is not None or contextual or getattr(obj, "allo_captured", ()):
        specialization = (*args, *getattr(obj, "allo_captured", ()))
        value = ctx.names.instance(base, specialization, parent, label)
    else:
        value = base
    node.allo_parameters = args
    return value


def kernel_prefix(ctx, node):
    value = ctx.names.base(node.name, definition_origin(node))
    parent = ctx.function
    if parent and (
        parent.instance is not None or parent.coordinates is not None or parent.scope
    ):
        value = replace(value, scope=(*parent.scope, replace(parent, scope=())))
    return format_function_name(value)


def is_kernel(node):
    return isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "kernel"


def kernel_arguments(ctx, decorator):
    keywords = {kw.arg: kw.value for kw in decorator.keywords}
    assert "mapping" in keywords, "Missing kernel mapping"
    args = keywords.get("args", ast.List(elts=[]))
    assert isinstance(args, ast.List), "Kernel args must be a list"
    # pylint: disable=eval-used
    return eval(ast.unparse(keywords["mapping"]), ctx.global_vars), args.elts


def kernel_context(ctx, coordinates):
    context = ctx.copy()
    context.rank = coordinates
    context.scopes = ctx.scopes
    context.global_vars.update(
        (f"df.p{axis}", value) for axis, value in enumerate(coordinates)
    )
    return context


@contextmanager
def meta_iteration(ctx, node, var, value):
    previous = ctx.lexical_parameters, ctx.lexical_bindings
    ctx.lexical_parameters = (
        *ctx.lexical_parameters,
        ("meta", node.lineno, node.col_offset, var, value),
    )
    ctx.lexical_bindings = {**ctx.lexical_bindings, var: value}
    try:
        yield
    finally:
        ctx.lexical_parameters, ctx.lexical_bindings = previous


def callsite(ctx, node):
    path = tuple(
        value
        for value in ctx.lexical_parameters
        if isinstance(value, tuple) and value[:1] == ("meta",)
    )
    return f"{node.lineno}:{node.col_offset}:{path!r}"
