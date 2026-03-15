# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from enum import Enum, auto


class FunctionType(Enum):
    KERNEL = auto()
    UNIT = auto()
    WORK = auto()


MODULE_CACHE = {}


def find_function_ast(fn):
    filename = fn.__code__.co_filename
    lineno = fn.__code__.co_firstlineno + 1
    if filename not in MODULE_CACHE:
        src = Path(filename).read_text()
        MODULE_CACHE[filename] = ast.parse(src)
    tree = MODULE_CACHE[filename]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.lineno == lineno:
            return node
    raise RuntimeError("Function AST not found")


def _annotate_function(fn, ftype: FunctionType):
    node = find_function_ast(fn)
    fn._source = fn.__code__.co_filename
    fn._ast = node
    fn._type = ftype
    return fn


def kernel(fn):
    return _annotate_function(fn, FunctionType.KERNEL)


def unit():
    def decorator(fn):
        return _annotate_function(fn, FunctionType.UNIT)

    return decorator


def work(*, grid: list[int]):
    def decorator(fn):
        return _annotate_function(fn, FunctionType.WORK)

    return decorator


def axes():
    """
    Get a tuple of axes of the work grid. The result must be unpacked.
    """
    raise NotImplementedError("This function should be called in a work.")
