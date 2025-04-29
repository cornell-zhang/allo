# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=global-statement

import textwrap
from contextlib import contextmanager

INDENT = -1


def format_str(s, indent=4, strip=True):
    if INDENT != -1:
        # global context
        indent = INDENT
    if strip:
        return textwrap.indent(textwrap.dedent(s).strip(), " " * indent) + "\n"
    return textwrap.indent(s, " " * indent) + "\n"


@contextmanager
def format_code(indent=4):
    global INDENT
    old_indent = INDENT
    try:
        INDENT = indent
        yield
    finally:
        INDENT = old_indent

def dfs_print(op, indent=0):
    op_name = str(op.name)
    if '.' in op_name:
        dialect = op_name.split('.')[0]
    else:
        dialect = "(x)"
    
    print('  ' * indent + f"Operation: {op_name},\tDialect: {dialect}")

    for region in op.regions:
        for block in region.blocks:
            for child_op in block.operations:
                dfs_print(child_op, indent + 1)

def print_module(module):
    dfs_print(module.operation)