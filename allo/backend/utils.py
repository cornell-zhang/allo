# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import textwrap
from contextlib import contextmanager

INDENT = -1

def format_str(s, indent=4, strip=True):
    if INDENT != -1:
        # global context
        indent = INDENT
    if strip:
        return textwrap.indent(textwrap.dedent(s).strip("\n"), " " * indent) + "\n"
    return textwrap.indent(textwrap.dedent(s), " " * indent) + "\n"


@contextmanager
def format_code(indent=4):
    global INDENT
    old_indent = INDENT
    try:
        INDENT = indent
        yield
    finally:
        INDENT = old_indent
