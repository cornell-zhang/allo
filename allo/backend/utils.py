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


def is_hls_top_definition(line, top):
    """Match the complete generated top function name, not a helper prefix."""
    if not top:
        return False
    signature = line.removeprefix('extern "C" ')
    prefix = f"void {top}"
    if not signature.startswith(prefix):
        return False
    return signature[len(prefix) :].lstrip().startswith("(")


@contextmanager
def format_code(indent=4):
    global INDENT
    old_indent = INDENT
    try:
        INDENT = indent
        yield
    finally:
        INDENT = old_indent
