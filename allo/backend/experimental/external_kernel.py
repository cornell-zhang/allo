# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import os
from pyparsing import Keyword, Literal, nestedExpr, originalTextFor
from ..ip import parse_cpp_function


def extract_extern_C_blocks(code: str) -> list[str]:
    """
    Scan `code` for all occurrences of:
       extern "C" { ... }
    (properly handling nested braces) and return them as raw strings.
    """
    extern_kw = Keyword("extern")
    c_literal = Literal('"C"')
    brace_group = nestedExpr("{", "}")
    extern_block = originalTextFor(extern_kw + c_literal + brace_group)

    blocks = []
    # scanString yields every non-overlapping match
    for tokens, _, _ in extern_block.scanString(code):
        # tokens[0] is the full matched text
        blocks.append(tokens[0])
    return blocks


class ExternalModule:
    """
    User defined external kernel for aie
    """

    def __init__(
        self, top: str, impl_path: str, input_idx: list[int], output_idx: list[int]
    ):
        self.top = top  # identifier
        self.impl_path = impl_path
        self.filename = os.path.basename(impl_path)
        assert self.filename.endswith(
            ".cc"
        ), f"Expected a .cc file, but got: {self.filename}"

        self.input_idx = input_idx
        self.output_idx = output_idx

        with open(self.impl_path, "r", encoding="utf-8") as f:
            code = f.read()
            extern_C_blocks = extract_extern_C_blocks(code)
            all_functions = []
            for block in extern_C_blocks:
                func_pattern = rf"\b[\w\s\[\]<>,:*&]+?\b{self.top}\s*\([^)]*\)\s*{{"
                functions = re.findall(func_pattern, block)
                all_functions.extend(functions)
            assert len(all_functions) == 1, "invalid external function"
            self.args = parse_cpp_function(all_functions[0], self.top)
        assert (self.args is not None) or len(self.args) != len(self.input_idx) + len(
            self.output_idx
        ), f"Failed to parse {self.impl}"
