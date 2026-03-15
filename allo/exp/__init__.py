# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
from typing import Union
from collections.abc import Callable
from .config import ir_builder_config_context
from .utils import SymbolTable, get_global_vars
from .ast_preprocessor import ASTPreProcessor
from .ir_builder import IRBuilder
from allo.backend.llvm import LLVMModule
from allo.backend.hls import HLSModule
from allo._mlir.dialects import allo as allo_d, func as func_d
from allo._mlir.passmanager import PassManager as mlir_pass_manager


def build(
    fn: Union[Callable, str],
    instantiate: list = None,
    typing: str = None,
    verbose: bool = False,
):
    typing = "default" if typing is None else typing
    with ir_builder_config_context(typing):
        symbol_table = SymbolTable()
        ast_processor = ASTPreProcessor(
            symbol_table, global_symbols=get_global_vars(fn)
        )
        # process the top function
        node, top_name = ast_processor.process(fn, instantiate=instantiate)
        if verbose:
            for name, constant in symbol_table.constants.items():
                print(name, "=", constant.value)
            for op in symbol_table.global_ops:
                print(ast.unparse(op))
            for node in symbol_table.functions.values():
                print(ast.unparse(node), "\n")
            print()
        builder = IRBuilder(symbol_table)
        module = builder.build()
        return module, top_name


def process(fn: Union[Callable, str], instantiate: list = None, typing: str = None):
    """
    Compile the input function.
    """
    module, top_name = build(fn, instantiate, typing)
    return LLVMModule(module, top_name)


def process_spmw(fn: Union[Callable, str], instantiate: list = None):
    """
    Compile the input function in SPMW model.
    """
    module, top_name = build(fn, instantiate)
    return module, top_name
