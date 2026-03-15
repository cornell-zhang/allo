# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class Interface:
    meta: str  # meta programming interface, e.g. the file defineing `meta_for`
    spmw: str  # spmw related interface
    lib: str  # allo's kernel library
    builtin: str = "__allo__"


_TYPING_RULE_CONFIG = "default"  # Global configuration for typing rules
_INTERFACE_CONFIG = Interface(meta="allo.template", spmw="allo.spmw", lib="allo.dsl")


def get_typing_rule_config():
    return _TYPING_RULE_CONFIG


@contextmanager
def ir_builder_config_context(typing_rule_config: str = None):
    """Context manager for setting the IR Builder configuration."""
    global _TYPING_RULE_CONFIG
    old_config = _TYPING_RULE_CONFIG
    if typing_rule_config is not None:
        _TYPING_RULE_CONFIG = typing_rule_config
    try:
        yield
    finally:
        _TYPING_RULE_CONFIG = old_config
