from dataclasses import dataclass
from enum import Enum

from . import ir

IDENTIFIER_ATTR_NAME: str = ...

def build_proxy(mod: ir.ModuleOp) -> OperationProxy: ...
def create_operation_proxy(
    identifier: str, kind: str, hier_name: str
) -> OperationProxy: ...
def parse_from_string(ctx: ir.Context, src: str) -> ir.ModuleOp: ...
def parse_from_file(ctx: ir.Context, filename: str) -> ir.ModuleOp: ...
def complete_hierarchy_name(tree: OperationProxy) -> None: ...
def finalize_transform(
    tree: OperationProxy, identifier_map: dict[str, str] = dict()
) -> None: ...

class ProxyState(Enum):
    VALID = ...
    INVALID = ...
    STALE = ...

VALID = ProxyState.VALID
INVALID = ProxyState.INVALID
STALE = ProxyState.STALE

@dataclass
class FrontendProxy:
    kind_str: str
    hierarchy_name: str
    identifier: str
    state: ProxyState

@dataclass
class OperationProxy(FrontendProxy):
    parent: OperationProxy | None
    children: list[OperationProxy]
    values: list[ValueProxy]

    def __str__(self) -> str: ...
    def add_child(
        self, index: int, id: str, kind: str, hier_name: str
    ) -> OperationProxy: ...
    def add_value(
        self, id: str, kind: str, hier_name: str, number: int = 0
    ) -> ValueProxy: ...
    def splice(self, src: OperationProxy, start: int = 0) -> None: ...

@dataclass
class ValueProxy(FrontendProxy):
    owner: OperationProxy
    number: int
