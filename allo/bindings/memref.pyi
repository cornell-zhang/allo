from typing import Sequence
from . import ir

class LoadOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        memref: ir.Value,
        indices: Sequence[ir.Value],
    ) -> LoadOp: ...

class StoreOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        value: ir.Value,
        memref: ir.Value,
        indices: Sequence[ir.Value],
    ) -> StoreOp: ...

class AllocOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, memref_type: ir.MemRefType) -> ir.Value: ...
