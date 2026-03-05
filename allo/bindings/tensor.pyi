from typing import Sequence, overload
from . import ir

class ExtractOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        tensor: ir.Value,
        indices: Sequence[ir.Value],
    ) -> ir.Value: ...

class InsertOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        value: ir.Value,
        tensor: ir.Value,
        indices: Sequence[ir.Value],
    ) -> ir.Value: ...

class SplatOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        value: ir.Value,
        shape: Sequence[int],
    ) -> ir.Value: ...

class CastOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, input: ir.Value, type: ir.Type
    ) -> ir.Value: ...

class EmptyOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @overload
    @staticmethod
    def create(builder: ir.AlloOpBuilder, type: ir.Type) -> ir.Value: ...
    @overload
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        shape: Sequence[int],
        elem_type: ir.Type,
    ) -> ir.Value: ...
