from typing import Sequence
from enum import Enum
from . import ir

class IteratorType(Enum):
    PAR = "parallel"
    RED = "reduction"

PAR = IteratorType.PAR
RED = IteratorType.RED

class MatmulOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        result: ir.Value,
    ) -> MatmulOp: ...

class FillOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        value: ir.Value,
        output: ir.Value,
    ) -> FillOp: ...

class BroadcastOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        input: ir.Value,
        init: ir.Value,
        dims: Sequence[int],
    ) -> BroadcastOp: ...

class AddOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> AddOp: ...

class MulOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> MulOp: ...

class SubOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> SubOp: ...

class DivOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> DivOp: ...

class DivUnsignedOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> DivUnsignedOp: ...

class PowFOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        base: ir.Value,
        exponent: ir.Value,
        init: ir.Value,
    ) -> PowFOp: ...

class FloorOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> FloorOp: ...

class ExpOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> ExpOp: ...

class LogOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> LogOp: ...

class SqrtOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> SqrtOp: ...

class ReciprocalOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> ReciprocalOp: ...

class RsqrtOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> RsqrtOp: ...

class SquareOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> SquareOp: ...

class DotOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> DotOp: ...

class GenericOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        res_types: Sequence[ir.Type],
        inputs: Sequence[ir.Value],
        outputs: Sequence[ir.Value],
        indexing_maps: Sequence[ir.AffineMap],
        iterator_types: Sequence[IteratorType],
    ) -> GenericOp: ...
    @property
    def body(self) -> ir.Region: ...
    def add_entry_block(self) -> ir.Block: ...

class YieldOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        values: Sequence[ir.Value],
    ) -> YieldOp: ...
