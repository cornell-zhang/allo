from typing import Sequence
from enum import Enum
from . import ir

class IteratorType(Enum):
    PAR = "parallel"
    RED = "reduction"

PAR = IteratorType.PAR
RED = IteratorType.RED

class MatmulOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        result: ir.Value,
    ) -> None: ...

class FillOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        value: ir.Value,
        output: ir.Value,
    ) -> None: ...

class BroadcastOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        input: ir.Value,
        init: ir.Value,
        dims: Sequence[int],
    ) -> None: ...

class AddOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> None: ...

class MulOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> None: ...

class SubOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> None: ...

class DivOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> None: ...

class DivUnsignedOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> None: ...

class PowFOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        base: ir.Value,
        exponent: ir.Value,
        init: ir.Value,
    ) -> None: ...

class FloorOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> None: ...

class ExpOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> None: ...

class LogOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> None: ...

class SqrtOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> None: ...

class ReciprocalOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> None: ...

class RsqrtOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> None: ...

class SquareOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        val: ir.Value,
        init: ir.Value,
    ) -> None: ...

class DotOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
    ) -> None: ...

class GenericOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        result_types: Sequence[ir.Type],
        inputs: Sequence[ir.Value],
        outputs: Sequence[ir.Value],
        indexing_maps: Sequence[ir.AffineMap],
        iterator_types: Sequence[IteratorType],
    ) -> None: ...
    def get_body(self) -> ir.Block: ...
    def add_entry_block(self) -> ir.Block: ...

class YieldOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        values: Sequence[ir.Value],
    ) -> None: ...
