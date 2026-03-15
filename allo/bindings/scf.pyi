from typing import Sequence
from . import ir

class YieldOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, results: Sequence[ir.Value]
    ) -> None: ...

class ConditionOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        cond: ir.Value,
        args: Sequence[ir.Value],
    ) -> None: ...

class ForOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        lb: ir.Value,
        ub: ir.Value,
        step: ir.Value,
        init_args: Sequence[ir.Value] = [],
    ) -> None: ...
    def get_body(self) -> ir.Block: ...
    def get_induction_var(self) -> ir.Value: ...

class IfOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        res_types: Sequence[ir.Type],
        cond: ir.Value,
        with_else: bool = False,
    ) -> None: ...
    def get_else_block(self) -> ir.Block: ...
    def get_else_yield(self) -> YieldOp: ...
    def get_then_block(self) -> ir.Block: ...
    def get_then_yield(self) -> YieldOp: ...

class WhileOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        result_types: Sequence[ir.Type],
        operands: Sequence[ir.Value],
    ) -> None: ...
    def get_after(self) -> ir.Region: ...
    def get_before(self) -> ir.Region: ...

class ParallelOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        lbs: Sequence[ir.Value],
        ubs: Sequence[ir.Value],
        steps: Sequence[ir.Value],
        init_args: Sequence[ir.Value] = [],
    ) -> None: ...
    def get_body(self) -> ir.Block: ...
    def get_induction_vars(self) -> list[ir.Value]: ...
