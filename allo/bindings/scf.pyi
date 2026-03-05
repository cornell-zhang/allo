from typing import Sequence
from . import ir

class YieldOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, results: Sequence[ir.Value]) -> YieldOp: ...

class ConditionOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        cond: ir.Value,
        args: Sequence[ir.Value],
    ) -> ConditionOp: ...

class ForOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        lb: ir.Value,
        ub: ir.Value,
        step: ir.Value,
        init_args: Sequence[ir.Value] = [],
    ) -> ForOp: ...
    @property
    def body(self) -> ir.Block: ...
    @property
    def induction_var(self) -> ir.Value: ...

class IfOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        res_types: Sequence[ir.Type],
        cond: ir.Value,
        with_else: bool = False,
    ) -> IfOp: ...
    @property
    def else_block(self) -> ir.Block: ...
    @property
    def else_yield(self) -> YieldOp: ...
    @property
    def then_block(self) -> ir.Block: ...
    @property
    def then_yield(self) -> YieldOp: ...

class WhileOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        arg0: ir.AlloOpBuilder,
        arg1: Sequence[ir.Type],
        arg2: Sequence[ir.Value],
        /,
    ) -> WhileOp: ...
    @property
    def after(self) -> ir.Region: ...
    @property
    def before(self) -> ir.Region: ...

class ParallelOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        lbs: Sequence[ir.Value],
        ubs: Sequence[ir.Value],
        steps: Sequence[ir.Value],
        init_args: Sequence[ir.Value] = [],
    ) -> ParallelOp: ...
    @property
    def body(self) -> ir.Block: ...
    @property
    def induction_vars(self) -> Sequence[ir.Value]: ...

class ReduceOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        operands: Sequence[ir.Value],
    ) -> ReduceOp: ...
