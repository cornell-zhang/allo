from typing import Sequence
from . import ir

class BranchOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        arg0: ir.AlloOpBuilder,
        arg1: ir.Block,
        arg2: Sequence[ir.Value],
        /,
    ) -> BranchOp: ...

class CondBranchOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        arg0: ir.AlloOpBuilder,
        arg1: ir.Value,
        arg2: ir.Block,
        arg3: ir.Block,
        /,
    ) -> CondBranchOp: ...
