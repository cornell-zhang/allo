from typing import Sequence
from . import ir

class BranchOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        dest: ir.Block,
        args: Sequence[ir.Value],
    ) -> None: ...

class CondBranchOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        cond: ir.Value,
        true_dest: ir.Block,
        false_dest: ir.Block,
    ) -> None: ...
